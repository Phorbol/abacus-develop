#include "source_main/driver.h"

#include "source_main/ipi_socket.h"
#include "source_base/global_function.h"
#include "source_base/mathzone.h"
#include "source_cell/check_atomic_stru.h"
#include "source_cell/update_cell.h"
#include "source_esolver/esolver.h"
#include "source_io/module_json/para_json.h"
#include "source_io/module_output/print_info.h"
#include "source_io/module_parameter/parameter.h"

#ifdef __MPI
#include <mpi.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
constexpr double RY_TO_HARTREE = 0.5;
constexpr int IPI_RANK_ROOT = 0;

bool is_root(const int my_rank)
{
    return my_rank == IPI_RANK_ROOT;
}

void bcast_int(int& value)
{
#ifdef __MPI
    MPI_Bcast(&value, 1, MPI_INT, IPI_RANK_ROOT, MPI_COMM_WORLD);
#else
    (void)value;
#endif
}

void bcast_double_vector(std::vector<double>& values)
{
#ifdef __MPI
    MPI_Bcast(values.data(), static_cast<int>(values.size()), MPI_DOUBLE, IPI_RANK_ROOT, MPI_COMM_WORLD);
#else
    (void)values;
#endif
}

std::string bcast_string(std::string value, const int my_rank)
{
    int nbytes = static_cast<int>(value.size());
    bcast_int(nbytes);
    if (nbytes < 0)
    {
        throw std::runtime_error("negative string length in i-PI broadcast");
    }
    if (!is_root(my_rank))
    {
        value.assign(static_cast<std::size_t>(nbytes), '\0');
    }
#ifdef __MPI
    if (nbytes > 0)
    {
        MPI_Bcast(&value[0], nbytes, MPI_CHAR, IPI_RANK_ROOT, MPI_COMM_WORLD);
    }
#endif
    return value;
}

void throw_if_root_io_failed(int root_failed, const std::string& root_message, const int my_rank)
{
    bcast_int(root_failed);
    const std::string message = bcast_string(root_message, my_rank);
    if (root_failed != 0)
    {
        throw std::runtime_error(message.empty() ? "i-PI socket I/O failed" : message);
    }
}

std::string bcast_header(const std::string& root_header, const int my_rank)
{
    char buffer[13] = {' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '\0'};
    if (is_root(my_rank))
    {
        const std::size_t n = root_header.size() > 12 ? 12 : root_header.size();
        for (std::size_t i = 0; i < n; ++i)
        {
            buffer[i] = root_header[i];
        }
    }
#ifdef __MPI
    MPI_Bcast(buffer, 12, MPI_CHAR, IPI_RANK_ROOT, MPI_COMM_WORLD);
#endif
    std::string out(buffer, 12);
    while (!out.empty() && out.back() == ' ')
    {
        out.pop_back();
    }
    return out;
}

std::string ipi_address()
{
    const char* env = std::getenv("ABACUS_IPI_ADDRESS");
    if (env == nullptr || std::string(env).empty())
    {
        return "localhost:31415";
    }
    return std::string(env);
}

std::vector<double> ipi_cell_bohr_from_unitcell(const UnitCell& ucell)
{
    const double lat0 = ucell.lat0;
    // ASE/i-PI sends POSDATA cell as cell.T in C order. ABACUS stores
    // lattice vectors as rows in latvec, so use the transposed order here.
    return {
        ucell.latvec.e11 * lat0, ucell.latvec.e21 * lat0, ucell.latvec.e31 * lat0,
        ucell.latvec.e12 * lat0, ucell.latvec.e22 * lat0, ucell.latvec.e32 * lat0,
        ucell.latvec.e13 * lat0, ucell.latvec.e23 * lat0, ucell.latvec.e33 * lat0,
    };
}

double max_abs_delta(const std::vector<double>& a, const std::vector<double>& b)
{
    if (a.size() != b.size())
    {
        return 1.0e99;
    }
    double out = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        out = std::max(out, std::abs(a[i] - b[i]));
    }
    return out;
}

void set_positions_from_ipi_bohr(UnitCell& ucell, const std::vector<double>& positions_bohr)
{
    if (positions_bohr.size() != static_cast<std::size_t>(3 * ucell.nat))
    {
        throw std::runtime_error("POSDATA atom count does not match STRU.");
    }
    int iat = 0;
    for (int it = 0; it < ucell.ntype; ++it)
    {
        Atom* atom = &ucell.atoms[it];
        for (int ia = 0; ia < atom->na; ++ia)
        {
            const double tau_x = positions_bohr[3 * iat + 0] / ucell.lat0;
            const double tau_y = positions_bohr[3 * iat + 1] / ucell.lat0;
            const double tau_z = positions_bohr[3 * iat + 2] / ucell.lat0;

            double dx = 0.0;
            double dy = 0.0;
            double dz = 0.0;
            ModuleBase::Mathzone::Cartesian_to_Direct(tau_x,
                                                      tau_y,
                                                      tau_z,
                                                      ucell.latvec.e11,
                                                      ucell.latvec.e12,
                                                      ucell.latvec.e13,
                                                      ucell.latvec.e21,
                                                      ucell.latvec.e22,
                                                      ucell.latvec.e23,
                                                      ucell.latvec.e31,
                                                      ucell.latvec.e32,
                                                      ucell.latvec.e33,
                                                      dx,
                                                      dy,
                                                      dz);

            atom->dis[ia].x = dx - atom->taud[ia].x;
            atom->dis[ia].y = dy - atom->taud[ia].y;
            atom->dis[ia].z = dz - atom->taud[ia].z;
            atom->taud[ia].x = dx;
            atom->taud[ia].y = dy;
            atom->taud[ia].z = dz;
            atom->tau[ia].x = tau_x;
            atom->tau[ia].y = tau_y;
            atom->tau[ia].z = tau_z;
            ++iat;
        }
    }
    unitcell::periodic_boundary_adjustment(ucell.atoms, ucell.latvec, ucell.ntype);
    ucell.ionic_position_updated = true;
    ucell.cell_parameter_updated = false;
}

std::vector<double> flatten_forces_hartree_per_bohr(const ModuleBase::matrix& force)
{
    std::vector<double> out(static_cast<std::size_t>(force.nr * force.nc));
    for (int iat = 0; iat < force.nr; ++iat)
    {
        for (int idir = 0; idir < force.nc; ++idir)
        {
            out[static_cast<std::size_t>(3 * iat + idir)] = force(iat, idir) * RY_TO_HARTREE;
        }
    }
    return out;
}

} // namespace

void Driver::driver_ipi_run(const Parameter& param, const int my_rank, std::ofstream& running_log)
{
    ModuleBase::TITLE("Driver", "driver_ipi_run");

    // "socket" is an outer driver mode. The KS/LCAO ESolver internals use the
    // standard SCF code path for each POSDATA request from the i-PI protocol.
    Input_para socket_input = param.inp;
    socket_input.calculation = "scf";

    UnitCell ucell;
    ucell.setup(socket_input.latname, socket_input.ntype, socket_input.lmaxmax, socket_input.init_vel, socket_input.fixed_axes);
    ucell.setup_cell(param.globalv.global_in_stru, running_log);
    unitcell::check_atomic_stru(ucell, socket_input.min_dist_coef);

    IpiSocket socket;
    std::unique_ptr<ModuleESolver::ESolver> p_esolver;
    bool hardware_initialized = false;
    bool esolver_ready = false;
    bool runner_completed = false;
    std::string pending_error;

    try
    {
        this->init_hardware();
        hardware_initialized = true;

        p_esolver.reset(ModuleESolver::init_esolver(socket_input, ucell));
        p_esolver->before_all_runners(ucell, socket_input);
        esolver_ready = true;

#ifdef __RAPIDJSON
        Json::gen_stru_wrapper(&ucell);
#endif

        int io_failed = 0;
        std::string io_message;
        if (is_root(my_rank))
        {
            try
            {
                const std::string address = ipi_address();
                running_log << " ABACUS socket driver connecting to i-PI endpoint " << address << std::endl;
                socket.connect(address);
            }
            catch (const std::exception& exc)
            {
                io_failed = 1;
                io_message = exc.what();
            }
        }
        throw_if_root_io_failed(io_failed, io_message, my_rank);

        bool isinit = false;
        bool hasdata = false;
        int istep = 0;
        int nat_return = ucell.nat;
        double energy_hartree = 0.0;
        std::vector<double> forces_hartree_bohr(static_cast<std::size_t>(3 * ucell.nat), 0.0);
        std::vector<double> virial_hartree(9, 0.0);

        const std::vector<double> reference_cell = ipi_cell_bohr_from_unitcell(ucell);

        while (true)
        {
            std::string header;
            io_failed = 0;
            io_message.clear();
            if (is_root(my_rank))
            {
                try
                {
                    header = socket.read_header();
                }
                catch (const std::exception& exc)
                {
                    io_failed = 1;
                    io_message = exc.what();
                }
            }
            throw_if_root_io_failed(io_failed, io_message, my_rank);
            header = bcast_header(header, my_rank);

            if (header == "STATUS")
            {
                io_failed = 0;
                io_message.clear();
                if (is_root(my_rank))
                {
                    try
                    {
                        if (hasdata)
                        {
                            socket.write_header("HAVEDATA");
                        }
                        else if (isinit)
                        {
                            socket.write_header("READY");
                        }
                        else
                        {
                            socket.write_header("NEEDINIT");
                        }
                    }
                    catch (const std::exception& exc)
                    {
                        io_failed = 1;
                        io_message = exc.what();
                    }
                }
                throw_if_root_io_failed(io_failed, io_message, my_rank);
            }
            else if (header == "INIT")
            {
                int rid = 0;
                int nbytes = 0;
                std::string params;
                io_failed = 0;
                io_message.clear();
                if (is_root(my_rank))
                {
                    try
                    {
                        rid = socket.read_int();
                        nbytes = socket.read_int();
                        if (nbytes < 0)
                        {
                            throw std::runtime_error("negative INIT payload length from i-PI socket");
                        }
                        if (nbytes > 0)
                        {
                            params = socket.read_string(static_cast<std::size_t>(nbytes));
                        }
                    }
                    catch (const std::exception& exc)
                    {
                        io_failed = 1;
                        io_message = exc.what();
                    }
                }
                throw_if_root_io_failed(io_failed, io_message, my_rank);
                bcast_int(rid);
                bcast_int(nbytes);
                if (nbytes > 0 && is_root(my_rank))
                {
                    running_log << " ABACUS socket INIT params " << params << std::endl;
                }
                isinit = true;
                if (is_root(my_rank))
                {
                    running_log << " ABACUS socket INIT replica " << rid << std::endl;
                }
            }
            else if (header == "POSDATA")
            {
                std::vector<double> cell(9, 0.0);
                std::vector<double> inv_cell(9, 0.0);
                int nat_socket = 0;
                std::vector<double> positions;
                io_failed = 0;
                io_message.clear();
                if (is_root(my_rank))
                {
                    try
                    {
                        cell = socket.read_doubles(9);
                        inv_cell = socket.read_doubles(9);
                        nat_socket = socket.read_int();
                        if (nat_socket < 0)
                        {
                            throw std::runtime_error("negative POSDATA atom count from i-PI socket");
                        }
                        positions = socket.read_doubles(static_cast<std::size_t>(3 * nat_socket));
                    }
                    catch (const std::exception& exc)
                    {
                        io_failed = 1;
                        io_message = exc.what();
                    }
                }
                throw_if_root_io_failed(io_failed, io_message, my_rank);
                bcast_double_vector(cell);
                bcast_double_vector(inv_cell);
                bcast_int(nat_socket);
                if (!is_root(my_rank))
                {
                    positions.assign(static_cast<std::size_t>(3 * nat_socket), 0.0);
                }
                bcast_double_vector(positions);

                if (nat_socket != ucell.nat)
                {
                    throw std::runtime_error("POSDATA atom count does not match STRU.");
                }
                const double max_cell_delta_bohr = max_abs_delta(cell, reference_cell);
                if (max_cell_delta_bohr > 1.0e-6)
                {
                    throw std::runtime_error("variable-cell socket updates are not supported yet.");
                }

                set_positions_from_ipi_bohr(ucell, positions);
                p_esolver->runner(ucell, istep);
                runner_completed = true;
                energy_hartree = p_esolver->cal_energy() * RY_TO_HARTREE;
                ModuleBase::matrix force;
                if (socket_input.cal_force)
                {
                    p_esolver->cal_force(ucell, force);
                    forces_hartree_bohr = flatten_forces_hartree_per_bohr(force);
                }
                ++istep;
                hasdata = true;
            }
            else if (header == "GETFORCE")
            {
                io_failed = 0;
                io_message.clear();
                if (is_root(my_rank))
                {
                    try
                    {
                        socket.write_header("FORCEREADY");
                        socket.write_double(energy_hartree);
                        socket.write_int(nat_return);
                        socket.write_doubles(forces_hartree_bohr);
                        socket.write_doubles(virial_hartree);
                        socket.write_int(0);
                    }
                    catch (const std::exception& exc)
                    {
                        io_failed = 1;
                        io_message = exc.what();
                    }
                }
                throw_if_root_io_failed(io_failed, io_message, my_rank);
                isinit = false;
                hasdata = false;
            }
            else
            {
                if (is_root(my_rank))
                {
                    running_log << " ABACUS socket driver exiting on header " << header << std::endl;
                }
                break;
            }
        }
    }
    catch (const std::exception& exc)
    {
        pending_error = exc.what();
        if (is_root(my_rank))
        {
            running_log << " ABACUS socket driver ended with error: " << pending_error << std::endl;
        }
    }

    if (is_root(my_rank))
    {
        socket.close();
    }
    if (esolver_ready && runner_completed && p_esolver)
    {
        p_esolver->after_all_runners(ucell);
    }
    p_esolver.reset();
    if (hardware_initialized)
    {
        this->finalize_hardware();
    }

#ifdef __RAPIDJSON
    Json::create_Json(&ucell, param);
#endif

    if (!pending_error.empty())
    {
        ModuleBase::WARNING_QUIT("ABACUS socket", pending_error);
    }
}
