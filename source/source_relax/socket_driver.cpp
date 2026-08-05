#include "socket_driver.h"

#include "source_relax/socket_frame.h"
#include "source_relax/socket_ipi.h"
#include "source_base/global_function.h"
#include "source_base/parallel_common.h"
#include "source_base/timer.h"
#include "source_cell/unitcell.h"
#include "source_cell/update_cell.h"
#include "source_esolver/esolver.h"
#include "source_io/module_parameter/input_parameter.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <limits>
#include <string>
#include <vector>

namespace
{
constexpr double RY_TO_HARTREE = 0.5;
constexpr int IPI_RANK_ROOT = 0;
constexpr double MAX_CELL_CONDITION = 1.0e12;
constexpr double INVERSE_ABSOLUTE_TOLERANCE
    = 64.0 * std::numeric_limits<double>::epsilon();
constexpr double INVERSE_RELATIVE_TOLERANCE = 64.0;
constexpr double STRESS_ABSOLUTE_TOLERANCE = 1.0e-10;
constexpr double STRESS_RELATIVE_TOLERANCE = 1.0e-8;

enum class DriverState
{
    NeedInit,
    Ready,
    HasData
};

struct ComputedFrame
{
    bool valid = false;
    double energy_hartree = 0.0;
    std::vector<double> forces_hartree_per_bohr;
    SocketFrame::Matrix9 virial_wire_hartree = {{0.0}};
};

struct PendingInputFrame
{
    SocketFrame::Matrix9 cell_wire = {{0.0}};
    SocketFrame::Matrix9 computed_inverse_wire_bohr_inv = {{0.0}};
    double volume_bohr3 = 0.0;
    std::vector<double> positions_bohr;
    bool cell_changed = false;
};

struct PendingAtomPosition
{
    ModuleBase::Vector3<double> taud;
    ModuleBase::Vector3<double> dis;
    ModuleBase::Vector3<int> boundary_shift;
};

using PendingBoundaryShifts = std::vector<std::vector<ModuleBase::Vector3<int>>>;

bool is_root()
{
#ifdef __MPI
    int rank = IPI_RANK_ROOT;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank == IPI_RANK_ROOT;
#else
    return true;
#endif
}

void bcast_double_vector(std::vector<double>& values)
{
#ifdef __MPI
    if (!values.empty())
    {
        Parallel_Common::bcast_double(values.data(), static_cast<int>(values.size()));
    }
#else
    (void)values;
#endif
}

void bcast_socket_double(double& value)
{
#ifdef __MPI
    Parallel_Common::bcast_double(value);
#else
    (void)value;
#endif
}

void bcast_matrix9(SocketFrame::Matrix9& values)
{
#ifdef __MPI
    Parallel_Common::bcast_double(values.data(), static_cast<int>(values.size()));
#else
    (void)values;
#endif
}

void bcast_socket_int(int& value)
{
#ifdef __MPI
    Parallel_Common::bcast_int(value);
#else
    (void)value;
#endif
}

void bcast_socket_chars(char* value, const int size)
{
#ifdef __MPI
    Parallel_Common::bcast_char(value, size);
#else
    (void)value;
    (void)size;
#endif
}

void bcast_socket_string(std::string& value)
{
    int size = static_cast<int>(value.size());
    bcast_socket_int(size);
    if (!is_root())
    {
        value.resize(static_cast<std::size_t>(size));
    }
    if (size > 0)
    {
        bcast_socket_chars(&value[0], size);
    }
}

void throw_if_root_failed(int root_failed, std::string root_message)
{
    bcast_socket_int(root_failed);
    bcast_socket_string(root_message);
    if (root_failed != 0)
    {
        throw std::runtime_error(root_message.empty() ? "i-PI socket operation failed" : root_message);
    }
}

void throw_if_any_rank_failed(int local_failed, std::string local_message)
{
    int any_failed = local_failed;
#ifdef __MPI
    MPI_Allreduce(MPI_IN_PLACE, &any_failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
    if (any_failed != 0)
    {
        if (local_message.empty())
        {
            local_message = "socket frame computation failed on another MPI rank";
        }
        throw std::runtime_error(local_message);
    }
}

[[noreturn]] void fail_during_collective_stage(const char* stage,
                                               const std::string& message)
{
#ifdef __MPI
    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::fprintf(stderr,
                 "ABACUS_SOCKET_MPI_FATAL stage=%s rank=%d message=%s\n",
                 stage,
                 rank,
                 message.c_str());
    std::fflush(stderr);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
#else
    (void)stage;
    throw std::runtime_error(message);
#endif
}

std::string bcast_header(std::string header)
{
    bcast_socket_string(header);
    return header;
}

std::string socket_address()
{
    const char* env = std::getenv("ABACUS_SOCKET_ADDRESS");
    if (env == nullptr || std::string(env).empty())
    {
        return "localhost:31415";
    }
    return std::string(env);
}

SocketFrame::Matrix9 ipi_cell_bohr_from_unitcell(const UnitCell& ucell)
{
    const double lat0 = ucell.lat0;
    // POSDATA H uses i-PI's column-vector cell convention. ASE therefore
    // sends its row-vector cell A as A^T. ABACUS stores lattice vectors as
    // rows in latvec, so use the transposed order here.
    return {{
        ucell.latvec.e11 * lat0, ucell.latvec.e21 * lat0, ucell.latvec.e31 * lat0,
        ucell.latvec.e12 * lat0, ucell.latvec.e22 * lat0, ucell.latvec.e32 * lat0,
        ucell.latvec.e13 * lat0, ucell.latvec.e23 * lat0, ucell.latvec.e33 * lat0,
    }};
}

double max_abs_component(const SocketFrame::Matrix9& values)
{
    double maximum = 0.0;
    for (std::size_t index = 0; index < values.size(); ++index)
    {
        maximum = std::max(maximum, std::fabs(values[index]));
    }
    return maximum;
}

double max_abs_delta(const SocketFrame::Matrix9& first, const SocketFrame::Matrix9& second)
{
    double maximum = 0.0;
    for (std::size_t index = 0; index < first.size(); ++index)
    {
        maximum = std::max(maximum, std::fabs(first[index] - second[index]));
    }
    return maximum;
}

double unchanged_cell_tolerance(const SocketFrame::Matrix9& cell_wire)
{
    return 32.0 * std::numeric_limits<double>::epsilon()
           * std::max(1.0, max_abs_component(cell_wire));
}

ModuleBase::Matrix3 matrix3_from_row_major(const SocketFrame::Matrix9& values)
{
    return ModuleBase::Matrix3(values[0], values[1], values[2],
                               values[3], values[4], values[5],
                               values[6], values[7], values[8]);
}

bool prepare_atom_positions(const UnitCell& ucell,
                            const std::vector<double>& positions_bohr,
                            const SocketFrame::Matrix9& inverse_abacus_bohr_inv,
                            std::vector<PendingAtomPosition>& pending,
                            std::string& message)
{
    pending.assign(static_cast<std::size_t>(ucell.nat), PendingAtomPosition());
    int iat = 0;
    for (int it = 0; it < ucell.ntype; ++it)
    {
        const Atom* atom = &ucell.atoms[it];
        for (int ia = 0; ia < atom->na; ++ia)
        {
            PendingAtomPosition& position = pending[static_cast<std::size_t>(iat)];
            for (int direct = 0; direct < 3; ++direct)
            {
                long double value = 0.0L;
                for (int cartesian = 0; cartesian < 3; ++cartesian)
                {
                    value += static_cast<long double>(positions_bohr[3 * iat + cartesian])
                             * inverse_abacus_bohr_inv[3 * cartesian + direct];
                }
                if (!std::isfinite(value)
                    || std::fabs(value) > static_cast<long double>(std::numeric_limits<double>::max()))
                {
                    message = "direct coordinates are not representable as finite doubles";
                    return false;
                }
                const double unwrapped = static_cast<double>(value);
                position.dis[direct] = unwrapped - atom->taud[ia][direct];
                position.taud[direct] = unwrapped;
                position.boundary_shift[direct] = 0;
                if (position.taud[direct] < 0.0)
                {
                    position.taud[direct] += 1.0;
                    position.boundary_shift[direct] = 1;
                }
                if (position.taud[direct] >= 1.0)
                {
                    position.taud[direct] -= 1.0;
                    position.boundary_shift[direct] = -1;
                }
                const double pbc_tolerance = 1.0e-12;
                if (position.taud[direct] < -pbc_tolerance
                    || position.taud[direct] >= 1.0 + pbc_tolerance)
                {
                    message = "Movement of atom is larger than the cell length";
                    return false;
                }
            }
            ++iat;
        }
    }
    message.clear();
    return true;
}

double max_wrapped_direct_delta(const UnitCell& ucell,
                                const std::vector<PendingAtomPosition>& pending)
{
    double maximum = 0.0;
    int iat = 0;
    for (int it = 0; it < ucell.ntype; ++it)
    {
        const Atom* atom = &ucell.atoms[it];
        for (int ia = 0; ia < atom->na; ++ia)
        {
            for (int direction = 0; direction < 3; ++direction)
            {
                double delta = pending[static_cast<std::size_t>(iat)].taud[direction]
                               - atom->taud[ia][direction];
                delta -= std::round(delta);
                maximum = std::max(maximum, std::fabs(delta));
            }
            ++iat;
        }
    }
    return maximum;
}

void prepare_commit_storage(const UnitCell& ucell,
                            const std::vector<PendingAtomPosition>& pending,
                            PendingBoundaryShifts& boundary_shifts)
{
    if (ucell.ntype < 0 || ucell.nat < 0 || ucell.atoms == nullptr)
    {
        throw std::runtime_error("UnitCell atom storage is not initialized");
    }
    if (pending.size() != static_cast<std::size_t>(ucell.nat))
    {
        throw std::runtime_error("pending socket positions do not match UnitCell nat");
    }
    boundary_shifts.assign(static_cast<std::size_t>(ucell.ntype),
                           std::vector<ModuleBase::Vector3<int>>());
    int iat = 0;
    for (int it = 0; it < ucell.ntype; ++it)
    {
        const Atom* atom = &ucell.atoms[it];
        if (atom->na < 0
            || atom->taud.size() < static_cast<std::size_t>(atom->na)
            || atom->dis.size() < static_cast<std::size_t>(atom->na)
            || atom->tau.size() < static_cast<std::size_t>(atom->na))
        {
            throw std::runtime_error("UnitCell atom position storage is inconsistent");
        }
        std::vector<ModuleBase::Vector3<int>>& shifts
            = boundary_shifts[static_cast<std::size_t>(it)];
        shifts.resize(static_cast<std::size_t>(atom->na));
        for (int ia = 0; ia < atom->na; ++ia)
        {
            if (iat >= ucell.nat)
            {
                throw std::runtime_error("UnitCell species atom counts exceed nat");
            }
            shifts[static_cast<std::size_t>(ia)]
                = pending[static_cast<std::size_t>(iat)].boundary_shift;
            ++iat;
        }
    }
    if (iat != ucell.nat)
    {
        throw std::runtime_error("UnitCell species atom counts do not sum to nat");
    }
}

void commit_frame_state(UnitCell& ucell,
                        const PendingInputFrame& frame,
                        const ModuleBase::Matrix3& new_latvec,
                        const std::vector<PendingAtomPosition>& pending,
                        PendingBoundaryShifts& boundary_shifts)
{
    if (frame.cell_changed)
    {
        ucell.latvec = new_latvec;
    }
    int iat = 0;
    for (int it = 0; it < ucell.ntype; ++it)
    {
        Atom* atom = &ucell.atoms[it];
        for (int ia = 0; ia < atom->na; ++ia)
        {
            const PendingAtomPosition& position = pending[static_cast<std::size_t>(iat)];
            atom->taud[ia] = position.taud;
            atom->dis[ia] = position.dis;
            ++iat;
        }
        atom->boundary_shift.swap(boundary_shifts[static_cast<std::size_t>(it)]);
    }
    ucell.ionic_position_updated = true;
    ucell.cell_parameter_updated = frame.cell_changed;
    if (!frame.cell_changed)
    {
        for (int it = 0; it < ucell.ntype; ++it)
        {
            Atom* atom = &ucell.atoms[it];
            for (int ia = 0; ia < atom->na; ++ia)
            {
                atom->tau[ia] = atom->taud[ia] * ucell.latvec;
            }
        }
    }
}

std::vector<double> flatten_forces_hartree_per_bohr(const ModuleBase::matrix& force,
                                                     const int nat)
{
    if (force.nr != nat || force.nc != 3)
    {
        throw std::runtime_error("force matrix must have nat rows and three columns");
    }
    std::vector<double> out(static_cast<std::size_t>(3 * nat));
    for (int iat = 0; iat < nat; ++iat)
    {
        for (int direction = 0; direction < 3; ++direction)
        {
            const double value = force(iat, direction);
            if (!std::isfinite(value))
            {
                throw std::runtime_error("force entries must be finite");
            }
            out[static_cast<std::size_t>(3 * iat + direction)] = value * RY_TO_HARTREE;
        }
    }
    return out;
}

SocketFrame::Matrix9 matrix9_from_stress(const ModuleBase::matrix& stress)
{
    if (stress.nr != 3 || stress.nc != 3)
    {
        throw std::runtime_error("stress matrix must have three rows and three columns");
    }
    SocketFrame::Matrix9 values;
    for (int row = 0; row < 3; ++row)
    {
        for (int column = 0; column < 3; ++column)
        {
            values[3 * row + column] = stress(row, column);
        }
    }
    return values;
}
} // namespace

void Socket_Driver::socket_driver(ModuleESolver::ESolver* p_esolver,
                                  UnitCell& ucell,
                                  const Input_para& inp,
                                  std::ofstream& ofs_running)
{
    ModuleBase::TITLE("Socket_Driver", "socket_driver");
    ModuleBase::timer::start("Socket_Driver", "socket_driver");

    if (p_esolver == nullptr)
    {
        ModuleBase::WARNING_QUIT("ABACUS socket", "socket driver requires a valid ESolver.");
    }
    if (!inp.cal_force)
    {
        ModuleBase::WARNING_QUIT("ABACUS socket", "socket_driver requires cal_force=1 for i-PI GETFORCE.");
    }
    if (inp.socket_variable_cell && !inp.cal_stress)
    {
        ModuleBase::WARNING_QUIT("ABACUS socket", "socket_variable_cell requires cal_stress=1.");
    }

    IpiSocket socket;

    try
    {
        int root_failed = 0;
        std::string root_message;
        if (is_root())
        {
            try
            {
                const std::string address = socket_address();
                ofs_running << " ABACUS socket driver connecting to i-PI endpoint " << address << std::endl;
                socket.connect(address);
            }
            catch (const std::exception& exc)
            {
                root_failed = 1;
                root_message = exc.what();
            }
        }
        throw_if_root_failed(root_failed, root_message);

        DriverState state = DriverState::NeedInit;
        int istep = 0;
        const int nat_return = ucell.nat;
        ComputedFrame published;
        const SocketFrame::Matrix9 reference_cell = ipi_cell_bohr_from_unitcell(ucell);
        bool checked_initial_positions = false;

        while (true)
        {
            std::string header;
            int peer_closed = 0;
            root_failed = 0;
            root_message.clear();
            if (is_root())
            {
                try
                {
                    header = socket.read_header();
                }
                catch (const IpiSocketClosed& exc)
                {
                    peer_closed = 1;
                    if (state == DriverState::HasData)
                    {
                        root_failed = 1;
                        root_message = std::string(exc.what())
                                       + "; peer closed while a computed frame was pending";
                    }
                }
                catch (const std::exception& exc)
                {
                    root_failed = 1;
                    root_message = exc.what();
                }
            }
            throw_if_root_failed(root_failed, root_message);
            bcast_socket_int(peer_closed);
            if (peer_closed != 0)
            {
                if (is_root())
                {
                    ofs_running << " ABACUS socket driver exiting after peer closed connection" << std::endl;
                }
                break;
            }
            header = bcast_header(header);

            if (header == "STATUS")
            {
                root_failed = 0;
                root_message.clear();
                if (is_root())
                {
                    try
                    {
                        if (state == DriverState::NeedInit)
                        {
                            socket.write_header("NEEDINIT");
                        }
                        else if (state == DriverState::Ready)
                        {
                            socket.write_header("READY");
                        }
                        else
                        {
                            socket.write_header("HAVEDATA");
                        }
                    }
                    catch (const std::exception& exc)
                    {
                        root_failed = 1;
                        root_message = exc.what();
                    }
                }
                throw_if_root_failed(root_failed, root_message);
            }
            else if (header == "INIT")
            {
                std::int32_t rid = 0;
                std::int32_t nbytes = 0;
                std::string params;
                root_failed = 0;
                root_message.clear();
                if (is_root())
                {
                    if (state != DriverState::NeedInit)
                    {
                        root_failed = 1;
                        root_message = "INIT requires NEEDINIT state";
                    }
                    else
                    {
                        try
                        {
                            rid = socket.read_int32();
                            nbytes = socket.read_int32();
                            if (nbytes < 0)
                            {
                                root_failed = 1;
                                root_message = "negative INIT payload length from i-PI socket";
                            }
                            else if (nbytes > 0)
                            {
                                params = socket.read_string(static_cast<std::size_t>(nbytes));
                            }
                        }
                        catch (const std::exception& exc)
                        {
                            root_failed = 1;
                            root_message = exc.what();
                        }
                    }
                }
                throw_if_root_failed(root_failed, root_message);
                if (nbytes > 0 && is_root())
                {
                    ofs_running << " ABACUS socket INIT params bytes " << nbytes << std::endl;
                }
                state = DriverState::Ready;
                if (is_root())
                {
                    ofs_running << " ABACUS socket INIT replica " << rid << std::endl;
                }
            }
            else if (header == "POSDATA")
            {
                PendingInputFrame frame;
                root_failed = 0;
                root_message.clear();
                if (is_root())
                {
                    if (state != DriverState::Ready)
                    {
                        root_failed = 1;
                        root_message = "POSDATA requires READY state";
                    }
                    else
                    {
                        try
                        {
                            const std::vector<double> cell_values = socket.read_doubles(9);
                            const std::vector<double> inverse_values = socket.read_doubles(9);
                            std::copy(cell_values.begin(), cell_values.end(), frame.cell_wire.begin());
                            SocketFrame::Matrix9 received_inverse;
                            std::copy(inverse_values.begin(), inverse_values.end(), received_inverse.begin());
                            const std::int32_t nat_socket = socket.read_int32();

                            const SocketFrame::CellValidation cell_validation
                                = SocketFrame::validate_ipi_cell(frame.cell_wire,
                                                                 received_inverse,
                                                                 MAX_CELL_CONDITION,
                                                                 INVERSE_ABSOLUTE_TOLERANCE,
                                                                 INVERSE_RELATIVE_TOLERANCE);
                            if (!cell_validation.ok)
                            {
                                root_failed = 1;
                                root_message = "invalid POSDATA cell: " + cell_validation.message;
                            }

                            std::size_t coordinate_count = 0;
                            if (root_failed == 0
                                && !SocketFrame::checked_position_count(nat_socket,
                                                                        ucell.nat,
                                                                        coordinate_count,
                                                                        root_message))
                            {
                                root_failed = 1;
                            }
                            if (root_failed == 0)
                            {
                                frame.positions_bohr = socket.read_doubles(coordinate_count);
                                if (!SocketFrame::validate_positions(frame.positions_bohr,
                                                                     coordinate_count,
                                                                     root_message))
                                {
                                    root_failed = 1;
                                }
                            }
                            if (root_failed == 0)
                            {
                                frame.computed_inverse_wire_bohr_inv
                                    = cell_validation.computed_inverse_wire_bohr_inv;
                                frame.volume_bohr3 = cell_validation.determinant_bohr3;
                                const SocketFrame::Matrix9 comparison_cell
                                    = inp.socket_variable_cell
                                          ? ipi_cell_bohr_from_unitcell(ucell)
                                          : reference_cell;
                                const bool changed
                                    = max_abs_delta(frame.cell_wire, comparison_cell)
                                      > unchanged_cell_tolerance(frame.cell_wire);
                                if (!inp.socket_variable_cell && changed)
                                {
                                    root_failed = 1;
                                    root_message
                                        = "fixed-cell socket mode rejects changed POSDATA cell";
                                }
                                frame.cell_changed = inp.socket_variable_cell && changed;
                            }
                        }
                        catch (const std::exception& exc)
                        {
                            root_failed = 1;
                            root_message = exc.what();
                        }
                    }
                }
                throw_if_root_failed(root_failed, root_message);
                bcast_matrix9(frame.cell_wire);
                bcast_matrix9(frame.computed_inverse_wire_bohr_inv);
                bcast_socket_double(frame.volume_bohr3);
                if (!is_root())
                {
                    frame.positions_bohr.assign(static_cast<std::size_t>(3 * ucell.nat), 0.0);
                }
                bcast_double_vector(frame.positions_bohr);
                int cell_changed = frame.cell_changed ? 1 : 0;
                bcast_socket_int(cell_changed);
                frame.cell_changed = cell_changed != 0;

                int local_failed = 0;
                std::string local_message;
                ModuleBase::Matrix3 new_latvec;
                std::vector<PendingAtomPosition> pending_positions;
                PendingBoundaryShifts pending_boundary_shifts;
                SocketFrame::Matrix9 scaled_abacus_cell;
                try
                {
                    if (!std::isfinite(ucell.lat0) || ucell.lat0 <= 0.0)
                    {
                        throw std::runtime_error("UnitCell lat0 must be finite and positive");
                    }
                    const SocketFrame::Matrix9 absolute_abacus_cell
                        = SocketFrame::transpose_matrix9(frame.cell_wire);
                    for (std::size_t index = 0; index < scaled_abacus_cell.size(); ++index)
                    {
                        scaled_abacus_cell[index] = absolute_abacus_cell[index] / ucell.lat0;
                        if (!std::isfinite(scaled_abacus_cell[index]))
                        {
                            throw std::runtime_error("scaled lattice entries must be finite");
                        }
                    }
                    new_latvec = matrix3_from_row_major(scaled_abacus_cell);
                    const SocketFrame::Matrix9 inverse_abacus_bohr_inv
                        = SocketFrame::transpose_matrix9(frame.computed_inverse_wire_bohr_inv);
                    std::string position_message;
                    if (!prepare_atom_positions(ucell,
                                                frame.positions_bohr,
                                                inverse_abacus_bohr_inv,
                                                pending_positions,
                                                position_message))
                    {
                        throw std::runtime_error(position_message);
                    }
                    prepare_commit_storage(ucell,
                                           pending_positions,
                                           pending_boundary_shifts);
                }
                catch (const std::exception& exc)
                {
                    local_failed = 1;
                    local_message = exc.what();
                }
                catch (...)
                {
                    local_failed = 1;
                    local_message = "unknown socket frame preflight failure";
                }
                throw_if_any_rank_failed(local_failed, local_message);
                if (!checked_initial_positions)
                {
                    checked_initial_positions = true;
                    if (max_wrapped_direct_delta(ucell, pending_positions) > 1.0e-5 && is_root())
                    {
                        ModuleBase::WARNING(
                            "ABACUS socket",
                            "first POSDATA positions are not PBC-equivalent to STRU atom order; "
                            "i-PI POSDATA carries no species, so the client atoms should use the same atom order as STRU.");
                    }
                }

                published = ComputedFrame();
                ComputedFrame computed;
                try
                {
                    commit_frame_state(ucell,
                                       frame,
                                       new_latvec,
                                       pending_positions,
                                       pending_boundary_shifts);
                }
                catch (const std::exception& exc)
                {
                    fail_during_collective_stage("cell-commit", exc.what());
                }
                catch (...)
                {
                    fail_during_collective_stage("cell-commit",
                                                 "unknown socket cell commit failure");
                }

                if (frame.cell_changed)
                {
                    try
                    {
                        unitcell::setup_cell_after_vc(ucell, ofs_running, inp.nspin);
                    }
                    catch (const std::exception& exc)
                    {
                        fail_during_collective_stage("setup_cell_after_vc", exc.what());
                    }
                    catch (...)
                    {
                        fail_during_collective_stage(
                            "setup_cell_after_vc",
                            "unknown socket setup_cell_after_vc failure");
                    }
                }

                try
                {
                    p_esolver->runner(ucell, istep);
                }
                catch (const std::exception& exc)
                {
                    fail_during_collective_stage("runner", exc.what());
                }
                catch (...)
                {
                    fail_during_collective_stage("runner",
                                                 "unknown socket runner failure");
                }

                local_failed = p_esolver->conv_esolver ? 0 : 1;
                local_message = local_failed != 0 ? "socket step SCF did not converge" : "";
                throw_if_any_rank_failed(local_failed, local_message);

                double energy_ry = 0.0;
                local_failed = 0;
                local_message.clear();
                try
                {
                    energy_ry = p_esolver->cal_energy();
                }
                catch (const std::exception& exc)
                {
                    fail_during_collective_stage("cal_energy", exc.what());
                }
                catch (...)
                {
                    fail_during_collective_stage("cal_energy",
                                                 "unknown socket energy failure");
                }
                local_failed = std::isfinite(energy_ry) ? 0 : 1;
                local_message = local_failed != 0 ? "energy must be finite" : "";
                throw_if_any_rank_failed(local_failed, local_message);
                computed.energy_hartree = energy_ry * RY_TO_HARTREE;
                if (is_root())
                {
                    ofs_running << " ABACUS socket return energy "
                                << energy_ry << " Ry, "
                                << energy_ry * ModuleBase::Ry_to_eV << " eV, "
                                << computed.energy_hartree << " Ha" << std::endl;
                }

                local_failed = 0;
                local_message.clear();
                ModuleBase::matrix force;
                try
                {
                    p_esolver->cal_force(ucell, force);
                }
                catch (const std::exception& exc)
                {
                    fail_during_collective_stage("cal_force", exc.what());
                }
                catch (...)
                {
                    fail_during_collective_stage("cal_force",
                                                 "unknown socket force failure");
                }
                try
                {
                    computed.forces_hartree_per_bohr
                        = flatten_forces_hartree_per_bohr(force, ucell.nat);
                }
                catch (const std::exception& exc)
                {
                    local_failed = 1;
                    local_message = exc.what();
                }
                catch (...)
                {
                    local_failed = 1;
                    local_message = "unknown socket force failure";
                }
                throw_if_any_rank_failed(local_failed, local_message);

                if (inp.cal_stress)
                {
                    local_failed = 0;
                    local_message.clear();
                    ModuleBase::matrix stress;
                    try
                    {
                        p_esolver->cal_stress(ucell, stress);
                    }
                    catch (const std::exception& exc)
                    {
                        fail_during_collective_stage("cal_stress", exc.what());
                    }
                    catch (...)
                    {
                        fail_during_collective_stage("cal_stress",
                                                     "unknown socket stress failure");
                    }
                    try
                    {
                        const SocketFrame::VirialConversion virial
                            = SocketFrame::make_ipi_virial(matrix9_from_stress(stress),
                                                           ucell.omega,
                                                           STRESS_ABSOLUTE_TOLERANCE,
                                                           STRESS_RELATIVE_TOLERANCE);
                        if (!virial.ok)
                        {
                            throw std::runtime_error(virial.message);
                        }
                        computed.virial_wire_hartree = virial.wire_virial_hartree;
                    }
                    catch (const std::exception& exc)
                    {
                        local_failed = 1;
                        local_message = exc.what();
                    }
                    catch (...)
                    {
                        local_failed = 1;
                        local_message = "unknown socket stress failure";
                    }
                    throw_if_any_rank_failed(local_failed, local_message);
                }
                computed.valid = true;
                published = computed;
                ++istep;
                state = DriverState::HasData;
            }
            else if (header == "GETFORCE")
            {
                root_failed = 0;
                root_message.clear();
                if (is_root())
                {
                    if (state != DriverState::HasData || !published.valid)
                    {
                        root_failed = 1;
                        root_message = "GETFORCE requires HAVEDATA state and a valid frame";
                    }
                    else
                    {
                        try
                        {
                            socket.write_header("FORCEREADY");
                            socket.write_double(published.energy_hartree);
                            socket.write_int32(static_cast<std::int32_t>(nat_return));
                            socket.write_doubles(published.forces_hartree_per_bohr);
                            const std::vector<double> virial(published.virial_wire_hartree.begin(),
                                                             published.virial_wire_hartree.end());
                            socket.write_doubles(virial);
                            socket.write_int32(0);
                        }
                        catch (const std::exception& exc)
                        {
                            root_failed = 1;
                            root_message = exc.what();
                        }
                    }
                }
                throw_if_root_failed(root_failed, root_message);
                published = ComputedFrame();
                state = DriverState::Ready;
            }
            else if (header == "EXIT")
            {
                if (is_root())
                {
                    ofs_running << " ABACUS socket driver received i-PI EXIT" << std::endl;
                }
                break;
            }
            else
            {
                if (is_root())
                {
                    root_failed = 1;
                    root_message = "unknown i-PI header: " + header;
                }
                throw_if_root_failed(root_failed, root_message);
            }
        }
    }
    catch (const std::exception& exc)
    {
        ModuleBase::WARNING_QUIT("ABACUS socket", exc.what());
    }

    if (is_root())
    {
        socket.close();
    }

    ModuleBase::timer::end("Socket_Driver", "socket_driver");
}
