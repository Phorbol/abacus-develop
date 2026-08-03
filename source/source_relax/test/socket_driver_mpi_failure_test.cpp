#include "source_relax/socket_driver.h"

#include "source_cell/unitcell.h"
#include "source_esolver/esolver.h"
#include "source_io/module_parameter/input_parameter.h"

#include <arpa/inet.h>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mpi.h>
#include <netinet/in.h>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <vector>

Magnetism::Magnetism()
{
}

Magnetism::~Magnetism()
{
}

namespace
{
constexpr std::size_t IPI_HEADER_LEN = 12;

std::string errno_message(const std::string& prefix)
{
    return prefix + ": " + std::strerror(errno);
}

void send_all(const int fd, const void* data, const std::size_t size)
{
    const char* cursor = static_cast<const char*>(data);
    std::size_t sent = 0;
    while (sent < size)
    {
        const ssize_t count = ::send(fd, cursor + sent, size - sent, 0);
        if (count < 0 && errno == EINTR)
        {
            continue;
        }
        if (count <= 0)
        {
            throw std::runtime_error(errno_message("MPI test peer send failed"));
        }
        sent += static_cast<std::size_t>(count);
    }
}

void send_header(const int fd, const std::string& value)
{
    std::string header = value;
    header.resize(IPI_HEADER_LEN, ' ');
    send_all(fd, header.data(), header.size());
}

template <typename T>
void send_value(const int fd, const T& value)
{
    send_all(fd, &value, sizeof(value));
}

class TcpServer
{
  public:
    TcpServer()
    {
        fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (fd_ < 0)
        {
            throw std::runtime_error(errno_message("MPI test socket failed"));
        }
        const int reuse = 1;
        if (::setsockopt(fd_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) != 0)
        {
            throw std::runtime_error(errno_message("MPI test setsockopt failed"));
        }
        sockaddr_in address;
        std::memset(&address, 0, sizeof(address));
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        address.sin_port = htons(0);
        if (::bind(fd_, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0
            || ::listen(fd_, 1) != 0)
        {
            throw std::runtime_error(errno_message("MPI test bind/listen failed"));
        }
        socklen_t size = sizeof(address);
        if (::getsockname(fd_, reinterpret_cast<sockaddr*>(&address), &size) != 0)
        {
            throw std::runtime_error(errno_message("MPI test getsockname failed"));
        }
        endpoint_ = "127.0.0.1:" + std::to_string(ntohs(address.sin_port));
    }

    ~TcpServer()
    {
        if (fd_ >= 0)
        {
            ::close(fd_);
        }
    }

    const std::string& endpoint() const
    {
        return endpoint_;
    }

    void send_one_frame() const
    {
        const int peer = ::accept(fd_, nullptr, nullptr);
        if (peer < 0)
        {
            throw std::runtime_error(errno_message("MPI test accept failed"));
        }
        try
        {
            send_header(peer, "INIT");
            const std::int32_t replica = 0;
            const std::int32_t parameter_bytes = 0;
            send_value(peer, replica);
            send_value(peer, parameter_bytes);

            send_header(peer, "POSDATA");
            const double cell[9] = {2.0, 0.0, 0.0,
                                    0.0, 2.0, 0.0,
                                    0.0, 0.0, 2.0};
            const double inverse[9] = {0.5, 0.0, 0.0,
                                       0.0, 0.5, 0.0,
                                       0.0, 0.0, 0.5};
            const std::int32_t nat = 2;
            const double positions[6] = {0.4, 0.6, 0.8, 1.4, 0.2, 0.4};
            send_all(peer, cell, sizeof(cell));
            send_all(peer, inverse, sizeof(inverse));
            send_value(peer, nat);
            send_all(peer, positions, sizeof(positions));
        }
        catch (...)
        {
            ::close(peer);
            throw;
        }
        ::close(peer);
    }

  private:
    int fd_ = -1;
    std::string endpoint_;
};

class RankSelectiveRunnerFailure : public ModuleESolver::ESolver
{
  public:
    void before_all_runners(BaseCell&, const Input_para&) override
    {
    }

    void runner(BaseCell&, const int) override
    {
        int rank = -1;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 1)
        {
            throw std::runtime_error("rank-selective runner failure");
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    void after_all_runners(BaseCell&) override
    {
    }

    double cal_energy() override
    {
        return 0.0;
    }

    void cal_force(BaseCell&, ModuleBase::matrix&) override
    {
    }

    void cal_stress(BaseCell&, ModuleBase::matrix&) override
    {
    }
};

void initialize_two_atom_cell(UnitCell& ucell)
{
    ucell.lat0 = 2.0;
    ucell.latvec.Identity();
    ucell.GT.Identity();
    ucell.G.Identity();
    ucell.omega = 8.0;
    ucell.ntype = 1;
    ucell.nat = 2;
    ucell.atoms = new Atom[1];
    ucell.set_atom_flag = true;
    ucell.atoms[0].na = 2;
    ucell.atoms[0].tau.resize(2);
    ucell.atoms[0].taud.resize(2);
    ucell.atoms[0].dis.resize(2);
    ucell.atoms[0].taud[0] = ModuleBase::Vector3<double>(0.1, 0.2, 0.3);
    ucell.atoms[0].taud[1] = ModuleBase::Vector3<double>(0.4, 0.5, 0.6);
    ucell.atoms[0].tau[0] = ucell.atoms[0].taud[0];
    ucell.atoms[0].tau[1] = ucell.atoms[0].taud[1];
}
} // namespace

int main(int argc, char** argv)
{
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = -1;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 2 || provided < MPI_THREAD_FUNNELED)
    {
        if (rank == 0)
        {
            std::fprintf(stderr, "MPI test setup requires two ranks and MPI_THREAD_FUNNELED\n");
        }
        MPI_Finalize();
        return 2;
    }

    TcpServer* server = nullptr;
    std::string endpoint;
    try
    {
        if (rank == 0)
        {
            server = new TcpServer();
            endpoint = server->endpoint();
        }
        int endpoint_size = static_cast<int>(endpoint.size());
        MPI_Bcast(&endpoint_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
        endpoint.resize(static_cast<std::size_t>(endpoint_size));
        MPI_Bcast(&endpoint[0], endpoint_size, MPI_CHAR, 0, MPI_COMM_WORLD);
        if (::setenv("ABACUS_SOCKET_ADDRESS", endpoint.c_str(), 1) != 0)
        {
            throw std::runtime_error(errno_message("MPI test setenv failed"));
        }
    }
    catch (const std::exception& error)
    {
        std::fprintf(stderr, "MPI test setup failed on rank %d: %s\n", rank, error.what());
        std::fflush(stderr);
        MPI_Abort(MPI_COMM_WORLD, 3);
    }

    std::thread peer_thread;
    if (rank == 0)
    {
        peer_thread = std::thread([server]() {
            try
            {
                server->send_one_frame();
            }
            catch (const std::exception& error)
            {
                std::fprintf(stderr, "MPI test peer failed: %s\n", error.what());
                std::fflush(stderr);
            }
        });
    }
    MPI_Barrier(MPI_COMM_WORLD);

    UnitCell ucell;
    initialize_two_atom_cell(ucell);
    Input_para input;
    input.cal_force = true;
    input.cal_stress = false;
    input.socket_variable_cell = false;
    input.nspin = 1;
    RankSelectiveRunnerFailure solver;
    std::ofstream running("/dev/null");
    Socket_Driver driver;
    driver.socket_driver(&solver, ucell, input, running);

    if (rank == 0 && peer_thread.joinable())
    {
        peer_thread.join();
        delete server;
    }
    std::fprintf(stderr, "MPI test unexpectedly returned on rank %d\n", rank);
    MPI_Finalize();
    return 4;
}
