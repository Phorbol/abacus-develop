#include "source_relax/socket_driver.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "source_cell/unitcell.h"
#include "source_esolver/esolver.h"
#include "source_io/module_parameter/input_parameter.h"
#include "for_test.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <poll.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace
{
constexpr std::size_t IPI_HEADER_LEN = 12;
constexpr int DRIVER_DEADLINE_MS = 5000;
constexpr int CHILD_TERM_GRACE_MS = 250;
constexpr int CHILD_KILL_GRACE_MS = 1000;

std::string errno_message(const std::string& prefix)
{
    return prefix + ": " + std::strerror(errno);
}

class MonotonicDeadline
{
  public:
    explicit MonotonicDeadline(const int timeout_ms)
        : expires_(std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms))
    {
    }

    int remaining_ms() const
    {
        const std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
        if (now >= expires_)
        {
            return 0;
        }
        const long long remaining
            = std::chrono::duration_cast<std::chrono::milliseconds>(expires_ - now).count();
        if (remaining >= INT_MAX)
        {
            return INT_MAX;
        }
        return static_cast<int>(remaining > 0 ? remaining : 1);
    }

  private:
    std::chrono::steady_clock::time_point expires_;
};

class UniqueFd
{
  public:
    explicit UniqueFd(const int fd = -1) : fd_(fd)
    {
    }

    ~UniqueFd()
    {
        reset();
    }

    UniqueFd(const UniqueFd&) = delete;
    UniqueFd& operator=(const UniqueFd&) = delete;

    int get() const
    {
        return fd_;
    }

    void reset(const int fd = -1)
    {
        if (fd_ >= 0)
        {
            ::close(fd_);
        }
        fd_ = fd;
    }

  private:
    int fd_;
};

void set_nonblocking(const int fd)
{
    const int flags = ::fcntl(fd, F_GETFL, 0);
    if (flags < 0 || ::fcntl(fd, F_SETFL, flags | O_NONBLOCK) != 0)
    {
        throw std::runtime_error(errno_message("fcntl failed"));
    }
}

void wait_for_fd(const int fd,
                 const short events,
                 const MonotonicDeadline& deadline,
                 const std::string& operation)
{
    while (true)
    {
        const int timeout_ms = deadline.remaining_ms();
        if (timeout_ms == 0)
        {
            throw std::runtime_error(operation + " timed out");
        }

        pollfd descriptor;
        descriptor.fd = fd;
        descriptor.events = events;
        descriptor.revents = 0;
        const int result = ::poll(&descriptor, 1, timeout_ms);
        if (result == 0)
        {
            throw std::runtime_error(operation + " timed out");
        }
        if (result < 0)
        {
            if (errno == EINTR)
            {
                continue;
            }
            throw std::runtime_error(errno_message(operation + " poll failed"));
        }
        if ((descriptor.revents & POLLNVAL) != 0)
        {
            throw std::runtime_error(operation + " encountered an invalid descriptor");
        }
        if ((descriptor.revents & (events | POLLERR | POLLHUP)) != 0)
        {
            return;
        }
    }
}

class ChildProcess
{
  public:
    explicit ChildProcess(const pid_t pid) : pid_(pid)
    {
    }

    ~ChildProcess()
    {
        terminate_and_reap();
    }

    ChildProcess(const ChildProcess&) = delete;
    ChildProcess& operator=(const ChildProcess&) = delete;

    int wait_until(const MonotonicDeadline& deadline)
    {
        while (true)
        {
            int status = 0;
            const pid_t result = ::waitpid(pid_, &status, WNOHANG);
            if (result == pid_)
            {
                pid_ = -1;
                return status;
            }
            if (result < 0)
            {
                if (errno == EINTR)
                {
                    continue;
                }
                if (errno == ECHILD)
                {
                    pid_ = -1;
                }
                throw std::runtime_error(errno_message("waitpid failed"));
            }

            const int remaining_ms = deadline.remaining_ms();
            if (remaining_ms == 0)
            {
                throw std::runtime_error("child wait timed out");
            }
            ::poll(nullptr, 0, remaining_ms < 10 ? remaining_ms : 10);
        }
    }

  private:
    bool reap_for(const int timeout_ms) noexcept
    {
        const std::chrono::steady_clock::time_point expires
            = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        while (pid_ > 0)
        {
            int status = 0;
            const pid_t result = ::waitpid(pid_, &status, WNOHANG);
            if (result == pid_ || (result < 0 && errno == ECHILD))
            {
                pid_ = -1;
                return true;
            }
            if (result < 0 && errno != EINTR)
            {
                return false;
            }
            if (std::chrono::steady_clock::now() >= expires)
            {
                return false;
            }
            ::poll(nullptr, 0, 10);
        }
        return true;
    }

    void terminate_and_reap() noexcept
    {
        if (pid_ <= 0 || reap_for(0))
        {
            return;
        }
        static_cast<void>(::kill(pid_, SIGTERM));
        if (reap_for(CHILD_TERM_GRACE_MS))
        {
            return;
        }
        static_cast<void>(::kill(pid_, SIGKILL));
        static_cast<void>(reap_for(CHILD_KILL_GRACE_MS));
    }

    pid_t pid_;
};

class PeerClosed : public std::runtime_error
{
  public:
    explicit PeerClosed(const std::string& message) : std::runtime_error(message)
    {
    }
};

void send_all(const int fd,
              const void* data,
              const std::size_t nbytes,
              const MonotonicDeadline& deadline)
{
    const char* cursor = static_cast<const char*>(data);
    std::size_t done = 0;
    while (done < nbytes)
    {
        wait_for_fd(fd, POLLOUT, deadline, "socket send");
#ifdef MSG_NOSIGNAL
        int flags = MSG_NOSIGNAL;
#else
        int flags = 0;
#endif
#ifdef MSG_DONTWAIT
        flags |= MSG_DONTWAIT;
#endif
        const ssize_t sent = ::send(fd, cursor + done, nbytes - done, flags);
        if (sent < 0)
        {
            if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)
            {
                continue;
            }
            if (errno == EPIPE || errno == ECONNRESET)
            {
                throw PeerClosed(errno_message("socket peer closed during send"));
            }
            throw std::runtime_error(errno_message("send failed"));
        }
        if (sent == 0)
        {
            throw std::runtime_error("send returned zero");
        }
        done += static_cast<std::size_t>(sent);
    }
}

template <typename T>
void send_value(const int fd, const T& value, const MonotonicDeadline& deadline)
{
    send_all(fd, &value, sizeof(value), deadline);
}

void send_header(const int fd, const std::string& header, const MonotonicDeadline& deadline)
{
    std::string padded = header;
    padded.resize(IPI_HEADER_LEN, ' ');
    send_all(fd, padded.data(), padded.size(), deadline);
}

bool try_send_status(const int fd, const MonotonicDeadline& deadline)
{
    try
    {
        send_header(fd, "STATUS", deadline);
        return true;
    }
    catch (const PeerClosed&)
    {
        return false;
    }
}

std::string read_header_or_close(const int fd, const MonotonicDeadline& deadline)
{
    char header[IPI_HEADER_LEN];
    std::size_t done = 0;
    while (done < sizeof(header))
    {
        wait_for_fd(fd, POLLIN, deadline, "socket receive");
#ifdef MSG_DONTWAIT
        const int flags = MSG_DONTWAIT;
#else
        const int flags = 0;
#endif
        const ssize_t received = ::recv(fd, header + done, sizeof(header) - done, flags);
        if (received == 0 || (received < 0 && errno == ECONNRESET))
        {
            if (done == 0)
            {
                return "";
            }
            throw std::runtime_error("socket closed during response header");
        }
        if (received < 0)
        {
            if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)
            {
                continue;
            }
            throw std::runtime_error(errno_message("receive failed"));
        }
        done += static_cast<std::size_t>(received);
    }

    std::string value(header, sizeof(header));
    while (!value.empty() && value.back() == ' ')
    {
        value.pop_back();
    }
    return value;
}

void read_all(const int fd,
              void* data,
              const std::size_t nbytes,
              const MonotonicDeadline& deadline)
{
    char* cursor = static_cast<char*>(data);
    std::size_t done = 0;
    while (done < nbytes)
    {
        wait_for_fd(fd, POLLIN, deadline, "socket receive");
#ifdef MSG_DONTWAIT
        const int flags = MSG_DONTWAIT;
#else
        const int flags = 0;
#endif
        const ssize_t received = ::recv(fd, cursor + done, nbytes - done, flags);
        if (received == 0 || (received < 0 && errno == ECONNRESET))
        {
            throw PeerClosed("socket peer closed during receive");
        }
        if (received < 0)
        {
            if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)
            {
                continue;
            }
            throw std::runtime_error(errno_message("receive failed"));
        }
        done += static_cast<std::size_t>(received);
    }
}

template <typename T>
T read_value(const int fd, const MonotonicDeadline& deadline)
{
    T value;
    read_all(fd, &value, sizeof(value), deadline);
    return value;
}

std::vector<double> read_doubles(const int fd,
                                 const std::size_t count,
                                 const MonotonicDeadline& deadline)
{
    std::vector<double> values(count);
    if (!values.empty())
    {
        read_all(fd, values.data(), values.size() * sizeof(double), deadline);
    }
    return values;
}

class UnixSocketServer
{
  public:
    UnixSocketServer()
    {
        char dir_template[] = "/tmp/abacus_socket_driver_test_XXXXXX";
        char* made_dir = ::mkdtemp(dir_template);
        if (made_dir == nullptr)
        {
            throw std::runtime_error(errno_message("mkdtemp failed"));
        }
        dir_ = made_dir;
        path_ = dir_ + "/ipi.sock";

        listen_fd_ = ::socket(AF_UNIX, SOCK_STREAM, 0);
        if (listen_fd_ < 0)
        {
            throw std::runtime_error(errno_message("socket failed"));
        }

        sockaddr_un address;
        std::memset(&address, 0, sizeof(address));
        address.sun_family = AF_UNIX;
        std::strncpy(address.sun_path, path_.c_str(), sizeof(address.sun_path) - 1);
        if (::bind(listen_fd_, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0)
        {
            throw std::runtime_error(errno_message("bind failed"));
        }
        if (::listen(listen_fd_, 1) != 0)
        {
            throw std::runtime_error(errno_message("listen failed"));
        }
        set_nonblocking(listen_fd_);
    }

    ~UnixSocketServer()
    {
        if (listen_fd_ >= 0)
        {
            ::close(listen_fd_);
        }
        if (!path_.empty())
        {
            ::unlink(path_.c_str());
        }
        if (!dir_.empty())
        {
            ::rmdir(dir_.c_str());
        }
    }

    UnixSocketServer(const UnixSocketServer&) = delete;
    UnixSocketServer& operator=(const UnixSocketServer&) = delete;

    std::string address() const
    {
        return path_ + ":UNIX";
    }

    int accept_until(const MonotonicDeadline& deadline) const
    {
        while (true)
        {
            wait_for_fd(listen_fd_, POLLIN, deadline, "socket accept");
            const int fd = ::accept(listen_fd_, nullptr, nullptr);
            if (fd >= 0)
            {
                try
                {
                    set_nonblocking(fd);
                }
                catch (...)
                {
                    ::close(fd);
                    throw;
                }
                return fd;
            }
            if (errno != EINTR && errno != EAGAIN && errno != EWOULDBLOCK)
            {
                throw std::runtime_error(errno_message("accept failed"));
            }
        }
    }

  private:
    int listen_fd_ = -1;
    std::string dir_;
    std::string path_;
};

using Matrix9 = std::array<double, 9>;

struct WireFrame
{
    Matrix9 cell;
    Matrix9 inverse;
    std::int32_t nat = 2;
    std::vector<double> positions;
};

WireFrame fixed_frame()
{
    WireFrame frame;
    frame.cell = {{2.0, 0.0, 0.0,
                   0.0, 2.0, 0.0,
                   0.0, 0.0, 2.0}};
    frame.inverse = {{0.5, 0.0, 0.0,
                      0.0, 0.5, 0.0,
                      0.0, 0.0, 0.5}};
    frame.positions = {0.4, 0.6, 0.8, 1.4, 0.2, 0.4};
    return frame;
}

WireFrame triclinic_frame()
{
    WireFrame frame;
    // Wire cell is the transpose of the absolute ABACUS row-vector cell.
    frame.cell = {{2.0, 0.3, 0.5,
                   0.1, 3.0, 0.6,
                   0.2, 0.4, 4.0}};
    frame.inverse = {{0.5078597339782346, -0.04318535152876144, -0.057004664017965105,
                      -0.012091898428053204, 0.3411642770772154, -0.04966315425807566,
                      -0.02418379685610641, -0.03195716013128347, 0.25781654862670583}};
    // Literal Cartesian products of direct rows (0.2,0.3,0.4) and
    // (0.7,0.1,0.2) with the absolute ABACUS cell above.
    frame.positions = {0.69, 1.16, 1.76, 1.53, 0.49, 0.98};
    return frame;
}

void send_init(const int fd, const MonotonicDeadline& deadline)
{
    const std::int32_t replica = 17;
    const std::string params = "task6";
    const std::int32_t parameter_bytes = static_cast<std::int32_t>(params.size());
    send_header(fd, "INIT", deadline);
    send_value(fd, replica, deadline);
    send_value(fd, parameter_bytes, deadline);
    send_all(fd, params.data(), params.size(), deadline);
}

void send_posdata(const int fd, const WireFrame& frame, const MonotonicDeadline& deadline)
{
    std::string header = "POSDATA";
    header.resize(IPI_HEADER_LEN, ' ');
    const std::size_t cell_bytes = frame.cell.size() * sizeof(double);
    const std::size_t position_bytes = frame.positions.size() * sizeof(double);
    std::vector<char> packet(IPI_HEADER_LEN + 2 * cell_bytes + sizeof(frame.nat) + position_bytes);
    std::size_t offset = 0;
    std::memcpy(packet.data() + offset, header.data(), header.size());
    offset += header.size();
    std::memcpy(packet.data() + offset, frame.cell.data(), cell_bytes);
    offset += cell_bytes;
    std::memcpy(packet.data() + offset, frame.inverse.data(), cell_bytes);
    offset += cell_bytes;
    std::memcpy(packet.data() + offset, &frame.nat, sizeof(frame.nat));
    offset += sizeof(frame.nat);
    if (position_bytes > 0)
    {
        std::memcpy(packet.data() + offset, frame.positions.data(), position_bytes);
    }
    send_all(fd, packet.data(), packet.size(), deadline);
}

std::string request_status(const int fd, const MonotonicDeadline& deadline)
{
    send_header(fd, "STATUS", deadline);
    return read_header_or_close(fd, deadline);
}

struct ForceResponse
{
    std::string header;
    double energy_hartree = 0.0;
    std::int32_t nat = 0;
    std::vector<double> forces_hartree_per_bohr;
    std::vector<double> virial_wire_hartree;
    std::int32_t extra_bytes = -1;
};

ForceResponse request_force(const int fd, const MonotonicDeadline& deadline)
{
    send_header(fd, "GETFORCE", deadline);
    ForceResponse response;
    response.header = read_header_or_close(fd, deadline);
    if (response.header.empty())
    {
        return response;
    }
    response.energy_hartree = read_value<double>(fd, deadline);
    response.nat = read_value<std::int32_t>(fd, deadline);
    response.forces_hartree_per_bohr
        = read_doubles(fd, static_cast<std::size_t>(3 * response.nat), deadline);
    response.virial_wire_hartree = read_doubles(fd, 9, deadline);
    response.extra_bytes = read_value<std::int32_t>(fd, deadline);
    return response;
}

struct DriverObservation
{
    int runner_calls;
    int force_calls;
    int stress_calls;
    int last_step;
    int ionic_position_updated;
    int cell_parameter_updated;
    int event_count;
    int events[4];
    double lat0;
    double omega;
    double latvec[9];
    double taud[6];
    double tau[6];
};

class SharedObservation
{
  public:
    SharedObservation()
    {
        memory_ = ::mmap(nullptr,
                         sizeof(DriverObservation),
                         PROT_READ | PROT_WRITE,
                         MAP_SHARED | MAP_ANONYMOUS,
                         -1,
                         0);
        if (memory_ == MAP_FAILED)
        {
            throw std::runtime_error(errno_message("mmap failed"));
        }
        std::memset(memory_, 0, sizeof(DriverObservation));
    }

    ~SharedObservation()
    {
        if (memory_ != MAP_FAILED)
        {
            ::munmap(memory_, sizeof(DriverObservation));
        }
    }

    SharedObservation(const SharedObservation&) = delete;
    SharedObservation& operator=(const SharedObservation&) = delete;

    DriverObservation* get() const
    {
        return static_cast<DriverObservation*>(memory_);
    }

  private:
    void* memory_ = MAP_FAILED;
};

struct SolverConfig
{
    bool converged = true;
    bool throw_in_runner = false;
    bool nonfinite_energy = false;
    bool nonfinite_force = false;
    bool nonfinite_stress = false;
    double energy_ry = 8.0;
    std::array<double, 6> force_ry_per_bohr = {{2.0, -4.0, 6.0, 8.0, -10.0, 12.0}};
    Matrix9 stress_ry_per_bohr3 = {{1.0, 2.0, 3.0,
                                    2.0, 4.0, 5.0,
                                    3.0, 5.0, 6.0}};
};

void record_unitcell(const UnitCell& ucell, DriverObservation& observation)
{
    observation.lat0 = ucell.lat0;
    observation.omega = ucell.omega;
    observation.ionic_position_updated = ucell.ionic_position_updated ? 1 : 0;
    observation.cell_parameter_updated = ucell.cell_parameter_updated ? 1 : 0;
    const double latvec[9] = {ucell.latvec.e11, ucell.latvec.e12, ucell.latvec.e13,
                              ucell.latvec.e21, ucell.latvec.e22, ucell.latvec.e23,
                              ucell.latvec.e31, ucell.latvec.e32, ucell.latvec.e33};
    std::copy(latvec, latvec + 9, observation.latvec);
    for (int ia = 0; ia < 2; ++ia)
    {
        observation.taud[3 * ia + 0] = ucell.atoms[0].taud[ia].x;
        observation.taud[3 * ia + 1] = ucell.atoms[0].taud[ia].y;
        observation.taud[3 * ia + 2] = ucell.atoms[0].taud[ia].z;
        observation.tau[3 * ia + 0] = ucell.atoms[0].tau[ia].x;
        observation.tau[3 * ia + 1] = ucell.atoms[0].tau[ia].y;
        observation.tau[3 * ia + 2] = ucell.atoms[0].tau[ia].z;
    }
}

class FakeESolver : public ModuleESolver::ESolver
{
  public:
    FakeESolver(const SolverConfig& config, DriverObservation& observation)
        : config_(config), observation_(observation)
    {
    }

    void before_all_runners(BaseCell&, const Input_para&) override
    {
    }

    void runner(BaseCell& cell, const int istep) override
    {
        UnitCell& ucell = dynamic_cast<UnitCell&>(cell);
        ++observation_.runner_calls;
        observation_.last_step = istep;
        observation_.event_count = 0;
        record_unitcell(ucell, observation_);
        this->conv_esolver = config_.converged;
        if (config_.throw_in_runner)
        {
            throw std::runtime_error("fake runner failed after commit");
        }
    }

    void after_all_runners(BaseCell&) override
    {
    }

    double cal_energy() override
    {
        if (config_.nonfinite_energy)
        {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return config_.energy_ry + 10.0 * (observation_.runner_calls - 1);
    }

    void cal_force(BaseCell& cell, ModuleBase::matrix& force) override
    {
        ++observation_.force_calls;
        observation_.events[observation_.event_count++] = 1;
        force.create(cell.nat(), 3);
        const double frame_offset = 20.0 * (observation_.runner_calls - 1);
        for (int index = 0; index < 6; ++index)
        {
            force(index / 3, index % 3) = config_.force_ry_per_bohr[index] + frame_offset;
        }
        if (config_.nonfinite_force)
        {
            force(1, 2) = std::numeric_limits<double>::infinity();
        }
    }

    void cal_stress(BaseCell&, ModuleBase::matrix& stress) override
    {
        ++observation_.stress_calls;
        observation_.events[observation_.event_count++] = 2;
        stress.create(3, 3);
        for (int index = 0; index < 9; ++index)
        {
            stress(index / 3, index % 3) = config_.stress_ry_per_bohr3[index];
        }
        if (config_.nonfinite_stress)
        {
            stress(2, 1) = std::numeric_limits<double>::quiet_NaN();
        }
    }

  private:
    SolverConfig config_;
    DriverObservation& observation_;
};

struct DriverConfig
{
    bool variable_cell = false;
    bool cal_stress = false;
    SolverConfig solver;
};

struct DriverResult
{
    int exit_code = -1;
    std::string diagnostic;
    DriverObservation observation;
};

enum class ChildMode
{
    run_driver,
    stall_before_connect
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
    ucell.atoms[0].na = 2;
    ucell.atoms[0].tau.resize(2);
    ucell.atoms[0].taud.resize(2);
    ucell.atoms[0].dis.resize(2);
    ucell.atoms[0].taud[0] = ModuleBase::Vector3<double>(0.1, 0.2, 0.3);
    ucell.atoms[0].taud[1] = ModuleBase::Vector3<double>(0.4, 0.5, 0.6);
    ucell.atoms[0].tau[0] = ucell.atoms[0].taud[0];
    ucell.atoms[0].tau[1] = ucell.atoms[0].taud[1];
}

std::string read_pipe(const int fd, const MonotonicDeadline& deadline)
{
    std::string output;
    char buffer[512];
    while (true)
    {
        wait_for_fd(fd, POLLIN, deadline, "diagnostic pipe read");
        const ssize_t nread = ::read(fd, buffer, sizeof(buffer));
        if (nread == 0)
        {
            break;
        }
        if (nread < 0)
        {
            if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)
            {
                continue;
            }
            throw std::runtime_error(errno_message("pipe read failed"));
        }
        output.append(buffer, static_cast<std::size_t>(nread));
    }
    return output;
}

using PeerAction = std::function<void(int, const MonotonicDeadline&)>;

DriverResult run_driver(const DriverConfig& config,
                        const PeerAction& peer_action,
                        const ChildMode child_mode,
                        const int timeout_ms,
                        pid_t* observed_child)
{
    const MonotonicDeadline deadline(timeout_ms);
    UnixSocketServer server;
    SharedObservation shared_observation;
    int output_pipe[2];
    if (::pipe(output_pipe) != 0)
    {
        throw std::runtime_error(errno_message("pipe failed"));
    }
    UniqueFd output_read(output_pipe[0]);
    UniqueFd output_write(output_pipe[1]);

    const pid_t child = ::fork();
    if (child < 0)
    {
        throw std::runtime_error(errno_message("fork failed"));
    }
    if (child == 0)
    {
        output_read.reset();
        if (::dup2(output_write.get(), STDOUT_FILENO) < 0
            || ::dup2(output_write.get(), STDERR_FILENO) < 0
            || ::setenv("ABACUS_SOCKET_ADDRESS", server.address().c_str(), 1) != 0)
        {
            ::_exit(2);
        }
        output_write.reset();

        if (child_mode == ChildMode::stall_before_connect)
        {
            while (true)
            {
                ::pause();
            }
        }

        UnitCell ucell;
        initialize_two_atom_cell(ucell);
        Input_para input;
        input.cal_force = true;
        input.cal_stress = config.cal_stress;
        input.socket_variable_cell = config.variable_cell;
        input.nspin = 1;
        FakeESolver solver(config.solver, *shared_observation.get());
        std::ofstream running("/dev/null");
        Socket_Driver driver;
        driver.socket_driver(&solver, ucell, input, running);
        std::cout.flush();
        std::cerr.flush();
        ::_exit(0);
    }

    if (observed_child != nullptr)
    {
        *observed_child = child;
    }
    ChildProcess child_process(child);
    output_write.reset();
    set_nonblocking(output_read.get());
    DriverResult result;
    UniqueFd peer(server.accept_until(deadline));
    peer_action(peer.get(), deadline);
    peer.reset();

    result.diagnostic = read_pipe(output_read.get(), deadline);
    output_read.reset();
    const int status = child_process.wait_until(deadline);
    if (WIFEXITED(status))
    {
        result.exit_code = WEXITSTATUS(status);
    }
    else if (WIFSIGNALED(status))
    {
        result.exit_code = 128 + WTERMSIG(status);
    }
    result.observation = *shared_observation.get();
    return result;
}

struct ChildCleanupProbe
{
    bool child_ready = false;
    bool wait_timed_out = false;
    bool child_reaped = false;
    long long elapsed_ms = 0;
};

ChildCleanupProbe child_scope_reaps_unresponsive_child()
{
    const std::chrono::steady_clock::time_point started = std::chrono::steady_clock::now();
    int ready_pipe[2];
    if (::pipe(ready_pipe) != 0)
    {
        throw std::runtime_error(errno_message("cleanup probe pipe failed"));
    }
    UniqueFd ready_read(ready_pipe[0]);
    UniqueFd ready_write(ready_pipe[1]);

    const pid_t child = ::fork();
    if (child < 0)
    {
        throw std::runtime_error(errno_message("cleanup probe fork failed"));
    }
    if (child == 0)
    {
        ready_read.reset();
        static_cast<void>(::signal(SIGTERM, SIG_IGN));
        const char ready = 'R';
        if (::write(ready_write.get(), &ready, 1) != 1)
        {
            ::_exit(3);
        }
        ready_write.reset();
        while (true)
        {
            ::pause();
        }
    }

    ChildCleanupProbe probe;
    {
        ChildProcess child_process(child);
        ready_write.reset();
        set_nonblocking(ready_read.get());
        probe.child_ready = read_pipe(ready_read.get(), MonotonicDeadline(500)) == "R";
        ready_read.reset();
        const MonotonicDeadline deadline(50);
        try
        {
            static_cast<void>(child_process.wait_until(deadline));
        }
        catch (const std::runtime_error& error)
        {
            probe.wait_timed_out = std::string(error.what()).find("timed out") != std::string::npos;
        }
    }

    errno = 0;
    int status = 0;
    probe.child_reaped = (::waitpid(child, &status, WNOHANG) < 0 && errno == ECHILD);
    probe.elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::steady_clock::now() - started)
                           .count();
    return probe;
}
} // namespace

TEST(SocketDriverTest, ProtocolStateSequenceEndsReadyAfterForceDelivery)
{
    std::vector<std::string> statuses;
    ForceResponse response;
    DriverConfig config;
    const DriverResult result = run_driver(
        config,
        [&](const int fd, const MonotonicDeadline& deadline) {
            statuses.push_back(request_status(fd, deadline));
            send_init(fd, deadline);
            statuses.push_back(request_status(fd, deadline));
            send_posdata(fd, fixed_frame(), deadline);
            statuses.push_back(request_status(fd, deadline));
            response = request_force(fd, deadline);
            statuses.push_back(request_status(fd, deadline));
        },
        ChildMode::run_driver,
        DRIVER_DEADLINE_MS,
        nullptr);

    EXPECT_THAT(statuses, testing::ElementsAre("NEEDINIT", "READY", "HAVEDATA", "READY"));
    EXPECT_EQ("FORCEREADY", response.header);
    EXPECT_EQ(0, result.exit_code);
}

TEST(SocketDriverTest, VariableCellCommitReturnsMappedEnergyForceAndVirial)
{
    DriverConfig config;
    config.variable_cell = true;
    config.cal_stress = true;
    std::string data_status;
    ForceResponse response;
    const DriverResult result = run_driver(
        config,
        [&](const int fd, const MonotonicDeadline& deadline) {
            send_init(fd, deadline);
            send_posdata(fd, triclinic_frame(), deadline);
            data_status = request_status(fd, deadline);
            if (data_status == "HAVEDATA")
            {
                response = request_force(fd, deadline);
            }
        },
        ChildMode::run_driver,
        DRIVER_DEADLINE_MS,
        nullptr);

    EXPECT_EQ("HAVEDATA", data_status);
    EXPECT_EQ("FORCEREADY", response.header);
    EXPECT_EQ(0, result.exit_code);
    EXPECT_DOUBLE_EQ(4.0, response.energy_hartree);
    EXPECT_EQ(2, response.nat);
    EXPECT_THAT(response.forces_hartree_per_bohr,
                testing::ElementsAre(1.0, -2.0, 3.0, 4.0, -5.0, 6.0));
    const double expected_virial[9] = {11.578, 23.156, 34.734,
                                        23.156, 46.312, 57.89,
                                        34.734, 57.89, 69.468};
    for (int index = 0; index < 9; ++index)
    {
        EXPECT_NEAR(expected_virial[index], response.virial_wire_hartree[index], 1.0e-12);
    }
    EXPECT_EQ(0, response.extra_bytes);

    const double expected_latvec[9] = {1.0, 0.05, 0.1,
                                       0.15, 1.5, 0.2,
                                       0.25, 0.3, 2.0};
    for (int index = 0; index < 9; ++index)
    {
        EXPECT_DOUBLE_EQ(expected_latvec[index], result.observation.latvec[index]);
    }
    EXPECT_DOUBLE_EQ(2.0, result.observation.lat0);
    EXPECT_DOUBLE_EQ(23.156, result.observation.omega);
    const double expected_taud[6] = {0.2, 0.3, 0.4, 0.7, 0.1, 0.2};
    const double expected_tau[6] = {0.345, 0.58, 0.88, 0.765, 0.245, 0.49};
    for (int index = 0; index < 6; ++index)
    {
        EXPECT_NEAR(expected_taud[index], result.observation.taud[index], 1.0e-12);
        EXPECT_NEAR(expected_tau[index], result.observation.tau[index], 1.0e-12);
    }
    EXPECT_EQ(1, result.observation.ionic_position_updated);
    EXPECT_EQ(1, result.observation.cell_parameter_updated);
    EXPECT_EQ(1, result.observation.force_calls);
    EXPECT_EQ(1, result.observation.stress_calls);
    ASSERT_EQ(2, result.observation.event_count);
    EXPECT_EQ(1, result.observation.events[0]);
    EXPECT_EQ(2, result.observation.events[1]);
}

TEST(SocketDriverTest, FixedModeReturnsProtocolZeroVirialWhenStressIsDisabled)
{
    ForceResponse response;
    const DriverResult result = run_driver(
        DriverConfig(),
        [&](const int fd, const MonotonicDeadline& deadline) {
            send_init(fd, deadline);
            send_posdata(fd, fixed_frame(), deadline);
            response = request_force(fd, deadline);
        },
        ChildMode::run_driver,
        DRIVER_DEADLINE_MS,
        nullptr);

    EXPECT_EQ(0, result.exit_code);
    EXPECT_EQ("FORCEREADY", response.header);
    EXPECT_THAT(response.virial_wire_hartree,
                testing::ElementsAre(0.0, 0.0, 0.0,
                                     0.0, 0.0, 0.0,
                                     0.0, 0.0, 0.0));
    EXPECT_EQ(0, result.observation.stress_calls);
    EXPECT_EQ(0, result.observation.cell_parameter_updated);
}

TEST(SocketDriverTest, ScaleAwareUnchangedCellDoesNotRequestCellRebuild)
{
    DriverConfig config;
    config.variable_cell = true;
    config.cal_stress = true;
    WireFrame frame = fixed_frame();
    const double delta = 16.0 * std::numeric_limits<double>::epsilon();
    frame.cell[0] += delta;
    frame.inverse[0] = 1.0 / frame.cell[0];
    ForceResponse response;
    const DriverResult result = run_driver(
        config,
        [&](const int fd, const MonotonicDeadline& deadline) {
            send_init(fd, deadline);
            send_posdata(fd, frame, deadline);
            response = request_force(fd, deadline);
        },
        ChildMode::run_driver,
        DRIVER_DEADLINE_MS,
        nullptr);

    EXPECT_EQ(0, result.exit_code);
    EXPECT_EQ("FORCEREADY", response.header);
    EXPECT_EQ(0, result.observation.cell_parameter_updated);
    EXPECT_DOUBLE_EQ(1.0, result.observation.latvec[0]);
    EXPECT_DOUBLE_EQ(1.0, result.observation.latvec[4]);
    EXPECT_DOUBLE_EQ(1.0, result.observation.latvec[8]);
}

TEST(SocketDriverTest, DirectCoordinatesUseRecomputedRatherThanReceivedInverse)
{
    DriverConfig config;
    config.variable_cell = true;
    config.cal_stress = true;
    WireFrame frame;
    frame.cell = {{1.0e-6, 0.0, 0.0,
                   0.0, 1.0, 0.0,
                   0.0, 0.0, 1.0}};
    frame.inverse = {{1.0e6 + 1.0e-3, 0.0, 0.0,
                      0.0, 1.0, 0.0,
                      0.0, 0.0, 1.0}};
    frame.positions = {5.0e-7, 0.25, 0.75, 2.5e-7, 0.1, 0.2};
    const DriverResult result = run_driver(
        config,
        [&](const int fd, const MonotonicDeadline& deadline) {
            send_init(fd, deadline);
            send_posdata(fd, frame, deadline);
            static_cast<void>(request_force(fd, deadline));
        },
        ChildMode::run_driver,
        DRIVER_DEADLINE_MS,
        nullptr);

    EXPECT_EQ(0, result.exit_code);
    EXPECT_NEAR(0.5, result.observation.taud[0], 1.0e-12);
    EXPECT_NEAR(0.25, result.observation.taud[1], 1.0e-12);
    EXPECT_NEAR(0.75, result.observation.taud[2], 1.0e-12);
}

TEST(SocketDriverTest, FixedModeRejectsChangedCell)
{
    WireFrame frame = fixed_frame();
    frame.cell[0] = 3.0;
    frame.inverse[0] = 1.0 / 3.0;
    const DriverResult result = run_driver(
        DriverConfig(),
        [&](const int fd, const MonotonicDeadline& deadline) {
            send_init(fd, deadline);
            send_posdata(fd, frame, deadline);
        },
        ChildMode::run_driver,
        DRIVER_DEADLINE_MS,
        nullptr);

    EXPECT_EQ(1, result.exit_code);
    EXPECT_THAT(result.diagnostic, testing::HasSubstr("fixed-cell socket mode"));
    EXPECT_EQ(0, result.observation.runner_calls);
}

TEST(SocketDriverTest, PosdataBeforeInitIsFatal)
{
    const DriverResult result = run_driver(
        DriverConfig(),
        [&](const int fd, const MonotonicDeadline& deadline) {
            send_posdata(fd, fixed_frame(), deadline);
            static_cast<void>(request_status(fd, deadline));
        },
        ChildMode::run_driver,
        DRIVER_DEADLINE_MS,
        nullptr);

    EXPECT_EQ(1, result.exit_code);
    EXPECT_THAT(result.diagnostic, testing::HasSubstr("POSDATA requires READY"));
    EXPECT_EQ(0, result.observation.runner_calls);
}

TEST(SocketDriverTest, DuplicatePosdataBeforeGetforceIsFatal)
{
    const DriverResult result = run_driver(
        DriverConfig(),
        [&](const int fd, const MonotonicDeadline& deadline) {
            send_init(fd, deadline);
            send_posdata(fd, fixed_frame(), deadline);
            EXPECT_EQ("HAVEDATA", request_status(fd, deadline));
            send_posdata(fd, fixed_frame(), deadline);
        },
        ChildMode::run_driver,
        DRIVER_DEADLINE_MS,
        nullptr);

    EXPECT_EQ(1, result.exit_code);
    EXPECT_THAT(result.diagnostic, testing::HasSubstr("POSDATA requires READY"));
    EXPECT_EQ(1, result.observation.runner_calls);
}

TEST(SocketDriverTest, GetforceWithoutDataIsFatal)
{
    ForceResponse response;
    const DriverResult result = run_driver(
        DriverConfig(),
        [&](const int fd, const MonotonicDeadline& deadline) {
            send_init(fd, deadline);
            response = request_force(fd, deadline);
        },
        ChildMode::run_driver,
        DRIVER_DEADLINE_MS,
        nullptr);

    EXPECT_TRUE(response.header.empty());
    EXPECT_EQ(1, result.exit_code);
    EXPECT_THAT(result.diagnostic, testing::HasSubstr("GETFORCE requires HAVEDATA"));
}

TEST(SocketDriverTest, UnknownHeaderIsFatal)
{
    const DriverResult result = run_driver(
        DriverConfig(),
        [&](const int fd, const MonotonicDeadline& deadline) {
            send_header(fd, "BOGUS", deadline);
        },
        ChildMode::run_driver,
        DRIVER_DEADLINE_MS,
        nullptr);

    EXPECT_EQ(1, result.exit_code);
    EXPECT_THAT(result.diagnostic, testing::HasSubstr("unknown i-PI header"));
}

TEST(SocketDriverTest, ExitHeaderEndsDriverSuccessfully)
{
    const DriverResult result = run_driver(
        DriverConfig(),
        [&](const int fd, const MonotonicDeadline& deadline) {
            send_header(fd, "EXIT", deadline);
        },
        ChildMode::run_driver,
        DRIVER_DEADLINE_MS,
        nullptr);

    EXPECT_EQ(0, result.exit_code);
    EXPECT_THAT(result.diagnostic,
                testing::Not(testing::HasSubstr("unknown i-PI header")));
    EXPECT_EQ(0, result.observation.runner_calls);
}

TEST(SocketDriverTest, AtomCountMismatchIsFatalBeforeRunner)
{
    WireFrame frame = fixed_frame();
    frame.nat = 1;
    frame.positions.resize(3);
    const DriverResult result = run_driver(
        DriverConfig(),
        [&](const int fd, const MonotonicDeadline& deadline) {
            send_init(fd, deadline);
            send_posdata(fd, frame, deadline);
        },
        ChildMode::run_driver,
        DRIVER_DEADLINE_MS,
        nullptr);

    EXPECT_EQ(1, result.exit_code);
    EXPECT_THAT(result.diagnostic, testing::HasSubstr("atom count"));
    EXPECT_EQ(0, result.observation.runner_calls);
}

TEST(SocketDriverTest, InvalidReceivedInverseIsFatalBeforeRunner)
{
    WireFrame frame = fixed_frame();
    frame.inverse.fill(0.0);
    const DriverResult result = run_driver(
        DriverConfig(),
        [&](const int fd, const MonotonicDeadline& deadline) {
            send_init(fd, deadline);
            send_posdata(fd, frame, deadline);
            static_cast<void>(request_status(fd, deadline));
        },
        ChildMode::run_driver,
        DRIVER_DEADLINE_MS,
        nullptr);

    EXPECT_EQ(1, result.exit_code);
    EXPECT_THAT(result.diagnostic, testing::HasSubstr("inverse"));
    EXPECT_EQ(0, result.observation.runner_calls);
}

TEST(SocketDriverTest, NonfinitePositionIsFatalBeforeRunner)
{
    WireFrame frame = fixed_frame();
    frame.positions[2] = std::numeric_limits<double>::quiet_NaN();
    const DriverResult result = run_driver(
        DriverConfig(),
        [&](const int fd, const MonotonicDeadline& deadline) {
            send_init(fd, deadline);
            send_posdata(fd, frame, deadline);
        },
        ChildMode::run_driver,
        DRIVER_DEADLINE_MS,
        nullptr);

    EXPECT_EQ(1, result.exit_code);
    EXPECT_THAT(result.diagnostic, testing::HasSubstr("position coordinates must be finite"));
    EXPECT_EQ(0, result.observation.runner_calls);
}

TEST(SocketDriverTest, MidFrameDisconnectIsFatal)
{
    const DriverResult result = run_driver(
        DriverConfig(),
        [&](const int fd, const MonotonicDeadline& deadline) {
            send_init(fd, deadline);
            send_header(fd, "POSDATA", deadline);
            const double partial_cell[3] = {2.0, 0.0, 0.0};
            send_all(fd, partial_cell, sizeof(partial_cell), deadline);
            ASSERT_EQ(0, ::shutdown(fd, SHUT_WR));
        },
        ChildMode::run_driver,
        DRIVER_DEADLINE_MS,
        nullptr);

    EXPECT_EQ(1, result.exit_code);
    EXPECT_THAT(result.diagnostic, testing::HasSubstr("closed while reading"));
    EXPECT_EQ(0, result.observation.runner_calls);
}

TEST(SocketDriverTest, FailedSecondFrameCannotRepublishFirstResult)
{
    ForceResponse first_response;
    std::string second_status;
    WireFrame invalid = fixed_frame();
    invalid.inverse.fill(0.0);
    const DriverResult result = run_driver(
        DriverConfig(),
        [&](const int fd, const MonotonicDeadline& deadline) {
            send_init(fd, deadline);
            send_posdata(fd, fixed_frame(), deadline);
            first_response = request_force(fd, deadline);
            send_posdata(fd, invalid, deadline);
            second_status = request_status(fd, deadline);
        },
        ChildMode::run_driver,
        DRIVER_DEADLINE_MS,
        nullptr);

    EXPECT_EQ("FORCEREADY", first_response.header);
    EXPECT_TRUE(second_status.empty());
    EXPECT_EQ(1, result.exit_code);
    EXPECT_EQ(1, result.observation.runner_calls);
}

TEST(SocketDriverTest, PeerCloseWithComputedFramePendingIsFatal)
{
    const DriverResult result = run_driver(
        DriverConfig(),
        [&](const int fd, const MonotonicDeadline& deadline) {
            send_init(fd, deadline);
            send_posdata(fd, fixed_frame(), deadline);
        },
        ChildMode::run_driver,
        DRIVER_DEADLINE_MS,
        nullptr);

    EXPECT_EQ(1, result.exit_code);
    EXPECT_THAT(result.diagnostic, testing::HasSubstr("computed frame was pending"));
    EXPECT_EQ(1, result.observation.runner_calls);
}

TEST(SocketDriverTest, RunnerFailureAfterCellCommitIsFatalWithoutRollback)
{
    DriverConfig config;
    config.variable_cell = true;
    config.cal_stress = true;
    config.solver.throw_in_runner = true;
    const DriverResult result = run_driver(
        config,
        [&](const int fd, const MonotonicDeadline& deadline) {
            send_init(fd, deadline);
            send_posdata(fd, triclinic_frame(), deadline);
            static_cast<void>(request_status(fd, deadline));
        },
        ChildMode::run_driver,
        DRIVER_DEADLINE_MS,
        nullptr);

    EXPECT_EQ(1, result.exit_code);
    EXPECT_THAT(result.diagnostic, testing::HasSubstr("fake runner failed after commit"));
    EXPECT_EQ(1, result.observation.runner_calls);
    EXPECT_DOUBLE_EQ(1.0, result.observation.latvec[0]);
    EXPECT_DOUBLE_EQ(0.05, result.observation.latvec[1]);
    EXPECT_DOUBLE_EQ(2.0, result.observation.latvec[8]);
    EXPECT_EQ(1, result.observation.cell_parameter_updated);
}

TEST(SocketDriverTest, NonconvergedFrameIsNotPublished)
{
    DriverConfig config;
    config.solver.converged = false;
    std::string status;
    const DriverResult result = run_driver(
        config,
        [&](const int fd, const MonotonicDeadline& deadline) {
            send_init(fd, deadline);
            send_posdata(fd, fixed_frame(), deadline);
            status = request_status(fd, deadline);
        },
        ChildMode::run_driver,
        DRIVER_DEADLINE_MS,
        nullptr);

    EXPECT_TRUE(status.empty());
    EXPECT_EQ(1, result.exit_code);
    EXPECT_THAT(result.diagnostic, testing::HasSubstr("SCF did not converge"));
    EXPECT_EQ(0, result.observation.force_calls);
}

TEST(SocketDriverTest, NonfiniteEnergyIsNotPublished)
{
    DriverConfig config;
    config.solver.nonfinite_energy = true;
    std::string status;
    const DriverResult result = run_driver(
        config,
        [&](const int fd, const MonotonicDeadline& deadline) {
            send_init(fd, deadline);
            send_posdata(fd, fixed_frame(), deadline);
            status = request_status(fd, deadline);
        },
        ChildMode::run_driver,
        DRIVER_DEADLINE_MS,
        nullptr);

    EXPECT_TRUE(status.empty());
    EXPECT_EQ(1, result.exit_code);
    EXPECT_THAT(result.diagnostic, testing::HasSubstr("energy must be finite"));
}

TEST(SocketDriverTest, NonfiniteForceIsNotPublished)
{
    DriverConfig config;
    config.solver.nonfinite_force = true;
    std::string status;
    const DriverResult result = run_driver(
        config,
        [&](const int fd, const MonotonicDeadline& deadline) {
            send_init(fd, deadline);
            send_posdata(fd, fixed_frame(), deadline);
            status = request_status(fd, deadline);
        },
        ChildMode::run_driver,
        DRIVER_DEADLINE_MS,
        nullptr);

    EXPECT_TRUE(status.empty());
    EXPECT_EQ(1, result.exit_code);
    EXPECT_THAT(result.diagnostic, testing::HasSubstr("force entries must be finite"));
}

TEST(SocketDriverTest, NonfiniteStressIsNotPublished)
{
    DriverConfig config;
    config.cal_stress = true;
    config.solver.nonfinite_stress = true;
    std::string status;
    const DriverResult result = run_driver(
        config,
        [&](const int fd, const MonotonicDeadline& deadline) {
            send_init(fd, deadline);
            send_posdata(fd, fixed_frame(), deadline);
            status = request_status(fd, deadline);
        },
        ChildMode::run_driver,
        DRIVER_DEADLINE_MS,
        nullptr);

    EXPECT_TRUE(status.empty());
    EXPECT_EQ(1, result.exit_code);
    EXPECT_THAT(result.diagnostic, testing::HasSubstr("stress entries must be finite"));
}

TEST(SocketDriverTest, ChildOwnershipReapsUnresponsiveChild)
{
    const ChildCleanupProbe probe = child_scope_reaps_unresponsive_child();

    EXPECT_TRUE(probe.child_ready);
    EXPECT_TRUE(probe.wait_timed_out);
    EXPECT_TRUE(probe.child_reaped);
    EXPECT_LT(probe.elapsed_ms, 1000);
}

TEST(SocketDriverTest, ChildBeforeConnectTimesOutAndIsReaped)
{
    const std::chrono::steady_clock::time_point started = std::chrono::steady_clock::now();
    pid_t child = -1;
    bool accept_timed_out = false;

    try
    {
        static_cast<void>(run_driver(DriverConfig(),
                                     [](const int, const MonotonicDeadline&) {},
                                     ChildMode::stall_before_connect,
                                     50,
                                     &child));
        FAIL() << "driver should time out when its child never connects";
    }
    catch (const std::runtime_error& error)
    {
        accept_timed_out = std::string(error.what()).find("socket accept timed out") != std::string::npos;
    }

    errno = 0;
    int status = 0;
    const bool child_reaped = child > 0 && ::waitpid(child, &status, WNOHANG) < 0 && errno == ECHILD;
    const long long elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                     std::chrono::steady_clock::now() - started)
                                     .count();
    EXPECT_TRUE(accept_timed_out);
    EXPECT_TRUE(child_reaped);
    EXPECT_LT(elapsed_ms, 1000);
}

TEST(SocketDriverTest, PipeDrainUsesMonotonicDeadline)
{
    int pipe_fds[2];
    ASSERT_EQ(0, ::pipe(pipe_fds));
    UniqueFd pipe_read(pipe_fds[0]);
    UniqueFd pipe_write(pipe_fds[1]);
    set_nonblocking(pipe_read.get());
    const MonotonicDeadline deadline(50);

    try
    {
        static_cast<void>(read_pipe(pipe_read.get(), deadline));
        FAIL() << "pipe drain should time out while a silent writer remains open";
    }
    catch (const std::runtime_error& error)
    {
        EXPECT_THAT(error.what(), testing::HasSubstr("diagnostic pipe read timed out"));
    }
}
