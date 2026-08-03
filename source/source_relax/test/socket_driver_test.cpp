#include "source_relax/socket_driver.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "source_cell/unitcell.h"
#include "source_esolver/esolver.h"
#include "source_io/module_parameter/input_parameter.h"
#include "for_test.h"

#include <cerrno>
#include <chrono>
#include <climits>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <poll.h>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

namespace unitcell
{
void periodic_boundary_adjustment(Atom* atoms, const ModuleBase::Matrix3& latvec, const int ntype)
{
    for (int it = 0; it < ntype; ++it)
    {
        for (int ia = 0; ia < atoms[it].na; ++ia)
        {
            atoms[it].tau[ia] = atoms[it].taud[ia] * latvec;
        }
    }
}
} // namespace unitcell

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

class FakeESolver : public ModuleESolver::ESolver
{
  public:
    explicit FakeESolver(const bool converged) : converged_(converged)
    {
    }

    void before_all_runners(BaseCell&, const Input_para&) override
    {
    }

    void runner(BaseCell&, const int) override
    {
        this->conv_esolver = converged_;
    }

    void after_all_runners(BaseCell&) override
    {
    }

    double cal_energy() override
    {
        return 4.0;
    }

    void cal_force(BaseCell& cell, ModuleBase::matrix& force) override
    {
        force.create(cell.nat(), 3);
    }

    void cal_stress(BaseCell&, ModuleBase::matrix& stress) override
    {
        stress.create(3, 3);
    }

  private:
    bool converged_;
};

struct DriverResult
{
    int exit_code = -1;
    std::string response_header;
    std::string diagnostic;
};

enum class ChildMode
{
    run_driver,
    stall_before_connect
};

void initialize_one_atom_cell(UnitCell& ucell)
{
    ucell.lat0 = 1.0;
    ucell.latvec.Identity();
    ucell.ntype = 1;
    ucell.nat = 1;
    ucell.atoms[0].na = 1;
    ucell.atoms[0].tau.resize(1);
    ucell.atoms[0].taud.resize(1);
    ucell.atoms[0].dis.resize(1);
}

void send_fixed_cell_frame(const int fd, const MonotonicDeadline& deadline)
{
    const std::int32_t replica = 0;
    const std::int32_t parameter_bytes = 0;
    send_header(fd, "INIT", deadline);
    send_value(fd, replica, deadline);
    send_value(fd, parameter_bytes, deadline);

    const double identity[9] = {1.0, 0.0, 0.0,
                                0.0, 1.0, 0.0,
                                0.0, 0.0, 1.0};
    const std::int32_t nat = 1;
    const double position[3] = {0.0, 0.0, 0.0};
    send_header(fd, "POSDATA", deadline);
    send_all(fd, identity, sizeof(identity), deadline);
    send_all(fd, identity, sizeof(identity), deadline);
    send_value(fd, nat, deadline);
    send_all(fd, position, sizeof(position), deadline);
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

DriverResult run_driver_frame(const bool converged,
                              const ChildMode child_mode,
                              const int timeout_ms,
                              pid_t* observed_child)
{
    const MonotonicDeadline deadline(timeout_ms);
    UnixSocketServer server;
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
        initialize_one_atom_cell(ucell);
        Input_para input;
        input.cal_force = true;
        FakeESolver solver(converged);
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
    send_fixed_cell_frame(peer.get(), deadline);
    if (try_send_status(peer.get(), deadline))
    {
        result.response_header = read_header_or_close(peer.get(), deadline);
    }
    peer.reset();

    result.diagnostic = read_pipe(output_read.get(), deadline);
    output_read.reset();
    const int status = child_process.wait_until(deadline);
    if (WIFEXITED(status))
    {
        result.exit_code = WEXITSTATUS(status);
    }
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

TEST(SocketDriverTest, NonconvergedFrameIsNotPublished)
{
    const DriverResult result = run_driver_frame(false, ChildMode::run_driver, DRIVER_DEADLINE_MS, nullptr);

    EXPECT_TRUE(result.response_header.empty()) << "unexpected response: " << result.response_header;
    EXPECT_EQ(1, result.exit_code);
    EXPECT_THAT(result.diagnostic, testing::HasSubstr("SCF did not converge"));
}

TEST(SocketDriverTest, ConvergedFrameReachesHaveData)
{
    const DriverResult result = run_driver_frame(true, ChildMode::run_driver, DRIVER_DEADLINE_MS, nullptr);

    EXPECT_EQ("HAVEDATA", result.response_header);
    EXPECT_EQ(0, result.exit_code);
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
        static_cast<void>(run_driver_frame(true, ChildMode::stall_before_connect, 50, &child));
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
