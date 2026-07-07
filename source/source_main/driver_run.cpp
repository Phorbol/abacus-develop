#include "source_main/driver.h"
#include "source_cell/check_atomic_stru.h"
#include "source_cell/module_neighbor/sltk_atom_arrange.h"
#include "source_relax/relax_driver.h"
#include "source_io/module_parameter/parameter.h"
#include "source_io/module_json/para_json.h"
#include "source_io/module_output/print_info.h"
#include "source_md/run_md.h"
#include "source_base/global_variable.h"
#include "source_base/module_device/device.h"
#include "source_base/module_device/memory_op.h"
#include "source_base/kernels/math_kernel_op.h"
#include "source_hsolver/kernels/hegvd_op.h"

#include <ATen/kernels/blas.h>
#include <ATen/kernels/lapack.h>

#ifdef __DSP
#include "source_base/kernels/dsp/dsp_connector.h"
#endif

/**
 * @brief This is the driver function which defines the workflow of ABACUS
 * calculations. It relies on the class Esolver, which is a class that organizes
 * workflows of single point calculations.
 *
 * For calculations involving change of configuration (lattice parameter & ionic
 * motion), this driver calls Esolver::Run and the configuration-changing
 * subroutine in a alternating manner.
 *
 * Information is passed between the two subroutines by class UnitCell
 *
 * Esolver::Run takes in a configuration and provides force and stress,
 * the configuration-changing subroutine takes force and stress and updates the
 * configuration
 */
void Driver::driver_run()
{
    ModuleBase::TITLE("Driver", "driver_run");

    const Parameter& param = PARAM;
    const Input_para& inp = param.inp;
    const std::string& cal = inp.calculation;
    const int my_rank = GlobalV::MY_RANK;
    auto& running_log = GlobalV::ofs_running;
    if (cal == "socket")
    {
        this->driver_ipi_run(param, my_rank, running_log);
        return;
    }

    //! 1: setup cell and atom information
    // this warning should not be here, mohan 2024-05-22
#ifndef __LCAO
    if (inp.basis_type == "lcao_in_pw" || inp.basis_type == "lcao") {
        ModuleBase::WARNING_QUIT("driver",
                                 "to use LCAO basis, compile with __LCAO");
    }
#endif

    // the life of ucell should begin here, mohan 2024-05-12
    UnitCell ucell;
    ucell.setup(inp.latname,
                inp.ntype,
                inp.lmaxmax,
                inp.init_vel,
                inp.fixed_axes);

    ucell.setup_cell(param.globalv.global_in_stru, running_log);
    unitcell::check_atomic_stru(ucell, inp.min_dist_coef);

    //! 2: initialize the ESolver (depends on a set-up ucell after `setup_cell`)
    this->init_hardware();

    ModuleESolver::ESolver* p_esolver = ModuleESolver::init_esolver(inp, ucell);

    //! 3: initialize Esolver and fill json-structure
    p_esolver->before_all_runners(ucell, inp);

    // this Json part should be moved to before_all_runners, mohan 2024-05-12
#ifdef __RAPIDJSON
    Json::gen_stru_wrapper(&ucell);
#endif

    //! 4: different types of calculations
    if (cal == "md")
    {
        Run_MD::md_line(ucell, p_esolver, param);
    }
    else if (cal == "scf" || cal == "relax" || cal == "cell-relax" || cal == "nscf")
    {
        Relax_Driver rl_driver;
        rl_driver.relax_driver(p_esolver, ucell, inp, running_log);
    }
    else if (cal == "get_s")
    {
        p_esolver->runner(ucell, 0);
    }
    else if (cal == "get_pchg" || cal == "get_wf" || cal == "gen_bessel" || cal == "gen_opt_abfs" ||
             cal == "test_memory" || cal == "test_neighbour")
    {
        const int istep = 0;
        p_esolver->others(ucell, istep);
    }
    else
    {
        ModuleBase::WARNING_QUIT("Driver::driver_run","cannot recognize the 'calculation' command");
    }

    //! 5: clean up esolver
    p_esolver->after_all_runners(ucell);

    delete p_esolver;

    this->finalize_hardware();

    //! 6: output the json file
    Json::create_Json(&ucell, param);

    return;
}

void Driver::init_hardware()
{
#if ((defined __CUDA) || (defined __ROCM)) || defined __DSP
    const Input_para& inp = PARAM.inp;
#endif
#ifdef __DSP
    const int my_rank = GlobalV::MY_RANK;
#endif
#if ((defined __CUDA) || (defined __ROCM))
    if (inp.device == "gpu")
    {
        ModuleBase::createGpuBlasHandle();
        hsolver::createGpuSolverHandle();
        container::kernels::createGpuBlasHandle();
        container::kernels::createGpuSolverHandle();
    }
#endif

#ifdef __DSP
    std::cout << " ** Initializing DSP Hardware..." << std::endl;
    mtfunc::dspInitHandle(my_rank % inp.dsp_count);
#endif
}

void Driver::finalize_hardware()
{
#if defined(__CUDA) || defined(__ROCM) || defined __DSP
    const Input_para& inp = PARAM.inp;
#endif
#ifdef __DSP
    const int my_rank = GlobalV::MY_RANK;
#endif
#if defined(__CUDA) || defined(__ROCM)
    if (inp.device == "gpu")
    {
        ModuleBase::destoryBLAShandle();
        hsolver::destroyGpuSolverHandle();
        container::kernels::destroyGpuBlasHandle();
        container::kernels::destroyGpuSolverHandle();
    }
#endif

#ifdef __DSP
    std::cout << " ** Closing DSP Hardware..." << std::endl;
    mtfunc::dspDestoryHandle(my_rank % inp.dsp_count);
#endif
}
