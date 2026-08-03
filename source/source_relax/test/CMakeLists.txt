abacus_disable_feature_definitions(__MPI)
abacus_disable_feature_definitions(__LCAO)
abacus_disable_feature_definitions(__MLALGO)
abacus_disable_feature_definitions(__CUDA)
abacus_disable_feature_definitions(__ROCM)

install(DIRECTORY support DESTINATION ${CMAKE_CURRENT_BINARY_DIR})


AddTest(
  TARGET MODULE_RELAX_socket_ipi_test
  SOURCES socket_ipi_test.cpp ../socket_ipi.cpp
)

AddTest(
  TARGET MODULE_RELAX_socket_frame_test
  SOURCES socket_frame_test.cpp ../socket_frame.cpp
)

AddTest(
  TARGET MODULE_RELAX_socket_driver_test
  LIBS base device
  SOURCES socket_driver_test.cpp
          ../socket_driver.cpp
          ../socket_frame.cpp
          ../socket_ipi.cpp
          ../../source_cell/update_cell.cpp
          ../../source_cell/bcast_cell.cpp
)
set_tests_properties(MODULE_RELAX_socket_driver_test PROPERTIES TIMEOUT 15)

add_executable(
  MODULE_RELAX_socket_driver_mpi_failure_test
  socket_driver_mpi_failure_test.cpp
  ../socket_driver.cpp
  ../socket_frame.cpp
  ../socket_ipi.cpp
)
target_compile_definitions(MODULE_RELAX_socket_driver_mpi_failure_test PRIVATE __MPI)
target_link_libraries(
  MODULE_RELAX_socket_driver_mpi_failure_test
  PRIVATE base device cell_info MPI::MPI_CXX Threads::Threads abacus::linalg_libs
)
find_program(ABACUS_BASH_EXECUTABLE bash)
if(NOT ABACUS_BASH_EXECUTABLE)
  message(FATAL_ERROR "bash is required for the socket driver MPI tests")
endif()
list(LENGTH MPIEXEC_NUMPROC_FLAG socket_mpi_numproc_flag_count)
if(socket_mpi_numproc_flag_count GREATER 1)
  message(FATAL_ERROR "MPIEXEC_NUMPROC_FLAG must contain at most one flag")
endif()
list(LENGTH MPIEXEC_PREFLAGS socket_mpi_preflags_count)
list(LENGTH MPIEXEC_POSTFLAGS socket_mpi_postflags_count)
add_test(
  NAME MODULE_RELAX_socket_driver_mpi_failure_wrapper_test
  COMMAND
    ${ABACUS_BASH_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/socket_driver_mpi_failure_wrapper_test.sh
    ${ABACUS_BASH_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/socket_driver_mpi_failure_test.sh
)
set_tests_properties(
  MODULE_RELAX_socket_driver_mpi_failure_wrapper_test
  PROPERTIES TIMEOUT 5
)
add_test(
  NAME MODULE_RELAX_socket_driver_mpi_failure_test
  COMMAND
    ${ABACUS_BASH_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/socket_driver_mpi_failure_test.sh
    ${MPIEXEC_EXECUTABLE}
    ${socket_mpi_numproc_flag_count}
    2
    $<TARGET_FILE:MODULE_RELAX_socket_driver_mpi_failure_test>
    ${socket_mpi_preflags_count}
    ${socket_mpi_postflags_count}
    ${MPIEXEC_NUMPROC_FLAG}
    ${MPIEXEC_PREFLAGS}
    ${MPIEXEC_POSTFLAGS}
  WORKING_DIRECTORY $<TARGET_FILE_DIR:MODULE_RELAX_socket_driver_mpi_failure_test>
)
set_tests_properties(
  MODULE_RELAX_socket_driver_mpi_failure_test
  PROPERTIES
    PROCESSORS 2
    TIMEOUT 10
    PASS_REGULAR_EXPRESSION
      "ABACUS_SOCKET_MPI_FATAL stage=runner rank=1 message=rank-selective runner failure"
)

AddTest(
  TARGET MODULE_RELAX_relax_new_line_search
  LIBS parameter
  SOURCES line_search_test.cpp ../line_search.cpp ../../source_base/global_variable.cpp ../../source_base/global_file.cpp ../../source_base/global_function.cpp ../../source_base/memory_recorder.cpp ../../source_base/timer.cpp ../../source_base/tool_quit.cpp
)

AddTest(
  TARGET MODULE_RELAX_relax_new_relax
  SOURCES  relax_test.cpp ../relax_sync.cpp ../line_search.cpp ../relax_data.cpp ../../source_base/tool_quit.cpp ../../source_base/global_variable.cpp ../../source_base/global_file.cpp ../../source_base/memory_recorder.cpp ../../source_base/timer.cpp
    ../../source_base/matrix3.cpp ../../source_base/intarray.cpp ../../source_base/tool_title.cpp
    ../../source_base/global_function.cpp ../../source_base/complexmatrix.cpp ../../source_base/matrix.cpp
    ../../source_base/complexarray.cpp ../../source_base/tool_quit.cpp ../../source_base/realarray.cpp
    ../../source_base/module_external/blas_connector_base.cpp ../../source_base/module_external/blas_connector_vector.cpp ../../source_base/module_external/blas_connector_matrix.cpp
    ../../source_cell/update_cell.cpp ../../source_cell/print_cell.cpp  ../../source_cell/bcast_cell.cpp ../../source_base/output.cpp
    LIBS parameter
)

list(APPEND cell_source_files
    ../../source_cell/update_cell.cpp
    ../../source_cell/bcast_cell.cpp
    ../../source_cell/print_cell.cpp
)
AddTest(
  TARGET MODULE_RELAX_lattice_change_methods_test
  LIBS parameter base device
  SOURCES lattice_change_methods_test.cpp ../lattice_change_methods.cpp ../lattice_change_basic.cpp ../relax_data.cpp mock_remake_cell.cpp
)

AddTest(
  TARGET MODULE_RELAX_lattice_change_basic_test
  LIBS parameter base device
  SOURCES lattice_change_basic_test.cpp ../lattice_change_basic.cpp ../relax_data.cpp mock_remake_cell.cpp
)

AddTest(
  TARGET MODULE_RELAX_lattice_change_cg_test
  LIBS parameter base device
  SOURCES lattice_change_cg_test.cpp
          ../lattice_change_cg.cpp
          ../lattice_change_basic.cpp
          ../cg_base.cpp
          ../relax_data.cpp
          mock_remake_cell.cpp
          ../../source_io/module_output/orb_io.cpp
)

AddTest(
  TARGET MODULE_RELAX_bfgs_basic_test
  LIBS parameter base device symmetry
  SOURCES bfgs_basic_test.cpp ../bfgs_basic.cpp ../relax_data.cpp ../ions_move_basic.cpp
          ../../source_io/module_output/orb_io.cpp
          ${cell_source_files}
)


AddTest(
  TARGET MODULE_RELAX_bfgs_test
  LIBS parameter base device
  SOURCES bfgs_test.cpp ../ions_move_bfgs2.cpp ../ions_move_basic.cpp ../matrix_methods.cpp ../relax_data.cpp ${cell_source_files}
)

AddTest(
  TARGET MODULE_RELAX_ions_move_basic_test
  LIBS parameter base device
  SOURCES ions_move_basic_test.cpp ../ions_move_basic.cpp ../relax_data.cpp ${cell_source_files}
)

AddTest(
  TARGET MODULE_RELAX_ions_move_bfgs_test
  LIBS parameter base device
  SOURCES ions_move_bfgs_test.cpp 
          ../ions_move_bfgs.cpp 
          ../ions_move_basic.cpp 
          ../bfgs_basic.cpp
          ../relax_data.cpp
          ../../source_io/module_output/orb_io.cpp
          ${cell_source_files}
)

AddTest(
  TARGET MODULE_RELAX_ions_move_methods_test
  LIBS parameter base device
  SOURCES ions_move_methods_test.cpp
          ../ions_move_methods.cpp
          ../ions_move_bfgs.cpp
          ../ions_move_bfgs2.cpp
          ../ions_move_cg.cpp
          ../ions_move_sd.cpp
          ../ions_move_lbfgs.cpp
          ../ions_move_basic.cpp
          ../bfgs_basic.cpp
          ../cg_base.cpp
          ../matrix_methods.cpp
          ../relax_data.cpp
          ../../source_io/module_output/orb_io.cpp
          ../../source_cell/bcast_cell.cpp
          ../../source_cell/print_cell.cpp
)

AddTest(
  TARGET MODULE_RELAX_ions_move_cg_test
  LIBS parameter base device
  SOURCES ions_move_cg_test.cpp 
          ../ions_move_cg.cpp 
          ../cg_base.cpp
          ../ions_move_basic.cpp
          ../relax_data.cpp
          ../../source_io/module_output/orb_io.cpp
          ${cell_source_files}
)

AddTest(
  TARGET MODULE_RELAX_ions_move_sd_test
  LIBS parameter base device
  SOURCES ions_move_sd_test.cpp ../ions_move_sd.cpp ../ions_move_basic.cpp ../relax_data.cpp ${cell_source_files}
)
