#include <chrono>
#include <iostream>

#include <Eigen/Dense>

#include "finite_differences.hpp"
#include "integrator.hpp"
#include "line_search.hpp"
#include "lqr.hpp"
#include "multi_agent_lqr.hpp"
#include "ocp.hpp"
#include "single_track_ocp.hpp"
#include "solver_output.hpp"
#include "solvers/constrained_gradient_descent.hpp"
#include "solvers/gradient_descent.hpp"
#include "solvers/ilqr.hpp"
#include "solvers/osqp_solver.hpp"
#include "types.hpp"

int
main( int /*num_arguments*/, char** /*arguments*/ )
{
  // lqr_test( false );

  // lqr_test( true );

  // single_track_test();
  multi_agent_lqr_example();
  return 0;
}
