#include <chrono>
#include <iostream>

#include <Eigen/Dense>

#include "finite_differences.hpp"
#include "integrator.hpp"
#include "line_search.hpp"
#include "multi_agent_lqr.hpp"
#include "multi_agent_single_track.hpp"
#include "ocp.hpp"
#include "single_track_ocp.hpp"
#include "solvers/cgd.hpp"
#include "solvers/ilqr.hpp"
#include "solvers/osqp_solver.hpp"
#include "types.hpp"

int
main( int /*num_arguments*/, char** /*arguments*/ )
{
  single_track_test();
  multi_agent_lqr_example();
  // multi_agent_circular_test( 1, 3 );
  // multi_agent_circular_test( 1, 10 );
  multi_agent_circular_test( 10, 15 );
  // multi_agent_circular_test( 8, 20 );
  // multi_agent_circular_test( 8, 30 );


  return 0;
}
