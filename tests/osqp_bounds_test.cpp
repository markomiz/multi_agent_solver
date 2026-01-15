#ifdef MAS_HAVE_OSQP
#include <gtest/gtest.h>

#include "multi_agent_solver/ocp.hpp"
#include "multi_agent_solver/solvers/osqp.hpp"
#include "multi_agent_solver/solvers/osqp_collocation.hpp"

namespace mas
{
namespace
{

MotionModel
create_integrator()
{
  return []( const State& state, const Control& control ) { return control; };
}

} // namespace

class OSQPBoundsTest : public ::testing::Test
{
protected:
  void
  SetUp() override
  {
    ocp.state_dim     = 1;
    ocp.control_dim   = 1;
    ocp.horizon_steps = 10;
    ocp.dt            = 0.1;
    ocp.initial_state = State::Constant( 1, 5.0 ); // Start at x=5
    ocp.dynamics      = create_integrator();
    
    // Cost: minimize x^2 + u^2
    // Without bounds, x should go to 0.
    ocp.stage_cost = []( const State& x, const Control& u, size_t ) {
      return x.squaredNorm() + u.squaredNorm();
    };
    ocp.terminal_cost = []( const State& x ) { return 10.0 * x.squaredNorm(); };

    // State Lower Bound: x >= 2.0
    // The solver should stop at x=2.0
    ocp.state_lower_bounds = State::Constant( 1, 2.0 );
    
    // Initialize
    ocp.initialize_problem();
  }

  OCP ocp;
};

TEST_F( OSQPBoundsTest, OSQPSolverRespectsStateBounds )
{
  // Configure Solver
  SolverParams params;
  params["max_iterations"] = 10;
  params["tolerance"]      = 1e-4;
  params["max_ms"]         = 1000.0;
  params["debug"]          = 1.0;

  OSQP solver;
  solver.set_params( params );
  solver.solve( ocp );

  // Check results
  const auto& states = ocp.best_states;
  
  // Verify that NO state is significantly below 2.0
  double min_state = states.minCoeff();
  std::cout << "DEBUG: Min state value (OSQP) = " << min_state << std::endl;

  // Allow a small tolerance for soft constraint/numerical issues
  EXPECT_GE( min_state, 1.99 ) << "State trajectory violated lower bound of 2.0";
  
  // Also verify that it actually went down (it started at 5)
  EXPECT_LT( states.col( ocp.horizon_steps )(0), 4.0 ) << "State did not decrease as expected";
}

TEST_F( OSQPBoundsTest, OSQPCollocationSolverRespectsStateBounds )
{
  // Configure Solver
  SolverParams params;
  params["max_iterations"] = 20;
  params["tolerance"]      = 1e-4;
  params["max_ms"]         = 1000.0;
  params["debug"]          = 1.0;

  OSQPCollocation solver;
  solver.set_params( params );
  solver.solve( ocp );

  // Check results
  const auto& states = ocp.best_states;
  
  // Verify that NO state is significantly below 2.0
  double min_state = states.minCoeff();
  std::cout << "DEBUG: Min state value (OSQPCollocation) = " << min_state << std::endl;

  // Allow a small tolerance
  EXPECT_GE( min_state, 1.99 ) << "State trajectory violated lower bound of 2.0";
  
   // Also verify that it actually went down (it started at 5)
  EXPECT_LT( states.col( ocp.horizon_steps )(0), 4.0 ) << "State did not decrease as expected";
}

} // namespace mas
#endif // MAS_HAVE_OSQP
