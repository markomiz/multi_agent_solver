#include <gtest/gtest.h>

#include "multi_agent_solver/agent.hpp"
#include "multi_agent_solver/finite_differences.hpp"
#include "multi_agent_solver/multi_agent_problem.hpp"
#include "multi_agent_solver/ocp.hpp"

namespace mas
{
namespace
{

MotionModel
create_integrator()
{
  return []( const State& state, const Control& control ) { return control + state * 0.0; };
}

} // namespace

TEST( OCPTest, InitializeProblemSetsDefaultsAndBestCost )
{
  OCP ocp;
  ocp.state_dim     = 1;
  ocp.control_dim   = 1;
  ocp.horizon_steps = 3;
  ocp.dt            = 0.1;
  ocp.initial_state = State::Zero( ocp.state_dim );
  ocp.dynamics      = create_integrator();
  ocp.stage_cost    = []( const State& x, const Control& u, size_t ) {
    return x.squaredNorm() + u.squaredNorm();
  };
  ocp.terminal_cost = []( const State& x ) { return x.squaredNorm(); };

  ocp.initialize_problem();

  EXPECT_EQ( ocp.best_states.rows(), ocp.state_dim );
  EXPECT_EQ( ocp.best_states.cols(), ocp.horizon_steps + 1 );
  EXPECT_EQ( ocp.best_controls.rows(), ocp.control_dim );
  EXPECT_EQ( ocp.best_controls.cols(), ocp.horizon_steps );
  EXPECT_DOUBLE_EQ( ocp.best_cost, 0.0 );

  ASSERT_TRUE( static_cast<bool>( ocp.cost_state_gradient ) );
  ASSERT_TRUE( static_cast<bool>( ocp.cost_control_gradient ) );

  auto state_grad = ocp.cost_state_gradient( ocp.stage_cost, ocp.best_states.col( 0 ),
                                             ocp.best_controls.col( 0 ), 0 );
  auto control_grad = ocp.cost_control_gradient( ocp.stage_cost, ocp.best_states.col( 0 ),
                                                 ocp.best_controls.col( 0 ), 0 );

  EXPECT_EQ( state_grad.size(), ocp.state_dim );
  EXPECT_EQ( control_grad.size(), ocp.control_dim );
  EXPECT_TRUE( ocp.verify_problem() );
}

TEST( OCPTest, UpdateInitialWithBestCopiesTrajectories )
{
  OCP ocp;
  ocp.state_dim     = 2;
  ocp.control_dim   = 2;
  ocp.horizon_steps = 2;
  ocp.dt            = 1.0;
  ocp.initial_state = State::Zero( ocp.state_dim );
  ocp.dynamics      = create_integrator();
  ocp.initialize_problem();

  ocp.best_controls = ControlTrajectory::Ones( ocp.control_dim, ocp.horizon_steps );
  ocp.best_states   = StateTrajectory::Ones( ocp.state_dim, ocp.horizon_steps + 1 );

  ocp.update_initial_with_best();

  EXPECT_TRUE( ocp.initial_controls.isApprox( ocp.best_controls ) );
  EXPECT_TRUE( ocp.initial_states.isApprox( ocp.best_states ) );
}

TEST( MultiAgentProblemTest, BuildGlobalProblemMergesAgents )
{
  auto ocp_a = std::make_shared<OCP>();
  ocp_a->state_dim     = 2;
  ocp_a->control_dim   = 1;
  ocp_a->horizon_steps = 2;
  ocp_a->dt            = 0.5;
  ocp_a->initial_state = State::Ones( ocp_a->state_dim );
  ocp_a->dynamics = []( const State& x, const Control& u ) {
    return x + u.replicate( x.size(), 1 );
  };
  ocp_a->stage_cost = []( const State& x, const Control& u, size_t ) { return x.sum() + u.sum(); };
  ocp_a->terminal_cost = []( const State& x ) { return 2.0 * x.sum(); };
  ocp_a->input_lower_bounds = Control::Constant( ocp_a->control_dim, -1.0 );
  ocp_a->input_upper_bounds = Control::Constant( ocp_a->control_dim, 1.0 );
  ocp_a->initialize_problem();

  auto ocp_b = std::make_shared<OCP>();
  ocp_b->state_dim     = 1;
  ocp_b->control_dim   = 2;
  ocp_b->horizon_steps = 2;
  ocp_b->dt            = 0.5;
  ocp_b->initial_state = State::Constant( ocp_b->state_dim, 3.0 );
  ocp_b->dynamics      = []( const State& x, const Control& u ) {
    return x + Control::Constant( x.size(), 2.0 * u.sum() );
  };
  ocp_b->stage_cost = []( const State& x, const Control& u, size_t ) {
    return 2.0 * x.sum() + 3.0 * u.sum();
  };
  ocp_b->terminal_cost = []( const State& x ) { return x.sum(); };
  ocp_b->input_lower_bounds = Control::Constant( ocp_b->control_dim, -2.0 );
  ocp_b->input_upper_bounds = Control::Constant( ocp_b->control_dim, 2.0 );
  ocp_b->initialize_problem();

  MultiAgentProblem problem;
  problem.add_agent( std::make_shared<Agent>( 2, ocp_b ) );
  problem.add_agent( std::make_shared<Agent>( 1, ocp_a ) );
  problem.compute_offsets();

  ASSERT_EQ( problem.blocks.size(), 2 );
  EXPECT_EQ( problem.blocks.front().agent_id, 1 );
  EXPECT_EQ( problem.blocks.back().agent_id, 2 );
  EXPECT_EQ( problem.blocks.front().state_offset, 0 );
  EXPECT_EQ( problem.blocks.front().control_offset, 0 );
  EXPECT_EQ( problem.blocks.back().state_offset, ocp_a->state_dim );
  EXPECT_EQ( problem.blocks.back().control_offset, ocp_a->control_dim );

  OCP global = problem.build_global_ocp();

  EXPECT_EQ( global.state_dim, ocp_a->state_dim + ocp_b->state_dim );
  EXPECT_EQ( global.control_dim, ocp_a->control_dim + ocp_b->control_dim );
  EXPECT_EQ( global.horizon_steps, ocp_a->horizon_steps );
  EXPECT_DOUBLE_EQ( global.dt, ocp_a->dt );

  ASSERT_TRUE( global.input_lower_bounds.has_value() );
  ASSERT_TRUE( global.input_upper_bounds.has_value() );
  EXPECT_DOUBLE_EQ( ( *global.input_lower_bounds )( 0 ), -1.0 );
  EXPECT_DOUBLE_EQ( ( *global.input_lower_bounds )( 1 ), -2.0 );
  EXPECT_DOUBLE_EQ( ( *global.input_lower_bounds )( 2 ), -2.0 );

  State expected_initial( global.state_dim );
  expected_initial << 1.0, 1.0, 3.0;
  EXPECT_TRUE( global.initial_state.isApprox( expected_initial ) );

  State state = State::LinSpaced( global.state_dim, 1.0, 3.0 );
  Control control = Control::LinSpaced( global.control_dim, -1.0, 1.0 );
  StateDerivative derivative = global.dynamics( state, control );

  EXPECT_EQ( derivative.size(), global.state_dim );
  EXPECT_DOUBLE_EQ( derivative( 0 ), state( 0 ) + control( 0 ) );
  EXPECT_DOUBLE_EQ( derivative( 1 ), state( 1 ) + control( 0 ) );
  EXPECT_DOUBLE_EQ( derivative( 2 ), state( 2 ) + 2.0 * control.tail( 2 ).sum() );

  double expected_stage_cost = ( state.segment( 0, 2 ).sum() + control.segment( 0, 1 ).sum() )
                               + ( 2.0 * state.tail( 1 ).sum()
                                   + 3.0 * control.tail( 2 ).sum() );
  EXPECT_DOUBLE_EQ( global.stage_cost( state, control, 0 ), expected_stage_cost );
  EXPECT_DOUBLE_EQ( global.terminal_cost( state ), 2.0 * state.segment( 0, 2 ).sum() + state.tail( 1 ).sum() );
}

TEST( FiniteDifferencesTest, GradientMatchesAnalyticalForQuadraticObjective )
{
  MotionModel dynamics = []( const State&, const Control& u ) { return u; };
  ObjectiveFunction objective = []( const StateTrajectory& states, const ControlTrajectory& controls ) {
    double state_sum = states.array().square().sum();
    double control_sum = controls.array().square().sum();
    return state_sum + control_sum;
  };

  State initial_state = State::Zero( 1 );
  ControlTrajectory controls( 1, 2 );
  controls << 1.0, -1.0;

  ControlGradient gradient
    = finite_differences_gradient( initial_state, controls, dynamics, objective, 1.0 );

  Control expected( 2 );
  expected << 4.0, -2.0;

  EXPECT_NEAR( gradient( 0, 0 ), expected( 0 ), 1e-3 );
  EXPECT_NEAR( gradient( 0, 1 ), expected( 1 ), 1e-3 );
}

} // namespace mas
