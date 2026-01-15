#include <gtest/gtest.h>
#include <memory>

#include "multi_agent_solver/agent.hpp"
#include "multi_agent_solver/ocp.hpp"

namespace mas
{

TEST( AgentTest, ConstructorInitializesMembers )
{
  auto ocp = std::make_shared<OCP>();
  ocp->state_dim   = 2;
  ocp->control_dim = 1;
  int id = 5;

  Agent agent( id, ocp );

  EXPECT_EQ( agent.id, id );
  EXPECT_EQ( agent.ocp, ocp );
}

TEST( AgentTest, UpdateInitialWithBestCallsOCPMethod )
{
  auto ocp = std::make_shared<OCP>();
  ocp->state_dim     = 1;
  ocp->control_dim   = 1;
  ocp->horizon_steps = 2;
  ocp->dt            = 0.1;

  ocp->best_controls = ControlTrajectory::Constant( ocp->control_dim, ocp->horizon_steps, 5.0 );
  ocp->best_states   = StateTrajectory::Constant( ocp->state_dim, ocp->horizon_steps + 1, 10.0 );
  
  // Initialize 'initial' with something else
  ocp->initial_controls = ControlTrajectory::Zero( ocp->control_dim, ocp->horizon_steps );
  ocp->initial_states   = StateTrajectory::Zero( ocp->state_dim, ocp->horizon_steps + 1 );

  Agent agent( 1, ocp );
  agent.update_initial_with_best();

  EXPECT_TRUE( ocp->initial_controls.isApprox( ocp->best_controls ) );
  EXPECT_TRUE( ocp->initial_states.isApprox( ocp->best_states ) );
}

} // namespace mas
