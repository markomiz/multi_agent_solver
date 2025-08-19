#pragma once
#include <functional>

#include <Eigen/Dense>

#include "multi_agent_solver/types.hpp"

namespace mas
{

// Single-step Euler integration
inline State
integrate_euler( const State& current_state, const Control& control, double dt, const MotionModel& motion_model )
{
  return current_state + dt * motion_model( current_state, control );
}

// Single-step RK4 integration
inline State
integrate_rk4( const State& current_state, const Control& control, double dt, const MotionModel& motion_model )
{
  State k1 = motion_model( current_state, control );
  State k2 = motion_model( current_state + 0.5 * dt * k1, control );
  State k3 = motion_model( current_state + 0.5 * dt * k2, control );
  State k4 = motion_model( current_state + dt * k3, control );

  return current_state + ( dt / 6.0 ) * ( k1 + 2 * k2 + 2 * k3 + k4 );
}

// Horizon integration function
inline StateTrajectory
integrate_horizon( const State& initial_state, const ControlTrajectory& controls, double dt, const MotionModel& motion_model,
                   const std::function<State( const State&, const Control&, double, const MotionModel& )>& single_step_integrator )
{
  // Initialize the state trajectory
  StateTrajectory state_trajectory( initial_state.size(), controls.cols() + 1 );
  state_trajectory.col( 0 ) = initial_state;

  // Integrate step by step
  State state = initial_state;
  for( int i = 0; i < controls.cols(); ++i )
  {
    state                         = single_step_integrator( state, controls.col( i ), dt, motion_model );
    state_trajectory.col( i + 1 ) = state;
  }

  return state_trajectory;
}
}