#pragma once
#include <functional>

#include <Eigen/Dense>

#include "multi_agent_solver/types.hpp"

namespace mas
{

// Single-step Euler integration
template <typename Scalar = double>
inline State<Scalar>
integrate_euler( const State<Scalar>& current_state, const Control<Scalar>& control, double dt,
                 const MotionModel<Scalar>& motion_model )
{
  return current_state + dt * motion_model( current_state, control );
}

// Single-step RK4 integration
template <typename Scalar = double>
inline State<Scalar>
integrate_rk4( const State<Scalar>& current_state, const Control<Scalar>& control, double dt,
               const MotionModel<Scalar>& motion_model )
{
  State<Scalar> k1 = motion_model( current_state, control );
  State<Scalar> k2 = motion_model( current_state + 0.5 * dt * k1, control );
  State<Scalar> k3 = motion_model( current_state + 0.5 * dt * k2, control );
  State<Scalar> k4 = motion_model( current_state + dt * k3, control );

  return current_state + ( dt / 6.0 ) * ( k1 + 2 * k2 + 2 * k3 + k4 );
}

// Horizon integration function
template <typename Scalar = double>
inline StateTrajectoryT<Scalar>
integrate_horizon( const State<Scalar>& initial_state, const ControlTrajectoryT<Scalar>& controls, double dt,
                   const MotionModel<Scalar>& motion_model,
                   const std::function<State<Scalar>( const State<Scalar>&, const Control<Scalar>&, double,
                                                     const MotionModel<Scalar>& )>& single_step_integrator )
{
  // Initialize the state trajectory
  StateTrajectoryT<Scalar> state_trajectory( initial_state.size(), controls.cols() + 1 );
  state_trajectory.col( 0 ) = initial_state;

  // Integrate step by step
  State<Scalar> state = initial_state;
  for( int i = 0; i < controls.cols(); ++i )
  {
    state                         = single_step_integrator( state, controls.col( i ), dt, motion_model );
    state_trajectory.col( i + 1 ) = state;
  }

  return state_trajectory;
}
}