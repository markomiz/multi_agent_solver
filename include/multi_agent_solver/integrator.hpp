#pragma once
#include <functional>

#include <Eigen/Dense>

#include "multi_agent_solver/types.hpp"

namespace mas
{

template<typename Scalar>
using SingleStepIntegratorT
  = std::function<StateT<Scalar>( const StateT<Scalar>&, const ControlT<Scalar>&, Scalar, const MotionModelT<Scalar>& )>;
using SingleStepIntegrator = SingleStepIntegratorT<double>;

// Single-step Euler integration
template<typename Scalar>
inline StateT<Scalar>
integrate_euler( const StateT<Scalar>& current_state, const ControlT<Scalar>& control, Scalar dt, const MotionModelT<Scalar>& motion_model )
{
  return current_state + dt * motion_model( current_state, control );
}

inline State
integrate_euler( const State& current_state, const Control& control, double dt, const MotionModel& motion_model )
{
  return integrate_euler<double>( current_state, control, dt, motion_model );
}

// Single-step RK4 integration
template<typename Scalar>
inline StateT<Scalar>
integrate_rk4( const StateT<Scalar>& current_state, const ControlT<Scalar>& control, Scalar dt, const MotionModelT<Scalar>& motion_model )
{
  const Scalar half_dt = static_cast<Scalar>( 0.5 ) * dt;
  const Scalar sixth   = dt / static_cast<Scalar>( 6.0 );

  StateT<Scalar> k1 = motion_model( current_state, control );
  StateT<Scalar> k2 = motion_model( current_state + half_dt * k1, control );
  StateT<Scalar> k3 = motion_model( current_state + half_dt * k2, control );
  StateT<Scalar> k4 = motion_model( current_state + dt * k3, control );

  return current_state + sixth * ( k1 + static_cast<Scalar>( 2 ) * k2 + static_cast<Scalar>( 2 ) * k3 + k4 );
}

inline State
integrate_rk4( const State& current_state, const Control& control, double dt, const MotionModel& motion_model )
{
  return integrate_rk4<double>( current_state, control, dt, motion_model );
}

// Horizon integration function
template<typename Scalar, typename SingleStepIntegrator>
inline StateTrajectoryT<Scalar>
integrate_horizon( const StateT<Scalar>& initial_state, const ControlTrajectoryT<Scalar>& controls, Scalar dt,
                   const MotionModelT<Scalar>& motion_model, SingleStepIntegrator&& single_step_integrator )
{
  StateTrajectoryT<Scalar> state_trajectory( initial_state.size(), controls.cols() + 1 );
  state_trajectory.col( 0 ) = initial_state;

  StateT<Scalar> state = initial_state;
  for( int i = 0; i < controls.cols(); ++i )
  {
    state                         = single_step_integrator( state, controls.col( i ), dt, motion_model );
    state_trajectory.col( i + 1 ) = state;
  }

  return state_trajectory;
}

template<typename Scalar>
inline StateTrajectoryT<Scalar>
integrate_horizon( const StateT<Scalar>& initial_state, const ControlTrajectoryT<Scalar>& controls, Scalar dt,
                   const MotionModelT<Scalar>& motion_model, const SingleStepIntegratorT<Scalar>& single_step_integrator )
{
  return integrate_horizon<Scalar, const SingleStepIntegratorT<Scalar>&>( initial_state, controls, dt, motion_model,
                                                                          single_step_integrator );
}

inline StateTrajectory
integrate_horizon( const State& initial_state, const ControlTrajectory& controls, double dt, const MotionModel& motion_model,
                   const SingleStepIntegrator& single_step_integrator )
{
  return integrate_horizon<double>( initial_state, controls, dt, motion_model, single_step_integrator );
}
} // namespace mas
