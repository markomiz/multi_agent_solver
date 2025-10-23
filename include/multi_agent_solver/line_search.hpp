#pragma once

#include <functional>
#include <map>
#include <string>

#include <Eigen/Dense>

#include "multi_agent_solver/integrator.hpp"
#include "multi_agent_solver/types.hpp"

namespace mas
{

// Generic line search function alias
template<typename Scalar>
using LineSearchFunctionT = std::function<Scalar( const StateT<Scalar>& initial_state, const ControlTrajectoryT<Scalar>& controls,
                                                  const ControlGradientT<Scalar>& gradients, const MotionModelT<Scalar>& dynamics,
                                                  const ObjectiveFunctionT<Scalar>& objective_function, Scalar dt,
                                                  const std::map<std::string, Scalar>& parameters )>;
using LineSearchFunction  = LineSearchFunctionT<double>;

// Utility function to get parameter value with default
template<typename Scalar>
inline Scalar
get_parameter( const std::map<std::string, Scalar>& parameters, const std::string& key, Scalar default_value )
{
  auto it = parameters.find( key );
  return ( it != parameters.end() ) ? it->second : default_value;
}

inline double
get_parameter( const std::map<std::string, double>& parameters, const std::string& key, double default_value )
{
  return get_parameter<double>( parameters, key, default_value );
}

// Armijo line search
template<typename Scalar>
inline Scalar
armijo_line_search( const StateT<Scalar>& initial_state, const ControlTrajectoryT<Scalar>& controls,
                    const ControlGradientT<Scalar>& gradients, const MotionModelT<Scalar>& dynamics,
                    const ObjectiveFunctionT<Scalar>& objective_function, Scalar dt, const std::map<std::string, Scalar>& parameters )
{
  Scalar initial_step_size = get_parameter<Scalar>( parameters, "initial_step_size", static_cast<Scalar>( 1.0 ) );
  Scalar beta              = get_parameter<Scalar>( parameters, "beta", static_cast<Scalar>( 0.5 ) );
  Scalar c1                = get_parameter<Scalar>( parameters, "c1", static_cast<Scalar>( 1e-6 ) );

  Scalar alpha    = initial_step_size;
  Scalar cost_ref = objective_function( integrate_horizon<Scalar>( initial_state, controls, dt, dynamics, integrate_rk4<Scalar> ),
                                        controls );
  Scalar directional_derivative = gradients.cwiseProduct( -gradients ).sum();

  while( true )
  {
    // Compute trial controls and cost
    ControlTrajectoryT<Scalar> trial_controls   = controls - alpha * gradients;
    StateTrajectoryT<Scalar>   trial_trajectory = integrate_horizon<Scalar>( initial_state, trial_controls, dt, dynamics,
                                                                             integrate_rk4<Scalar> );
    Scalar                     trial_cost       = objective_function( trial_trajectory, trial_controls );

    // Check Armijo condition. directional_derivative is negative, so the
    // right-hand side is less than cost_ref when a descent direction is
    // used.
    if( trial_cost <= cost_ref + c1 * alpha * directional_derivative )
    {
      break;
    }

    // Reduce step size
    alpha *= beta;

    // Avoid excessively small step sizes
    if( alpha < static_cast<Scalar>( 1e-8 ) )
    {
      break;
    }
  }

  return alpha;
}

inline double
armijo_line_search( const State& initial_state, const ControlTrajectory& controls, const ControlGradient& gradients,
                    const MotionModel& dynamics, const ObjectiveFunction& objective_function, double dt,
                    const std::map<std::string, double>& parameters )
{
  return armijo_line_search<double>( initial_state, controls, gradients, dynamics, objective_function, dt, parameters );
}

// Backtracking line search
template<typename Scalar>
inline Scalar
backtracking_line_search( const StateT<Scalar>& initial_state, const ControlTrajectoryT<Scalar>& controls,
                          const ControlGradientT<Scalar>& gradients, const MotionModelT<Scalar>& dynamics,
                          const ObjectiveFunctionT<Scalar>& objective_function, Scalar dt, const std::map<std::string, Scalar>& parameters )
{
  Scalar initial_step_size = get_parameter<Scalar>( parameters, "initial_step_size", static_cast<Scalar>( 1.0 ) );
  Scalar beta              = get_parameter<Scalar>( parameters, "beta", static_cast<Scalar>( 0.5 ) );

  Scalar alpha    = initial_step_size;
  Scalar cost_ref = objective_function( integrate_horizon<Scalar>( initial_state, controls, dt, dynamics, integrate_rk4<Scalar> ),
                                        controls );

  while( true )
  {
    // Compute trial controls and cost
    ControlTrajectoryT<Scalar> trial_controls   = controls - alpha * gradients;
    StateTrajectoryT<Scalar>   trial_trajectory = integrate_horizon<Scalar>( initial_state, trial_controls, dt, dynamics,
                                                                             integrate_rk4<Scalar> );
    Scalar                     trial_cost       = objective_function( trial_trajectory, trial_controls );

    // Check if cost decreased
    if( trial_cost < cost_ref )
    {
      break;
    }

    // Reduce step size
    alpha *= beta;

    // Avoid excessively small step sizes
    if( alpha < static_cast<Scalar>( 1e-8 ) )
    {
      break;
    }
  }

  return alpha;
}

inline double
backtracking_line_search( const State& initial_state, const ControlTrajectory& controls, const ControlGradient& gradients,
                          const MotionModel& dynamics, const ObjectiveFunction& objective_function, double dt,
                          const std::map<std::string, double>& parameters )
{
  return backtracking_line_search<double>( initial_state, controls, gradients, dynamics, objective_function, dt, parameters );
}

// Constant step size line search for simplicity
template<typename Scalar>
inline Scalar
constant_line_search( const StateT<Scalar>& /*initial_state*/, const ControlTrajectoryT<Scalar>& /*controls*/,
                      const ControlGradientT<Scalar>& /*gradients*/, const MotionModelT<Scalar>& /*dynamics*/,
                      const ObjectiveFunctionT<Scalar>& /*objective_function*/, Scalar /*dt*/,
                      const std::map<std::string, Scalar>& parameters )
{
  return get_parameter<Scalar>( parameters, "step_size", static_cast<Scalar>( 0.1 ) );
}

inline double
constant_line_search( const State& initial_state, const ControlTrajectory& controls, const ControlGradient& gradients,
                      const MotionModel& dynamics, const ObjectiveFunction& objective_function, double dt,
                      const std::map<std::string, double>& parameters )
{
  return constant_line_search<double>( initial_state, controls, gradients, dynamics, objective_function, dt, parameters );
}

} // namespace mas
