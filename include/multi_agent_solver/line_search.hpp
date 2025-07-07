#pragma once

#include <functional>
#include <map>
#include <string>

#include <Eigen/Dense>

#include "multi_agent_solver/integrator.hpp"
#include "multi_agent_solver/ocp.hpp"
#include "multi_agent_solver/types.hpp"

// Generic line search function alias
using LineSearchFunction = std::function<
  double( const State& initial_state, const ControlTrajectory& controls, const ControlGradient& gradients, const MotionModel& dynamics,
          const ObjectiveFunction& objective_function, double dt, const std::map<std::string, double>& parameters )>;

// Utility function to get parameter value with default
inline double
get_parameter( const std::map<std::string, double>& parameters, const std::string& key, double default_value )
{
  auto it = parameters.find( key );
  return ( it != parameters.end() ) ? it->second : default_value;
}

// Armijo line search
inline double
armijo_line_search( const State& initial_state, const ControlTrajectory& controls, const ControlGradient& gradients,
                    const MotionModel& dynamics, const ObjectiveFunction& objective_function, double dt,
                    const std::map<std::string, double>& parameters )
{

  double initial_step_size = get_parameter( parameters, "initial_step_size", 1.0 );
  double beta              = get_parameter( parameters, "beta", 0.5 );
  double c1                = get_parameter( parameters, "c1", 1e-6 );

  double alpha    = initial_step_size;
  double cost_ref = objective_function( integrate_horizon( initial_state, controls, dt, dynamics, integrate_rk4 ), controls );

  // The search direction is the negative gradient, so grad^T * (-grad)
  // should be negative. The Armijo condition expects f(x + alpha p)
  // <= f(x) + c1 * alpha * grad^T p. With p = -gradients this becomes
  // cost_ref + c1 * alpha * directional_derivative where
  // directional_derivative = grad^T * (-grad).
  double directional_derivative = gradients.cwiseProduct( -gradients ).sum();

  while( true )
  {
    // Compute trial controls and cost
    ControlTrajectory trial_controls   = controls - alpha * gradients;
    StateTrajectory   trial_trajectory = integrate_horizon( initial_state, trial_controls, dt, dynamics, integrate_rk4 );
    double            trial_cost       = objective_function( trial_trajectory, trial_controls );

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
    if( alpha < 1e-8 )
    {
      break;
    }
  }

  return alpha;
}

// Backtracking line search
inline double
backtracking_line_search( const State& initial_state, const ControlTrajectory& controls, const ControlGradient& gradients,
                          const MotionModel& dynamics, const ObjectiveFunction& objective_function, double dt,
                          const std::map<std::string, double>& parameters )
{

  double initial_step_size = get_parameter( parameters, "initial_step_size", 1.0 );
  double beta              = get_parameter( parameters, "beta", 0.5 );

  double alpha    = initial_step_size;
  double cost_ref = objective_function( integrate_horizon( initial_state, controls, dt, dynamics, integrate_rk4 ), controls );

  while( true )
  {
    // Compute trial controls and cost
    ControlTrajectory trial_controls   = controls - alpha * gradients;
    StateTrajectory   trial_trajectory = integrate_horizon( initial_state, trial_controls, dt, dynamics, integrate_rk4 );
    double            trial_cost       = objective_function( trial_trajectory, trial_controls );

    // Check if cost decreased
    if( trial_cost < cost_ref )
    {
      break;
    }

    // Reduce step size
    alpha *= beta;

    // Avoid excessively small step sizes
    if( alpha < 1e-8 )
    {
      break;
    }
  }

  return alpha;
}

// Constant step size line search for simplicity
inline double
constant_line_search( const State& /*initial_state*/, const ControlTrajectory& /*controls*/, const ControlGradient& /*gradients*/,
                      const MotionModel& /*dynamics*/, const ObjectiveFunction& /*objective_function*/, double /*dt*/,
                      const std::map<std::string, double>& parameters )
{

  return get_parameter( parameters, "step_size", 0.1 );
}
