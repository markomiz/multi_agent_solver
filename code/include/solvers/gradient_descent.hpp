#pragma once
#include <functional>
#include <iostream>

#include <Eigen/Dense>

#include "finite_differences.hpp"
#include "integrator.hpp"
#include "line_search.hpp"
#include "ocp.hpp"
#include "types.hpp"

inline void
gradient_descent_solver( OCP& problem, int max_iterations, double tolerance )
{
  auto& controls         = problem.best_controls;
  auto& state_trajectory = problem.best_states;
  auto& cost             = problem.best_cost;

  // Integrate the initial state trajectory and compute the initial cost
  state_trajectory = integrate_horizon( problem.initial_state, controls, problem.dt, problem.dynamics, integrate_rk4 );
  cost             = problem.objective_function( state_trajectory, controls );

  for( int iter = 0; iter < max_iterations; ++iter )
  {
    // Compute the gradients
    ControlGradient gradients = finite_differences_gradient( problem.initial_state, controls, problem.dynamics, problem.objective_function,
                                                             problem.dt );

    // Perform line search to find optimal step size
    double step_size = armijo_line_search( problem.initial_state, controls, gradients, problem.dynamics, problem.objective_function,
                                           problem.dt, {} );

    // Create trial solution with updated controls
    ControlTrajectory trial_controls = controls - step_size * gradients;

    if( problem.input_lower_bounds && problem.input_upper_bounds )
    {
      clamp_controls( trial_controls, problem.input_lower_bounds.value(), problem.input_upper_bounds.value() );
    }

    StateTrajectory trial_trajectory = integrate_horizon( problem.initial_state, trial_controls, problem.dt, problem.dynamics,
                                                          integrate_rk4 );
    double          trial_cost       = problem.objective_function( trial_trajectory, trial_controls );

    // Update solution only if cost improves
    double old_cost = cost;
    if( trial_cost < cost )
    {
      controls         = trial_controls;
      state_trajectory = trial_trajectory;
      cost             = trial_cost;
    }


    // Check for convergence based on cost improvement
    if( std::abs( old_cost - trial_cost ) < tolerance )
    {
      break;
    }
  }
}
