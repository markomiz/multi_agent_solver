#pragma once
#include <chrono>
#include <functional>
#include <iostream>

#include <Eigen/Dense>

#include "multi_agent_solver/constraint_helpers.hpp"
#include "multi_agent_solver/finite_differences.hpp"
#include "multi_agent_solver/integrator.hpp"
#include "multi_agent_solver/line_search.hpp"
#include "multi_agent_solver/ocp.hpp"
#include "multi_agent_solver/solvers/solver.hpp"
#include "multi_agent_solver/types.hpp"

inline void
cgd_solver( OCP& problem, const SolverParams& params )
{

  // Extract parameters
  const int    max_iterations = static_cast<int>( params.at( "max_iterations" ) );
  const double tolerance      = params.at( "tolerance" );
  const double max_ms         = params.at( "max_ms" );
  using clock                 = std::chrono::high_resolution_clock;
  auto start_time             = clock::now();

  // Initialize Lagrange multipliers and penalty parameter
  ConstraintViolations equality_multipliers   = problem.equality_constraints
                                                ? ConstraintViolations::Zero(
                                                  problem.equality_constraints( problem.initial_state, {} ).size() )
                                                : ConstraintViolations();
  ConstraintViolations inequality_multipliers = problem.inequality_constraints
                                                ? ConstraintViolations::Zero(
                                                    problem.inequality_constraints( problem.initial_state, {} ).size() )
                                                : ConstraintViolations();
  double               penalty_parameter      = 1.0;

  // Initialize control trajectory
  auto& controls         = problem.best_controls;
  auto& state_trajectory = problem.best_states;
  auto& cost             = problem.best_cost;

  // Integrate the initial state trajectory and compute the initial cost
  state_trajectory = integrate_horizon( problem.initial_state, controls, problem.dt, problem.dynamics, integrate_rk4 );
  cost = compute_augmented_cost( problem, equality_multipliers, inequality_multipliers, penalty_parameter, state_trajectory, controls );

  for( int iter = 0; iter < max_iterations; ++iter )
  {
    auto   now        = clock::now();
    double elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>( now - start_time ).count();
    if( elapsed_ms > max_ms )
    {
      std::cout << "iLQR solver terminated early due to max_ms constraint (" << elapsed_ms << " ms > " << max_ms << " ms)\n";
      break;
    }
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

    // Integrate trajectory with trial controls
    StateTrajectory trial_trajectory = integrate_horizon( problem.initial_state, trial_controls, problem.dt, problem.dynamics,
                                                          integrate_rk4 );

    double trial_cost = compute_augmented_cost( problem, equality_multipliers, inequality_multipliers, penalty_parameter, trial_trajectory,
                                                trial_controls );

    // Update solution only if cost improves
    double old_cost = cost;
    if( trial_cost < cost )
    {
      controls         = trial_controls;
      state_trajectory = trial_trajectory;
      cost             = trial_cost;
    }

    // Update multipliers and penalty parameter across the entire horizon
    update_lagrange_multipliers( problem, state_trajectory, controls, equality_multipliers, inequality_multipliers, penalty_parameter );
    increase_penalty_parameter( penalty_parameter, problem, state_trajectory, controls, tolerance );
    // Check for convergence
    if( std::abs( old_cost - trial_cost ) < tolerance )
    {
      break;
    }
  }
}