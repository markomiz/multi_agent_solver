#pragma once
#include <chrono>
#include <iostream>

#include <Eigen/Dense>

#include "multi_agent_solver/constraint_helpers.hpp"
#include "multi_agent_solver/finite_differences.hpp"
#include "multi_agent_solver/integrator.hpp"
#include "multi_agent_solver/line_search.hpp"
#include "multi_agent_solver/ocp.hpp"
#include "multi_agent_solver/solvers/solver.hpp"
#include "multi_agent_solver/types.hpp"

namespace mas
{

/**
 * @brief Constrained-gradient-descent solver with reusable memory.
 *
 * Create once, call #solve(OCP&) as many times as you like.
 * All scratch buffers (multipliers, penalty parameter) live on the object,
 * so nothing is reallocated between calls.
 */
class CGD
{
public:

  /// Construct from the usual parameter map.
  explicit CGD() {}

  void
  set_params( const SolverParams& params )
  {
    max_iterations = static_cast<int>( params.at( "max_iterations" ) );
    tolerance      = params.at( "tolerance" );
    max_ms         = params.at( "max_ms" );
    penalty_param  = 1.0;
    debug          = params.count( "debug" ) && params.at( "debug" ) > 0.5;
    momentum       = params.count( "momentum" ) ? params.at( "momentum" ) : 0.0;
  }

  /**
   * @brief Solve one optimal-control problem.
   *
   * The method modifies @p problem.best_controls, @p problem.best_states and
   * @p problem.best_cost
   */
  void
  solve( OCP& problem )
  {
    using clock           = std::chrono::high_resolution_clock;
    const auto start_time = clock::now();

    resize_multipliers( problem );

    auto& controls         = problem.best_controls;
    auto& state_trajectory = problem.best_states;
    auto& cost             = problem.best_cost;

    if( velocity.rows() != controls.rows() || velocity.cols() != controls.cols() )
    {
      velocity.setZero( controls.rows(), controls.cols() );
    }
    else
    {
      velocity.setZero();
    }

    state_trajectory = integrate_horizon( problem.initial_state, controls, problem.dt, problem.dynamics, integrate_rk4 );

    cost = compute_augmented_cost( problem, eq_multipliers, ineq_multipliers, penalty_param, state_trajectory, controls );

    for( int iter = 0; iter < max_iterations; ++iter )
    {
      const double elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>( clock::now() - start_time ).count();

      if( elapsed_ms > max_ms && debug )
      {
        std::cout << "CGD solver terminated early: " << elapsed_ms << " ms > " << max_ms << " ms\n";
        break;
      }

      const ControlTrajectory lookahead_controls = controls - momentum * velocity;

      const ControlGradient gradients = finite_differences_gradient( problem.initial_state, lookahead_controls, problem.dynamics,
                                                                     problem.objective_function, problem.dt );

      const double step_size = armijo_line_search( problem.initial_state, lookahead_controls, gradients, problem.dynamics, problem.objective_function,
                                                   problem.dt, {} );

      ControlTrajectory trial_controls = lookahead_controls - step_size * gradients;

      ControlTrajectory new_velocity = momentum * velocity + step_size * gradients;
      if( problem.input_lower_bounds && problem.input_upper_bounds )
      {
        clamp_controls( trial_controls, problem.input_lower_bounds.value(), problem.input_upper_bounds.value() );
      }

      const StateTrajectory trial_trajectory = integrate_horizon( problem.initial_state, trial_controls, problem.dt, problem.dynamics,
                                                                  integrate_rk4 );

      const double trial_cost = compute_augmented_cost( problem, eq_multipliers, ineq_multipliers, penalty_param, trial_trajectory,
                                                        trial_controls );

      const double old_cost = cost;
      if( trial_cost < cost )
      {
        controls         = std::move( trial_controls );
        state_trajectory = std::move( trial_trajectory );
        cost             = trial_cost;
        velocity         = std::move( new_velocity );
      }

      update_lagrange_multipliers( problem, state_trajectory, controls, eq_multipliers, ineq_multipliers, penalty_param );

      increase_penalty_parameter( penalty_param, problem, state_trajectory, controls, tolerance );

      if( std::abs( old_cost - trial_cost ) < tolerance )
      {
        std::cout << "CGD solver converged in " << iter << "steps" << std::endl;
        break;
      }
    }
  }

private:

  void
  resize_multipliers( const OCP& problem )
  {
    if( problem.equality_constraints )
    {
      const auto m = problem.equality_constraints( problem.initial_state, {} ).size();
      eq_multipliers.setZero( m );
    }
    else
    {
      eq_multipliers.resize( 0 );
    }

    if( problem.inequality_constraints )
    {
      const auto p = problem.inequality_constraints( problem.initial_state, {} ).size();
      ineq_multipliers.setZero( p );
    }
    else
    {
      ineq_multipliers.resize( 0 );
    }
  }

  int    max_iterations;
  double tolerance;
  double max_ms;
  bool   debug = false;

  ConstraintViolations eq_multipliers;
  ConstraintViolations ineq_multipliers;
  double               penalty_param;
  ControlTrajectory    velocity;
  double               momentum = 0.0;
};

} // namespace mas
