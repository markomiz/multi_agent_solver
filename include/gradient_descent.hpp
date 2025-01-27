#include <functional>
#include <iostream>

#include <Eigen/Dense>

#include "finite_differences.hpp"
#include "integrator.hpp"
#include "line_search.hpp"
#include "ocp.hpp"
#include "solver_ouput.hpp"
#include "types.hpp"

inline SolverOutput
gradient_descent_solver( const OCP& problem, const GradientComputer& gradient_computer, int max_iterations, double tolerance,
                         const LineSearchFunction& line_search_function = armijo_line_search )
{
  SolverOutput output;
  // Initialize control trajectory
  output.controls        = ControlTrajectory::Zero( problem.control_dim, problem.horizon_steps );
  auto& controls         = output.controls;
  auto& state_trajectory = output.trajectory;

  // Integrate the initial state trajectory and compute the initial cost
  state_trajectory = integrate_horizon( problem.initial_state, controls, problem.dt, problem.dynamics, integrate_rk4 );
  output.cost      = problem.objective_function( state_trajectory, controls );

  for( int iter = 0; iter < max_iterations; ++iter )
  {
    // Compute the gradients
    ControlGradient gradients = gradient_computer( problem.initial_state, controls, problem.dynamics, problem.objective_function,
                                                   problem.dt );

    // Perform line search to find optimal step size
    double step_size = line_search_function( problem, problem.initial_state, controls, gradients, problem.dynamics,
                                             problem.objective_function, problem.dt, {} );

    // Create trial solution with updated controls
    ControlTrajectory trial_controls = controls - step_size * gradients;
    // if( problem.input_lower_bounds && problem.input_upper_bounds )
    //   trial_controls = trial_controls.cwiseMin( problem.input_upper_bounds.value() ).cwiseMax( problem.input_lower_bounds.value() );
    StateTrajectory trial_trajectory = integrate_horizon( problem.initial_state, trial_controls, problem.dt, problem.dynamics,
                                                          integrate_rk4 );
    double          trial_cost       = problem.objective_function( trial_trajectory, trial_controls );

    // Update solution only if cost improves
    double old_cost = output.cost;
    if( trial_cost < output.cost )
    {
      controls         = trial_controls;
      state_trajectory = trial_trajectory;
      output.cost      = trial_cost;
    }
    // std::cout << "Iteration " << iter << ", Cost: " << output.cost << ", Gradient Norm: " << gradients.norm() << "  step size " <<
    // step_size
    //           << std::endl;


    // Check for convergence based on cost improvement
    if( std::abs( old_cost - trial_cost ) < tolerance )
    {
      break;
    }
  }

  output.trajectory = state_trajectory;
  return output;
}
