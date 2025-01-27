#pragma once

#include <Eigen/Dense>

#include "integrator.hpp"
#include "line_search.hpp"
#include "ocp.hpp"
#include "solver_ouput.hpp"
#include "types.hpp"

// iLQR Solver
inline SolverOutput
ilqr_solver( const OCP& problem, int max_iterations = 100, double tolerance = 1e-6 )
{
  const int horizon_steps = problem.horizon_steps;
  const int state_dim     = problem.state_dim;
  const int control_dim   = problem.control_dim;

  // Initialize state and control trajectories
  StateTrajectory   states   = StateTrajectory::Zero( state_dim, horizon_steps + 1 );
  ControlTrajectory controls = ControlTrajectory::Zero( control_dim, horizon_steps );

  // Initialize trajectory from the initial state using integrate_horizon
  states = integrate_horizon( problem.initial_state, controls, problem.dt, problem.dynamics, integrate_rk4 );

  double cost = problem.objective_function( states, controls );

  for( int iter = 0; iter < max_iterations; ++iter )
  {
    // Backward pass: compute value function approximation
    Eigen::MatrixXd V_x  = Eigen::MatrixXd::Zero( state_dim, 1 );
    Eigen::MatrixXd V_xx = Eigen::MatrixXd::Zero( state_dim, state_dim );
    Eigen::MatrixXd Q_x, Q_u, Q_xx, Q_ux, Q_uu;
    Eigen::MatrixXd K, k;

    for( int t = horizon_steps - 1; t >= 0; --t )
    {
      // Compute derivatives of cost and dynamics
      Eigen::MatrixXd A, B; // Jacobians of dynamics w.r.t. state and control
      A = problem.dynamics_state_jacobian( problem.dynamics, states.col( t ), controls.col( t ) );
      B = problem.dynamics_control_jacobian( problem.dynamics, states.col( t ), controls.col( t ) );

      Eigen::VectorXd l_x, l_u;         // Gradients of cost function
      Eigen::MatrixXd l_xx, l_uu, l_ux; // Hessians of cost function
      l_x  = problem.cost_state_gradient( problem.objective_function, states.col( t ), controls.col( t ) );
      l_u  = problem.cost_control_gradient( problem.objective_function, states.col( t ), controls.col( t ) );
      l_xx = problem.cost_state_hessian( problem.objective_function, states.col( t ), controls.col( t ) );
      l_uu = problem.cost_control_hessian( problem.objective_function, states.col( t ), controls.col( t ) );
      l_ux = problem.cost_cross_term( problem.objective_function, states.col( t ), controls.col( t ) );

      // Compute Q terms
      Q_x  = l_x + A.transpose() * V_x;
      Q_u  = l_u + B.transpose() * V_x;
      Q_xx = l_xx + A.transpose() * V_xx * A;
      Q_ux = l_ux + B.transpose() * V_xx * A;
      Q_uu = l_uu + B.transpose() * V_xx * B;

      // Compute gains
      Eigen::MatrixXd Q_uu_inv = Q_uu.inverse();
      K                        = -Q_uu_inv * Q_ux;
      k                        = -Q_uu_inv * Q_u;

      // Update value function
      V_x  = Q_x + K.transpose() * Q_uu * k;
      V_xx = Q_xx + K.transpose() * Q_uu * K;
    }

    double step_size = armijo_line_search( problem, problem.initial_state, controls, k, problem.dynamics, problem.objective_function,
                                           problem.dt, {} );

    // Apply the step size
    StateTrajectory   new_states   = integrate_horizon( problem.initial_state, controls, problem.dt, problem.dynamics, integrate_rk4 );
    ControlTrajectory new_controls = ControlTrajectory::Zero( control_dim, horizon_steps );

    for( int t = 0; t < horizon_steps; ++t )
    {
      Control u             = controls.col( t ) + step_size * ( k + K * ( new_states.col( t ) - states.col( t ) ) );
      new_controls.col( t ) = u;
    }

    // Evaluate new cost
    double new_cost = problem.objective_function( new_states, new_controls );

    // Check for convergence
    if( std::abs( cost - new_cost ) < tolerance )
    {
      break;
    }

    // Update trajectories and cost
    states   = new_states;
    controls = new_controls;
    cost     = new_cost;
  }

  SolverOutput solution;
  solution.cost       = cost;
  solution.trajectory = states;
  solution.controls   = controls;
  return solution;
}