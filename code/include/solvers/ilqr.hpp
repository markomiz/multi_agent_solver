#pragma once

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "integrator.hpp"
#include "ocp.hpp"
#include "solver.hpp"
#include "types.hpp"

/**
 * @brief A standard iLQR solver that stores K[t], k[t] for each step t and performs
 *        a proper forward pass using these gains.
 *
 * @param problem        The OCP containing dynamics, cost function, etc.
 * @param max_iterations Maximum iLQR iterations
 * @param tolerance      Convergence threshold on cost improvement
 */
inline void
ilqr_solver( OCP& problem, const SolverParams& params )
{

  // Extract parameters
  const int    max_iterations = static_cast<int>( params.at( "max_iterations" ) );
  const double tolerance      = params.at( "tolerance" );

  // Extract problem dimensions
  const int    T   = problem.horizon_steps;
  const int    n_x = problem.state_dim;
  const int    n_u = problem.control_dim;
  const double dt  = problem.dt;

  // Allocate state/control trajectories
  StateTrajectory&   x    = problem.best_states;
  ControlTrajectory& u    = problem.best_controls;
  auto&              cost = problem.best_cost;

  // Forward simulate once to get initial trajectory & cost
  x    = integrate_horizon( problem.initial_state, u, dt, problem.dynamics, integrate_rk4 );
  cost = problem.objective_function( x, u );

  // Storage for the backward pass
  // K[t] is an n_u x n_x matrix, k[t] is n_u x 1
  std::vector<Eigen::MatrixXd> K( T, Eigen::MatrixXd::Zero( n_u, n_x ) );
  std::vector<Eigen::VectorXd> k( T, Eigen::VectorXd::Zero( n_u ) );
  for( int iter = 0; iter < max_iterations; ++iter )
  {

    Eigen::VectorXd V_x  = Eigen::VectorXd::Zero( n_x );
    Eigen::MatrixXd V_xx = Eigen::MatrixXd::Zero( n_x, n_x );

    for( int t = T - 1; t >= 0; --t )
    {
      // Compute Jacobians A = d f / d x, B = d f / d u
      const Eigen::MatrixXd A = problem.dynamics_state_jacobian( problem.dynamics, x.col( t ), u.col( t ) );
      const Eigen::MatrixXd B = problem.dynamics_control_jacobian( problem.dynamics, x.col( t ), u.col( t ) );

      // Cost derivatives wrt x, u
      const Eigen::VectorXd l_x = problem.cost_state_gradient( problem.objective_function, x.col( t ), u.col( t ) );

      const Eigen::VectorXd l_u = problem.cost_control_gradient( problem.objective_function, x.col( t ), u.col( t ) );

      const Eigen::MatrixXd l_xx = problem.cost_state_hessian( problem.objective_function, x.col( t ), u.col( t ) );

      const Eigen::MatrixXd l_uu = problem.cost_control_hessian( problem.objective_function, x.col( t ), u.col( t ) );

      const Eigen::MatrixXd l_ux = problem.cost_cross_term( problem.objective_function, x.col( t ), u.col( t ) );


      Eigen::VectorXd Q_x  = l_x + A.transpose() * V_x;
      Eigen::VectorXd Q_u  = l_u + B.transpose() * V_x;
      Eigen::MatrixXd Q_xx = l_xx + A.transpose() * V_xx * A;
      Eigen::MatrixXd Q_ux = l_ux + B.transpose() * V_xx * A;
      Eigen::MatrixXd Q_uu = l_uu + B.transpose() * V_xx * B;

      // Ensure Q_uu is positive definite (regularize if necessary)
      Eigen::MatrixXd             Q_uu_reg = Q_uu;
      Eigen::LLT<Eigen::MatrixXd> llt( Q_uu_reg );
      // minimal approach: keep adding small diag until success
      double reg_term = 1e-6;
      while( llt.info() != Eigen::Success )
      {
        Q_uu_reg += reg_term * Eigen::MatrixXd::Identity( n_u, n_u );
        llt.compute( Q_uu_reg );
        reg_term *= 10;
      }
      // get inverse from factor
      Eigen::MatrixXd Q_uu_inv = llt.solve( Eigen::MatrixXd::Identity( n_u, n_u ) );

      K[t] = -Q_uu_inv * Q_ux;
      k[t] = -Q_uu_inv * Q_u;

      V_x  = Q_x + K[t].transpose() * ( Q_uu * k[t] );
      V_xx = Q_xx + K[t].transpose() * Q_uu * K[t];
      // symmetrize
      V_xx = 0.5 * ( V_xx + V_xx.transpose() );
    } // end backward pass

    double            best_cost = cost;
    StateTrajectory   best_x    = x;
    ControlTrajectory best_u    = u;

    double       alpha     = 1.0;
    const double alpha_min = 1e-7;

    while( alpha >= alpha_min )
    {
      // We'll build a candidate trajectory x_new, u_new
      StateTrajectory   x_new = StateTrajectory::Zero( n_x, T + 1 );
      ControlTrajectory u_new = ControlTrajectory::Zero( n_u, T );

      // First state is always the same initial condition
      x_new.col( 0 ) = problem.initial_state;

      // Forward simulate
      for( int t = 0; t < T; ++t )
      {
        // The standard iLQR update:
        //   u_new[t] = u_old[t] + alpha*k[t] + K[t]*[x_new[t] - x_old[t]]
        Eigen::VectorXd dx = x_new.col( t ) - x.col( t );
        u_new.col( t )     = u.col( t ) + alpha * k[t] + K[t] * dx;

        // Optional clamp
        if( problem.input_lower_bounds && problem.input_upper_bounds )
        {
          clamp_controls( u_new, *problem.input_lower_bounds, *problem.input_upper_bounds );
        }

        // Now propagate forward
        x_new.col( t + 1 ) = integrate_rk4( x_new.col( t ), u_new.col( t ), dt, problem.dynamics );
      }

      double new_cost = problem.objective_function( x_new, u_new );

      // If cost improved, accept this alpha and break
      if( new_cost < best_cost )
      {
        best_cost = new_cost;
        best_x    = x_new;
        best_u    = u_new;
        break;
      }
      // Otherwise reduce alpha
      alpha *= 0.5;
    } // end while alpha

    // Check for improvement
    double cost_improv = cost - best_cost;
    if( cost_improv < tolerance )
    {
      // Not enough improvement => we consider it converged
      x    = best_x;
      u    = best_u;
      cost = best_cost;
      break;
    }
    else
    {
      // Accept the best
      x    = best_x;
      u    = best_u;
      cost = best_cost;
    }
  } // end main iLQR iteration
}
