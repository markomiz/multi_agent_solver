#pragma once

#include <chrono>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "multi_agent_solver/integrator.hpp"
#include "multi_agent_solver/ocp.hpp"
#include "multi_agent_solver/solvers/solver.hpp"
#include "multi_agent_solver/types.hpp"

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
  const bool debug = params.count( "debug" ) && params.at( "debug" ) > 0.5;
  using clock      = std::chrono::high_resolution_clock;
  auto start_time  = clock::now();

  const int    max_iterations = static_cast<int>( params.at( "max_iterations" ) );
  const double tolerance      = params.at( "tolerance" );
  const double max_ms         = params.at( "max_ms" );

  const int    T   = problem.horizon_steps;
  const int    n_x = problem.state_dim;
  const int    n_u = problem.control_dim;
  const double dt  = problem.dt;

  StateTrajectory&   x    = problem.best_states;
  ControlTrajectory& u    = problem.best_controls;
  auto&              cost = problem.best_cost;

  x    = integrate_horizon( problem.initial_state, u, dt, problem.dynamics, integrate_rk4 );
  cost = problem.objective_function( x, u );

  if( debug )
    std::cerr << "Initial cost: " << cost << std::endl;

  std::vector<Eigen::MatrixXd> K( T, Eigen::MatrixXd::Zero( n_u, n_x ) );
  std::vector<Eigen::VectorXd> k( T, Eigen::VectorXd::Zero( n_u ) );

  for( int iter = 0; iter < max_iterations; ++iter )
  {
    auto   now        = clock::now();
    double elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>( now - start_time ).count();
    if( elapsed_ms > max_ms )
    {
      if( debug )
        std::cerr << "iLQR exited due to time constraint at iteration " << iter << ": " << elapsed_ms << " ms " << max_ms << " ms"
                  << std::endl;
      break;
    }

    Eigen::VectorXd V_x  = Eigen::VectorXd::Zero( n_x );
    Eigen::MatrixXd V_xx = Eigen::MatrixXd::Zero( n_x, n_x );

    for( int t = T - 1; t >= 0; --t )
    {
      const Eigen::MatrixXd A = problem.dynamics_state_jacobian( problem.dynamics, x.col( t ), u.col( t ) );
      const Eigen::MatrixXd B = problem.dynamics_control_jacobian( problem.dynamics, x.col( t ), u.col( t ) );

      const Eigen::VectorXd l_x  = problem.cost_state_gradient( problem.stage_cost, x.col( t ), u.col( t ), t );
      const Eigen::VectorXd l_u  = problem.cost_control_gradient( problem.stage_cost, x.col( t ), u.col( t ), t );
      const Eigen::MatrixXd l_xx = problem.cost_state_hessian( problem.stage_cost, x.col( t ), u.col( t ), t );
      const Eigen::MatrixXd l_uu = problem.cost_control_hessian( problem.stage_cost, x.col( t ), u.col( t ), t );
      const Eigen::MatrixXd l_ux = problem.cost_cross_term( problem.stage_cost, x.col( t ), u.col( t ), t );

      Eigen::VectorXd Q_x  = l_x + A.transpose() * V_x;
      Eigen::VectorXd Q_u  = l_u + B.transpose() * V_x;
      Eigen::MatrixXd Q_xx = l_xx + A.transpose() * V_xx * A;
      Eigen::MatrixXd Q_ux = l_ux + B.transpose() * V_xx * A;
      Eigen::MatrixXd Q_uu = l_uu + B.transpose() * V_xx * B;

      Eigen::MatrixXd             Q_uu_reg = Q_uu;
      Eigen::LLT<Eigen::MatrixXd> llt( Q_uu_reg );
      double                      reg_term = 1e-6;
      while( llt.info() != Eigen::Success )
      {
        Q_uu_reg += reg_term * Eigen::MatrixXd::Identity( n_u, n_u );
        llt.compute( Q_uu_reg );
        reg_term *= 10;
      }

      Eigen::MatrixXd Q_uu_inv = llt.solve( Eigen::MatrixXd::Identity( n_u, n_u ) );
      K[t]                     = -Q_uu_inv * Q_ux;
      k[t]                     = -Q_uu_inv * Q_u;

      V_x  = Q_x - K[t].transpose() * ( Q_uu * k[t] + Q_u );
      V_xx = Q_xx - K[t].transpose() * ( Q_uu * K[t] + Q_ux );
      V_xx = 0.5 * ( V_xx + V_xx.transpose() );
    }

    double            best_cost = cost;
    StateTrajectory   best_x    = x;
    ControlTrajectory best_u    = u;

    double       alpha     = 1.0;
    const double alpha_min = 1e-7;

    while( alpha >= alpha_min )
    {
      StateTrajectory   x_new = StateTrajectory::Zero( n_x, T + 1 );
      ControlTrajectory u_new = ControlTrajectory::Zero( n_u, T );
      x_new.col( 0 )          = problem.initial_state;

      for( int t = 0; t < T; ++t )
      {
        Eigen::VectorXd dx = x_new.col( t ) - x.col( t );
        u_new.col( t )     = u.col( t ) + alpha * k[t] + K[t] * dx;

        if( problem.input_lower_bounds && problem.input_upper_bounds )
          clamp_controls( u_new, *problem.input_lower_bounds, *problem.input_upper_bounds );

        x_new.col( t + 1 ) = integrate_rk4( x_new.col( t ), u_new.col( t ), dt, problem.dynamics );
      }

      double new_cost = problem.objective_function( x_new, u_new );

      if( new_cost < best_cost )
      {
        best_cost = new_cost;
        best_x    = x_new;
        best_u    = u_new;
        break;
      }

      alpha *= 0.5;
    }

    double cost_improv = cost - best_cost;
    if( debug )
      std::cerr << "Iteration " << iter << ": cost = " << best_cost << ", improvement = " << cost_improv << std::endl;

    if( cost_improv < tolerance )
    {
      if( debug )
      {
        auto   now        = clock::now();
        double elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>( now - start_time ).count();
        std::cerr << "Converged at iteration " << iter << ", improvement " << cost_improv << " < tolerance " << tolerance << " in "
                  << elapsed_ms << " ms" << std::endl;
      }

      x    = best_x;
      u    = best_u;
      cost = best_cost;
      break;
    }
    else
    {
      x    = best_x;
      u    = best_u;
      cost = best_cost;
    }
  }
}
