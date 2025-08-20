// Re-usable-memory iLQR solver
#pragma once

#include <chrono>
#include <iostream>
#include <vector>

#include <Eigen/Dense>

#include "multi_agent_solver/integrator.hpp"
#include "multi_agent_solver/ocp.hpp"
#include "multi_agent_solver/solvers/solver.hpp"
#include "multi_agent_solver/types.hpp"

namespace mas
{

/**
 * @brief iLQR solver.
 *
 *  - Class name:  (iLQR)
 *
 * Call `solve(ocp)` as many times as you like—the buffers are resized only
 * when problem dimensions (T, n_x, n_u) change.
 */
class iLQR
{
public:

  explicit iLQR() {}

  void
  set_params( const SolverParams& params )
  {
    max_iterations = static_cast<int>( params.at( "max_iterations" ) );
    tolerance      = params.at( "tolerance" );
    max_ms         = params.at( "max_ms" );
    debug          = params.count( "debug" ) && params.at( "debug" ) > 0.5;
  }

  //------------------------------- API ----------------------------------//
  void
  solve( OCP& problem )
  {
    using clock      = std::chrono::high_resolution_clock;
    const auto start = clock::now();

    resize_buffers( problem ); // make sure caches fit this problem

    const int    T  = problem.horizon_steps;
    const int    nx = problem.state_dim;
    const int    nu = problem.control_dim;
    const double dt = problem.dt;

    // Aliases to OCP solution references
    StateTrajectory&   x    = problem.best_states;
    ControlTrajectory& u    = problem.best_controls;
    double&            cost = problem.best_cost;
    if( debug )
      std::cerr << "1 initial cost: " << cost << '\n';

    // ---------- initial rollout & cost ----------------------------------
    x    = integrate_horizon( problem.initial_state, u, dt, problem.dynamics, integrate_rk4 );
    cost = problem.objective_function( x, u );
    if( debug )
      std::cerr << "2 initial cost: " << cost << '\n';

    // ------------------------ main loop ---------------------------------
    for( int iter = 0; iter < max_iterations; ++iter )
    {
      const double elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>( clock::now() - start ).count();
      if( elapsed_ms > max_ms )
      {
        if( debug )
          std::cerr << "iLQR time limit hit after " << elapsed_ms << " ms / " << max_ms << " ms\n";
        break;
      }

      //---------------------- backward pass -----------------------------//
      v_x.setZero();
      v_xx.setZero();

      for( int t = T - 1; t >= 0; --t )
      {
        // Pre-compute per-step dynamics & cost Jacobians/Hessians ----------
        a_step[t] = problem.dynamics_state_jacobian( problem.dynamics, x.col( t ), u.col( t ) );
        b_step[t] = problem.dynamics_control_jacobian( problem.dynamics, x.col( t ), u.col( t ) );

        l_x_step[t]  = problem.cost_state_gradient( problem.stage_cost, x.col( t ), u.col( t ), t );
        l_u_step[t]  = problem.cost_control_gradient( problem.stage_cost, x.col( t ), u.col( t ), t );
        l_xx_step[t] = problem.cost_state_hessian( problem.stage_cost, x.col( t ), u.col( t ), t );
        l_uu_step[t] = problem.cost_control_hessian( problem.stage_cost, x.col( t ), u.col( t ), t );
        l_ux_step[t] = problem.cost_cross_term( problem.stage_cost, x.col( t ), u.col( t ), t );

        // Construct Q terms using cached buffers -------------------------
        q_x_step[t]  = l_x_step[t] + a_step[t].transpose() * v_x;
        q_u_step[t]  = l_u_step[t] + b_step[t].transpose() * v_x;
        q_xx_step[t] = l_xx_step[t] + a_step[t].transpose() * v_xx * a_step[t];
        q_ux_step[t] = l_ux_step[t] + b_step[t].transpose() * v_xx * a_step[t];
        q_uu_step[t] = l_uu_step[t] + b_step[t].transpose() * v_xx * b_step[t];

        // Regularised Cholesky of Q_uu -----------------------------------
        q_uu_reg_step[t] = q_uu_step[t];
        auto&  llt       = llt_step[t];
        double reg       = 1e-6;
        while( true )
        {
          llt.compute( q_uu_reg_step[t] );
          if( llt.info() == Eigen::Success )
            break;
          q_uu_reg_step[t] += reg * identity_nu;
          reg              *= 10.0;
        }
        q_uu_inv_step[t] = llt.solve( identity_nu );

        // Gains -----------------------------------------------------------
        k[t]        = -q_uu_inv_step[t] * q_u_step[t];
        k_matrix[t] = -q_uu_inv_step[t] * q_ux_step[t];

        // Value function update ------------------------------------------
        // V_x = Q_x + K^T Q_u + Q_{ux}^T k + K^T Q_{uu} k
        v_x = q_x_step[t] + k_matrix[t].transpose() * q_u_step[t] + q_ux_step[t].transpose() * k[t]
            + k_matrix[t].transpose() * q_uu_step[t] * k[t];

        // V_xx = Q_xx + K^T Q_{ux} + Q_{ux}^T K + K^T Q_{uu} K
        v_xx = q_xx_step[t] + k_matrix[t].transpose() * q_ux_step[t] + q_ux_step[t].transpose() * k_matrix[t]
             + k_matrix[t].transpose() * q_uu_step[t] * k_matrix[t];

        // Ensure symmetry of V_xx numerically
        v_xx = 0.5 * ( v_xx + v_xx.transpose() );
      }

      //---------------------- forward pass ------------------------------//
      x_trial.setZero();
      u_trial.setZero();
      x_trial.col( 0 ) = problem.initial_state;

      double            alpha     = 1.0;
      const double      amin      = 1e-3;
      double            best_cost = cost;
      StateTrajectory   best_x    = x;
      ControlTrajectory best_u    = u;

      while( alpha >= amin )
      {
        for( int t = 0; t < T; ++t )
        {
          const Eigen::VectorXd dx = x_trial.col( t ) - x.col( t );
          u_trial.col( t )         = u.col( t ) + alpha * k[t] + k_matrix[t] * dx;

          if( problem.input_lower_bounds && problem.input_upper_bounds )
            clamp_controls( u_trial, *problem.input_lower_bounds, *problem.input_upper_bounds );

          x_trial.col( t + 1 ) = integrate_rk4( x_trial.col( t ), u_trial.col( t ), dt, problem.dynamics );
        }

        const double new_cost = problem.objective_function( x_trial, u_trial );
        if( new_cost < best_cost )
        {
          best_cost = new_cost;
          best_x    = x_trial;
          best_u    = u_trial;
          break;
        }
        alpha *= 0.5;
      }

      const double improvement = cost - best_cost;
      if( debug )
        std::cerr << "iLQR iter " << iter << ": cost=" << best_cost << ", Δ=" << improvement << '\n';

      x    = best_x;
      u    = best_u;
      cost = best_cost;

      if( improvement < tolerance )
      {
        if( debug )
        {
          const double total_ms = std::chrono::duration_cast<std::chrono::milliseconds>( clock::now() - start ).count();
          std::cerr << "iLQR converged in " << total_ms << " ms\n";
        }
        break;
      }
    }
  }

private:

  //---------------- buffer management ----------------------------------//
  void
  resize_buffers( const OCP& problem )
  {
    const int T  = problem.horizon_steps;
    const int nx = problem.state_dim;
    const int nu = problem.control_dim;

    // Resize per-time-step vectors/matrices ------------------------------
    auto resize_mat_vec = [&]( auto& container, auto&& prototype ) {
      if( static_cast<int>( container.size() ) != T )
        container.assign( T, prototype );
      else
        for( auto& m : container )
          m.setZero();
    };

    resize_mat_vec( k, Eigen::VectorXd::Zero( nu ) );
    resize_mat_vec( k_matrix, Eigen::MatrixXd::Zero( nu, nx ) );

    resize_mat_vec( a_step, Eigen::MatrixXd::Zero( nx, nx ) );
    resize_mat_vec( b_step, Eigen::MatrixXd::Zero( nx, nu ) );
    resize_mat_vec( l_x_step, Eigen::VectorXd::Zero( nx ) );
    resize_mat_vec( l_u_step, Eigen::VectorXd::Zero( nu ) );
    resize_mat_vec( l_xx_step, Eigen::MatrixXd::Zero( nx, nx ) );
    resize_mat_vec( l_uu_step, Eigen::MatrixXd::Zero( nu, nu ) );
    resize_mat_vec( l_ux_step, Eigen::MatrixXd::Zero( nu, nx ) );

    resize_mat_vec( q_x_step, Eigen::VectorXd::Zero( nx ) );
    resize_mat_vec( q_u_step, Eigen::VectorXd::Zero( nu ) );
    resize_mat_vec( q_xx_step, Eigen::MatrixXd::Zero( nx, nx ) );
    resize_mat_vec( q_ux_step, Eigen::MatrixXd::Zero( nu, nx ) );
    resize_mat_vec( q_uu_step, Eigen::MatrixXd::Zero( nu, nu ) );
    resize_mat_vec( q_uu_reg_step, Eigen::MatrixXd::Zero( nu, nu ) );
    resize_mat_vec( q_uu_inv_step, Eigen::MatrixXd::Zero( nu, nu ) );

    // LLT factor objects -------------------------------------------------
    if( static_cast<int>( llt_step.size() ) != T )
      llt_step.assign( T, Eigen::LLT<Eigen::MatrixXd>( nu ) );

    // Trial trajectories --------------------------------------------------
    x_trial.resize( nx, T + 1 );
    u_trial.resize( nu, T );

    // Value function scratch & identity matrix ---------------------------
    v_x.resize( nx );
    v_xx.resize( nx, nx );
    identity_nu = Eigen::MatrixXd::Identity( nu, nu );
  }

  //---------------- data members ---------------------------------------//
  // — configuration
  int    max_iterations;
  double tolerance;
  double max_ms;
  bool   debug;

  // — time-step dependent caches (size T)
  std::vector<Eigen::VectorXd> k;        // nu
  std::vector<Eigen::MatrixXd> k_matrix; // nu x nx

  std::vector<Eigen::MatrixXd> a_step; // nx x nx
  std::vector<Eigen::MatrixXd> b_step; // nx x nu

  std::vector<Eigen::VectorXd> l_x_step;  // nx
  std::vector<Eigen::VectorXd> l_u_step;  // nu
  std::vector<Eigen::MatrixXd> l_xx_step; // nx x nx
  std::vector<Eigen::MatrixXd> l_uu_step; // nu x nu
  std::vector<Eigen::MatrixXd> l_ux_step; // nu x nx

  std::vector<Eigen::VectorXd> q_x_step;      // nx
  std::vector<Eigen::VectorXd> q_u_step;      // nu
  std::vector<Eigen::MatrixXd> q_xx_step;     // nx x nx
  std::vector<Eigen::MatrixXd> q_ux_step;     // nu x nx
  std::vector<Eigen::MatrixXd> q_uu_step;     // nu x nu
  std::vector<Eigen::MatrixXd> q_uu_reg_step; // nu x nu
  std::vector<Eigen::MatrixXd> q_uu_inv_step; // nu x nu

  std::vector<Eigen::LLT<Eigen::MatrixXd>> llt_step; // per-step factors

  // — global scratch
  StateTrajectory   x_trial; // nx x (T+1)
  ControlTrajectory u_trial; // nu x T

  Eigen::VectorXd v_x;         // nx
  Eigen::MatrixXd v_xx;        // nx x nx
  Eigen::MatrixXd identity_nu; // nu x nu
};

} // namespace mas
