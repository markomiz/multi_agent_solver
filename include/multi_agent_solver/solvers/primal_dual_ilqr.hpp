// Primal-Dual iLQR solver with augmented Lagrangian handling of constraints
#pragma once

#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

#include <Eigen/Dense>

#include "multi_agent_solver/constraint_helpers.hpp"
#include "multi_agent_solver/integrator.hpp"
#include "multi_agent_solver/ocp.hpp"
#include "multi_agent_solver/types.hpp"

namespace mas
{

class PrimalDualiLQR
{
public:
  PrimalDualiLQR()
    : max_iterations( 50 )
    , tolerance( 1e-6 )
    , max_ms( std::numeric_limits<double>::infinity() )
    , debug( false )
    , penalty_parameter( 10.0 )
    , penalty_increase( 5.0 )
    , constraint_tolerance( 1e-4 )
    , inequality_activation_tolerance( 1e-6 )
    , equality_dim( 0 )
    , inequality_dim( 0 )
  {}

  void
  set_params( const SolverParams& params )
  {
    auto get = [&]( const std::string& key, double default_value ) {
      const auto it = params.find( key );
      return it != params.end() ? it->second : default_value;
    };

    max_iterations                  = static_cast<int>( get( "max_iterations", static_cast<double>( max_iterations ) ) );
    tolerance                       = get( "tolerance", tolerance );
    max_ms                          = get( "max_ms", max_ms );
    penalty_parameter               = get( "penalty", penalty_parameter );
    penalty_increase                = get( "penalty_increase", penalty_increase );
    constraint_tolerance            = get( "constraint_tolerance", constraint_tolerance );
    inequality_activation_tolerance = get( "inequality_activation_tolerance", inequality_activation_tolerance );
    debug                           = get( "debug", debug ? 1.0 : 0.0 ) > 0.5;
  }

  void
  solve( OCP& problem )
  {
    using clock = std::chrono::high_resolution_clock;
    const auto start = clock::now();

    resize_buffers( problem );

    const int    T  = problem.horizon_steps;
    const int    nx = problem.state_dim;
    const int    nu = problem.control_dim;
    const double dt = problem.dt;

    StateTrajectory&   x    = problem.best_states;
    ControlTrajectory& u    = problem.best_controls;
    double&            cost = problem.best_cost;

    x    = integrate_horizon( problem.initial_state, u, dt, problem.dynamics, integrate_rk4 );
    cost = problem.objective_function( x, u );

    double current_merit = compute_merit( problem, x, u );
    if( debug )
      std::cout << "pd-iLQR initial cost=" << cost << " merit=" << current_merit << '\n';

    for( int iter = 0; iter < max_iterations; ++iter )
    {
      const double elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>( clock::now() - start ).count();
      if( elapsed_ms > max_ms )
      {
        if( debug )
          std::cout << "pd-iLQR time limit hit after " << elapsed_ms << " ms / " << max_ms << " ms\n";
        break;
      }

      v_x.setZero();
      v_xx.setZero();

      for( int t = T - 1; t >= 0; --t )
      {
        a_step[t] = problem.dynamics_state_jacobian( problem.dynamics, x.col( t ), u.col( t ) );
        b_step[t] = problem.dynamics_control_jacobian( problem.dynamics, x.col( t ), u.col( t ) );

        l_x_step[t]  = problem.cost_state_gradient( problem.stage_cost, x.col( t ), u.col( t ), t );
        l_u_step[t]  = problem.cost_control_gradient( problem.stage_cost, x.col( t ), u.col( t ), t );
        l_xx_step[t] = problem.cost_state_hessian( problem.stage_cost, x.col( t ), u.col( t ), t );
        l_uu_step[t] = problem.cost_control_hessian( problem.stage_cost, x.col( t ), u.col( t ), t );
        l_ux_step[t] = problem.cost_cross_term( problem.stage_cost, x.col( t ), u.col( t ), t );

        q_x_step[t]  = l_x_step[t] + a_step[t].transpose() * v_x;
        q_u_step[t]  = l_u_step[t] + b_step[t].transpose() * v_x;
        q_xx_step[t] = l_xx_step[t] + a_step[t].transpose() * v_xx * a_step[t];
        q_ux_step[t] = l_ux_step[t] + b_step[t].transpose() * v_xx * a_step[t];
        q_uu_step[t] = l_uu_step[t] + b_step[t].transpose() * v_xx * b_step[t];

        if( equality_dim > 0 && problem.equality_constraints )
        {
          eq_residuals[t] = problem.equality_constraints( x.col( t ), u.col( t ) );
          eq_jacobian_x[t]
            = problem.equality_constraints_state_jacobian ? problem.equality_constraints_state_jacobian( x.col( t ), u.col( t ) )
                                                           : compute_constraints_state_jacobian( problem.equality_constraints, x.col( t ), u.col( t ) );
          eq_jacobian_u[t]
            = problem.equality_constraints_control_jacobian ? problem.equality_constraints_control_jacobian( x.col( t ), u.col( t ) )
                                                             : compute_constraints_control_jacobian( problem.equality_constraints, x.col( t ), u.col( t ) );

          const Eigen::VectorXd dual = eq_multipliers[t] + penalty_parameter * eq_residuals[t];
          q_x_step[t] += eq_jacobian_x[t].transpose() * dual;
          q_u_step[t] += eq_jacobian_u[t].transpose() * dual;

          q_xx_step[t] += penalty_parameter * eq_jacobian_x[t].transpose() * eq_jacobian_x[t];
          q_ux_step[t] += penalty_parameter * eq_jacobian_u[t].transpose() * eq_jacobian_x[t];
          q_uu_step[t] += penalty_parameter * eq_jacobian_u[t].transpose() * eq_jacobian_u[t];
        }

        if( inequality_dim > 0 && problem.inequality_constraints )
        {
          ineq_residuals[t] = problem.inequality_constraints( x.col( t ), u.col( t ) );
          ineq_jacobian_x[t]
            = problem.inequality_constraints_state_jacobian
                ? problem.inequality_constraints_state_jacobian( x.col( t ), u.col( t ) )
                : compute_constraints_state_jacobian( problem.inequality_constraints, x.col( t ), u.col( t ) );
          ineq_jacobian_u[t]
            = problem.inequality_constraints_control_jacobian
                ? problem.inequality_constraints_control_jacobian( x.col( t ), u.col( t ) )
                : compute_constraints_control_jacobian( problem.inequality_constraints, x.col( t ), u.col( t ) );

          const Eigen::VectorXd slack  = ineq_residuals[t].cwiseMax( 0.0 );
          const Eigen::ArrayXd   active
            = ( ineq_residuals[t].array() > -inequality_activation_tolerance ).cast<double>();
          const Eigen::VectorXd dual = ineq_multipliers[t].array() * active + penalty_parameter * slack.array() * active;

          q_x_step[t] += ineq_jacobian_x[t].transpose() * dual;
          q_u_step[t] += ineq_jacobian_u[t].transpose() * dual;

          if( active.any() )
          {
            const Eigen::MatrixXd active_diag = active.matrix().asDiagonal();
            q_xx_step[t] += penalty_parameter * ineq_jacobian_x[t].transpose() * active_diag * ineq_jacobian_x[t];
            q_ux_step[t] += penalty_parameter * ineq_jacobian_u[t].transpose() * active_diag * ineq_jacobian_x[t];
            q_uu_step[t] += penalty_parameter * ineq_jacobian_u[t].transpose() * active_diag * ineq_jacobian_u[t];
          }
        }

        q_uu_reg_step[t] = q_uu_step[t];
        auto&  llt       = llt_step[t];
        double reg       = 1e-6;
        while( true )
        {
          llt.compute( q_uu_reg_step[t] );
          if( llt.info() == Eigen::Success )
            break;
          q_uu_reg_step[t] += reg * identity_nu;
          reg *= 10.0;
        }
        q_uu_inv_step[t] = llt.solve( identity_nu );

        k[t]        = -q_uu_inv_step[t] * q_u_step[t];
        k_matrix[t] = -q_uu_inv_step[t] * q_ux_step[t];

        v_x = q_x_step[t] + k_matrix[t].transpose() * q_u_step[t] + q_ux_step[t].transpose() * k[t]
            + k_matrix[t].transpose() * q_uu_step[t] * k[t];
        v_xx = q_xx_step[t] + k_matrix[t].transpose() * q_ux_step[t] + q_ux_step[t].transpose() * k_matrix[t]
             + k_matrix[t].transpose() * q_uu_step[t] * k_matrix[t];
        v_xx = 0.5 * ( v_xx + v_xx.transpose() );
      }

      x_trial.setZero();
      u_trial.setZero();
      x_trial.col( 0 ) = problem.initial_state;

      const double amin = 1e-3;
      double       alpha = 1.0;

      double            best_merit = current_merit;
      StateTrajectory   best_x     = x;
      ControlTrajectory best_u     = u;

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

        const double trial_merit = compute_merit( problem, x_trial, u_trial );
        if( trial_merit < best_merit )
        {
          best_merit = trial_merit;
          best_x     = x_trial;
          best_u     = u_trial;
          break;
        }
        alpha *= 0.5;
      }

      const double improvement = current_merit - best_merit;
      x    = best_x;
      u    = best_u;
      cost = problem.objective_function( x, u );
      current_merit = best_merit;

      double eq_violation_norm   = 0.0;
      double ineq_violation_norm = 0.0;

      for( int t = 0; t < T; ++t )
      {
        if( equality_dim > 0 && problem.equality_constraints )
        {
          const Eigen::VectorXd residual = problem.equality_constraints( x.col( t ), u.col( t ) );
          eq_multipliers[t] += penalty_parameter * residual;
          eq_violation_norm += residual.squaredNorm();
        }
        if( inequality_dim > 0 && problem.inequality_constraints )
        {
          const Eigen::VectorXd residual = problem.inequality_constraints( x.col( t ), u.col( t ) );
          const Eigen::VectorXd positive = residual.cwiseMax( 0.0 );
          ineq_multipliers[t]            = ( ineq_multipliers[t] + penalty_parameter * positive ).cwiseMax( 0.0 );
          ineq_violation_norm += positive.squaredNorm();
        }
      }

      eq_violation_norm   = std::sqrt( eq_violation_norm );
      ineq_violation_norm = std::sqrt( ineq_violation_norm );

      if( eq_violation_norm > constraint_tolerance || ineq_violation_norm > constraint_tolerance )
        penalty_parameter *= penalty_increase;

      if( debug )
      {
        std::cout << "pd-iLQR iter " << iter << ": cost=" << cost << " merit=" << current_merit << " d_merit="
                  << improvement << " eq_violation=" << eq_violation_norm
                  << " ineq_violation=" << ineq_violation_norm << '\n';
      }

      if( improvement < tolerance && eq_violation_norm < constraint_tolerance && ineq_violation_norm < constraint_tolerance )
        break;
    }
  }

private:
  void
  resize_buffers( const OCP& problem )
  {
    const int T  = problem.horizon_steps;
    const int nx = problem.state_dim;
    const int nu = problem.control_dim;

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

    if( static_cast<int>( llt_step.size() ) != T )
      llt_step.assign( T, Eigen::LLT<Eigen::MatrixXd>( nu ) );

    Control default_control = Control::Zero( nu );
    if( problem.initial_controls.cols() == T )
      default_control = problem.initial_controls.col( 0 );

    equality_dim = 0;
    inequality_dim = 0;
    if( problem.equality_constraints )
      equality_dim = static_cast<int>( problem.equality_constraints( problem.initial_state, default_control ).size() );
    if( problem.inequality_constraints )
      inequality_dim = static_cast<int>( problem.inequality_constraints( problem.initial_state, default_control ).size() );

    if( equality_dim > 0 )
    {
      resize_mat_vec( eq_residuals, Eigen::VectorXd::Zero( equality_dim ) );
      resize_mat_vec( eq_jacobian_x, Eigen::MatrixXd::Zero( equality_dim, nx ) );
      resize_mat_vec( eq_jacobian_u, Eigen::MatrixXd::Zero( equality_dim, nu ) );

      if( static_cast<int>( eq_multipliers.size() ) != T )
        eq_multipliers.assign( T, Eigen::VectorXd::Zero( equality_dim ) );
      else
        for( auto& m : eq_multipliers )
        {
          if( m.size() != equality_dim )
            m = Eigen::VectorXd::Zero( equality_dim );
        }
    }
    else
    {
      eq_residuals.clear();
      eq_jacobian_x.clear();
      eq_jacobian_u.clear();
      eq_multipliers.clear();
    }

    if( inequality_dim > 0 )
    {
      resize_mat_vec( ineq_residuals, Eigen::VectorXd::Zero( inequality_dim ) );
      resize_mat_vec( ineq_jacobian_x, Eigen::MatrixXd::Zero( inequality_dim, nx ) );
      resize_mat_vec( ineq_jacobian_u, Eigen::MatrixXd::Zero( inequality_dim, nu ) );

      if( static_cast<int>( ineq_multipliers.size() ) != T )
        ineq_multipliers.assign( T, Eigen::VectorXd::Zero( inequality_dim ) );
      else
        for( auto& m : ineq_multipliers )
        {
          if( m.size() != inequality_dim )
            m = Eigen::VectorXd::Zero( inequality_dim );
        }
    }
    else
    {
      ineq_residuals.clear();
      ineq_jacobian_x.clear();
      ineq_jacobian_u.clear();
      ineq_multipliers.clear();
    }

    x_trial.resize( nx, T + 1 );
    u_trial.resize( nu, T );

    v_x.resize( nx );
    v_xx.resize( nx, nx );
    identity_nu = Eigen::MatrixXd::Identity( nu, nu );
  }

  double
  compute_merit( const OCP& problem, const StateTrajectory& states, const ControlTrajectory& controls ) const
  {
    const int T = problem.horizon_steps;
    double    merit
      = problem.objective_function ? problem.objective_function( states, controls ) : problem.best_cost;

    for( int t = 0; t < T; ++t )
    {
      if( equality_dim > 0 && problem.equality_constraints )
      {
        const Eigen::VectorXd residual = problem.equality_constraints( states.col( t ), controls.col( t ) );
        merit += eq_multipliers[t].dot( residual ) + 0.5 * penalty_parameter * residual.squaredNorm();
      }
      if( inequality_dim > 0 && problem.inequality_constraints )
      {
        const Eigen::VectorXd residual = problem.inequality_constraints( states.col( t ), controls.col( t ) );
        const Eigen::VectorXd slack    = residual.cwiseMax( 0.0 );
        const Eigen::ArrayXd   active
          = ( residual.array() > -inequality_activation_tolerance ).cast<double>();
        const Eigen::VectorXd active_slack      = slack.array() * active;
        const Eigen::VectorXd weighted_multiply = ineq_multipliers[t].array() * active;
        merit += weighted_multiply.dot( active_slack );
        merit += 0.5 * penalty_parameter * active_slack.squaredNorm();
      }
    }

    return merit;
  }

  int    max_iterations;
  double tolerance;
  double max_ms;
  bool   debug;

  double penalty_parameter;
  double penalty_increase;
  double constraint_tolerance;
  double inequality_activation_tolerance;

  int equality_dim;
  int inequality_dim;

  std::vector<Eigen::VectorXd> k;
  std::vector<Eigen::MatrixXd> k_matrix;

  std::vector<Eigen::MatrixXd> a_step;
  std::vector<Eigen::MatrixXd> b_step;

  std::vector<Eigen::VectorXd> l_x_step;
  std::vector<Eigen::VectorXd> l_u_step;
  std::vector<Eigen::MatrixXd> l_xx_step;
  std::vector<Eigen::MatrixXd> l_uu_step;
  std::vector<Eigen::MatrixXd> l_ux_step;

  std::vector<Eigen::VectorXd> q_x_step;
  std::vector<Eigen::VectorXd> q_u_step;
  std::vector<Eigen::MatrixXd> q_xx_step;
  std::vector<Eigen::MatrixXd> q_ux_step;
  std::vector<Eigen::MatrixXd> q_uu_step;
  std::vector<Eigen::MatrixXd> q_uu_reg_step;
  std::vector<Eigen::MatrixXd> q_uu_inv_step;

  std::vector<Eigen::LLT<Eigen::MatrixXd>> llt_step;

  std::vector<Eigen::VectorXd> eq_residuals;
  std::vector<Eigen::MatrixXd> eq_jacobian_x;
  std::vector<Eigen::MatrixXd> eq_jacobian_u;
  std::vector<Eigen::VectorXd> eq_multipliers;

  std::vector<Eigen::VectorXd> ineq_residuals;
  std::vector<Eigen::MatrixXd> ineq_jacobian_x;
  std::vector<Eigen::MatrixXd> ineq_jacobian_u;
  std::vector<Eigen::VectorXd> ineq_multipliers;

  StateTrajectory   x_trial;
  ControlTrajectory u_trial;

  Eigen::VectorXd v_x;
  Eigen::MatrixXd v_xx;
  Eigen::MatrixXd identity_nu;
};

} // namespace mas
