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

/**
 * @brief Multiple-shooting variant of iLQR solved with a dense Gauss-Newton
 *        step and augmented Lagrangian enforcement of the shooting defects.
 */
class MultiShootILQR
{
public:
  MultiShootILQR()
    : max_iterations( 50 )
    , tolerance( 1e-6 )
    , max_ms( std::numeric_limits<double>::infinity() )
    , debug( false )
    , penalty_parameter( 10.0 )
    , penalty_increase( 5.0 )
    , constraint_tolerance( 1e-4 )
  {}

  void
  set_params( const SolverParams& params )
  {
    if( auto it = params.find( "max_iterations" ); it != params.end() )
      max_iterations = static_cast<int>( it->second );
    if( auto it = params.find( "tolerance" ); it != params.end() )
      tolerance = it->second;
    if( auto it = params.find( "max_ms" ); it != params.end() )
      max_ms = it->second;
    if( auto it = params.find( "debug" ); it != params.end() )
      debug = it->second > 0.5;
    if( auto it = params.find( "penalty" ); it != params.end() )
      penalty_parameter = it->second;
    if( auto it = params.find( "penalty_increase" ); it != params.end() )
      penalty_increase = it->second;
    if( auto it = params.find( "constraint_tolerance" ); it != params.end() )
      constraint_tolerance = it->second;
  }

  void
  solve( OCP& problem )
  {
    using clock      = std::chrono::high_resolution_clock;
    const auto start = clock::now();

    problem.initialize_problem();
    resize_buffers( problem );

    StateTrajectory   x = problem.best_states;
    ControlTrajectory u = problem.best_controls;
    const bool        clamped_initial = enforce_input_bounds( u, problem );
    if( clamped_initial )
      x = integrate_horizon( problem.initial_state, u, problem.dt, problem.dynamics, integrate_rk4 );
    double            cost
      = problem.objective_function ? problem.objective_function( x, u ) : problem.best_cost;

    compute_residuals_and_jacobians( problem, x, u );
    double merit = augmented_merit( cost );

    if( debug )
      std::cout << "MultiShootILQR initial cost=" << cost << " merit=" << merit << '\n';

    double prev_constraint_norm = total_defect_norm();

    for( int iter = 0; iter < max_iterations; ++iter )
    {
      const double elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>( clock::now() - start ).count();
      if( elapsed_ms > max_ms )
      {
        if( debug )
          std::cout << "MultiShootILQR time limit hit after " << elapsed_ms << " ms / " << max_ms << " ms\n";
        break;
      }

      Eigen::VectorXd gradient;
      Eigen::MatrixXd hessian;
      build_gauss_newton_system( problem, gradient, hessian );

      // Regularise to guarantee positive definiteness
      hessian.diagonal().array() += 1e-8;
      Eigen::LLT<Eigen::MatrixXd> llt( hessian );
      if( llt.info() != Eigen::Success )
      {
        if( debug )
          std::cout << "MultiShootILQR failed factorisation, increasing regularisation" << '\n';
        hessian.diagonal().array() += 1e-4;
        llt.compute( hessian );
      }
      if( llt.info() != Eigen::Success )
      {
        if( debug )
          std::cout << "MultiShootILQR giving up due to singular system" << '\n';
        break;
      }

      const Eigen::VectorXd step = llt.solve( -gradient );

      const double step_norm = step.norm();
      if( step_norm < tolerance )
      {
        if( debug )
          std::cout << "MultiShootILQR terminating: small step" << '\n';
        break;
      }

      StateTrajectory   candidate_x = x;
      ControlTrajectory candidate_u = u;

      apply_step( step, candidate_x, candidate_u );
      enforce_input_bounds( candidate_u, problem );
      candidate_x.col( 0 ) = problem.initial_state;

      double candidate_cost = problem.objective_function( candidate_x, candidate_u );
      compute_residuals_and_jacobians( problem, candidate_x, candidate_u );
      double candidate_merit = augmented_merit( candidate_cost );

      double alpha    = 1.0;
      bool   accepted = candidate_merit < merit;
      while( !accepted && alpha > 1e-4 )
      {
        alpha *= 0.5;
        candidate_x = x;
        candidate_u = u;
        apply_step( alpha * step, candidate_x, candidate_u );
        enforce_input_bounds( candidate_u, problem );
        candidate_x.col( 0 ) = problem.initial_state;

        candidate_cost = problem.objective_function( candidate_x, candidate_u );
        compute_residuals_and_jacobians( problem, candidate_x, candidate_u );
        candidate_merit = augmented_merit( candidate_cost );
        accepted        = candidate_merit < merit;
      }

      if( !accepted )
      {
        if( debug )
          std::cout << "MultiShootILQR line search failed" << '\n';
        break;
      }

      x     = candidate_x;
      u     = candidate_u;
      cost  = candidate_cost;
      merit = candidate_merit;

      if( debug )
        std::cout << "MultiShootILQR iter " << iter << " cost=" << cost << " merit=" << merit
                  << " defect_norm=" << total_defect_norm() << '\n';

      // Update multipliers and possibly increase penalty
      update_multipliers();

      const double constraint_norm = total_defect_norm();
      if( constraint_norm > prev_constraint_norm * 0.8 )
        penalty_parameter *= penalty_increase;
      prev_constraint_norm = constraint_norm;
      merit                = augmented_merit( cost );

      if( constraint_norm < constraint_tolerance )
      {
        if( debug )
          std::cout << "MultiShootILQR terminating: constraints satisfied" << '\n';
        break;
      }
    }

    problem.best_states   = x;
    problem.best_controls = u;
    problem.best_cost     = cost;
  }

private:
  int    max_iterations;
  double tolerance;
  double max_ms;
  bool   debug;
  double penalty_parameter;
  double penalty_increase;
  double constraint_tolerance;

  std::vector<Eigen::MatrixXd> discrete_A;
  std::vector<Eigen::MatrixXd> discrete_B;
  std::vector<Eigen::VectorXd> defects;
  std::vector<Eigen::VectorXd> multipliers;
  StateTrajectory               cached_states;
  ControlTrajectory             cached_controls;

  void
  resize_buffers( const OCP& problem )
  {
    const int T  = problem.horizon_steps;
    const int nx = problem.state_dim;
    const int nu = problem.control_dim;

    discrete_A.assign( T, Eigen::MatrixXd::Zero( nx, nx ) );
    discrete_B.assign( T, Eigen::MatrixXd::Zero( nx, nu ) );
    defects.assign( T, Eigen::VectorXd::Zero( nx ) );
    multipliers.assign( T, Eigen::VectorXd::Zero( nx ) );

    cached_states   = StateTrajectory::Zero( nx, T + 1 );
    cached_controls = ControlTrajectory::Zero( nu, T );
  }

  void
  compute_residuals_and_jacobians( const OCP& problem, const StateTrajectory& x, const ControlTrajectory& u )
  {
    cached_states   = x;
    cached_controls = u;

    const int T = problem.horizon_steps;
    for( int t = 0; t < T; ++t )
    {
      const State predicted = integrate_rk4( x.col( t ), u.col( t ), problem.dt, problem.dynamics );
      defects[t]            = x.col( t + 1 ) - predicted;
      discrete_A[t]         = discrete_state_jacobian( problem, x.col( t ), u.col( t ) );
      discrete_B[t]         = discrete_control_jacobian( problem, x.col( t ), u.col( t ) );
    }
  }

  Eigen::MatrixXd
  discrete_state_jacobian( const OCP& problem, const State& x, const Control& u ) const
  {
    const int nx      = problem.state_dim;
    const double eps  = 1e-6;
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero( nx, nx );

    for( int i = 0; i < nx; ++i )
    {
      State dx = State::Zero( nx );
      dx( i )  = eps;
      const State forward = integrate_rk4( x + dx, u, problem.dt, problem.dynamics );
      const State back    = integrate_rk4( x - dx, u, problem.dt, problem.dynamics );
      A.col( i )          = ( forward - back ) / ( 2.0 * eps );
    }
    return A;
  }

  Eigen::MatrixXd
  discrete_control_jacobian( const OCP& problem, const State& x, const Control& u ) const
  {
    const int nx      = problem.state_dim;
    const int nu      = problem.control_dim;
    const double eps  = 1e-6;
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero( nx, nu );

    for( int i = 0; i < nu; ++i )
    {
      Control du = Control::Zero( nu );
      du( i )    = eps;
      const State forward = integrate_rk4( x, u + du, problem.dt, problem.dynamics );
      const State back    = integrate_rk4( x, u - du, problem.dt, problem.dynamics );
      B.col( i )          = ( forward - back ) / ( 2.0 * eps );
    }
    return B;
  }

  void
  build_gauss_newton_system( const OCP& problem, Eigen::VectorXd& gradient, Eigen::MatrixXd& hessian ) const
  {
    const int nx = problem.state_dim;
    const int nu = problem.control_dim;
    const int T  = problem.horizon_steps;

    const int x_vars = nx * T; // exclude x0
    const int n_vars = x_vars + nu * T;
    const Eigen::MatrixXd I = Eigen::MatrixXd::Identity( nx, nx );

    gradient = Eigen::VectorXd::Zero( n_vars );
    hessian  = Eigen::MatrixXd::Zero( n_vars, n_vars );

    auto x_index = [&]( int t ) { return ( t - 1 ) * nx; };
    auto u_index = [&]( int t ) { return x_vars + t * nu; };

    for( int t = 0; t < T; ++t )
    {
      const State&  xt = cached_states.col( t );
      const Control& ut = cached_controls.col( t );

      const Eigen::VectorXd l_x  = problem.cost_state_gradient( problem.stage_cost, xt, ut, t );
      const Eigen::VectorXd l_u  = problem.cost_control_gradient( problem.stage_cost, xt, ut, t );
      const Eigen::MatrixXd l_xx = problem.cost_state_hessian( problem.stage_cost, xt, ut, t );
      const Eigen::MatrixXd l_uu = problem.cost_control_hessian( problem.stage_cost, xt, ut, t );
      const Eigen::MatrixXd l_ux = problem.cost_cross_term( problem.stage_cost, xt, ut, t );

      const Eigen::MatrixXd& A = discrete_A[t];
      const Eigen::MatrixXd& B = discrete_B[t];
      const Eigen::VectorXd& r = defects[t];
      const Eigen::VectorXd  dual = multipliers[t] + penalty_parameter * r;

      if( t > 0 )
      {
        const int ix = x_index( t );
        gradient.segment( ix, nx ) += l_x - A.transpose() * dual;
        hessian.block( ix, ix, nx, nx ) += l_xx + penalty_parameter * A.transpose() * A;
      }

      const int iu = u_index( t );
      gradient.segment( iu, nu ) += l_u - B.transpose() * dual;
      hessian.block( iu, iu, nu, nu ) += l_uu + penalty_parameter * B.transpose() * B;

      if( t > 0 )
      {
        const int ix = x_index( t );
        hessian.block( iu, ix, nu, nx ) += l_ux + penalty_parameter * B.transpose() * A;
        hessian.block( ix, iu, nx, nu ) += l_ux.transpose() + penalty_parameter * A.transpose() * B;
      }

      if( t + 1 <= T )
      {
        const int ixp = x_index( t + 1 );
        gradient.segment( ixp, nx ) += dual;
        hessian.block( ixp, ixp, nx, nx ) += penalty_parameter * I;

        if( t > 0 )
        {
          const int ix = x_index( t );
          hessian.block( ix, ixp, nx, nx ) -= penalty_parameter * A.transpose();
          hessian.block( ixp, ix, nx, nx ) -= penalty_parameter * A;
        }

        hessian.block( iu, ixp, nu, nx ) -= penalty_parameter * B.transpose();
        hessian.block( ixp, iu, nx, nu ) -= penalty_parameter * B;
      }
    }
  }

  void
  apply_step( const Eigen::VectorXd& step, StateTrajectory& x, ControlTrajectory& u ) const
  {
    const int nx = x.rows();
    const int nu = u.rows();
    const int T  = u.cols();

    const int x_vars = nx * T;

    for( int t = 1; t <= T; ++t )
    {
      const int ix = ( t - 1 ) * nx;
      x.col( t ) += step.segment( ix, nx );
    }

    for( int t = 0; t < T; ++t )
    {
      const int iu = x_vars + t * nu;
      u.col( t ) += step.segment( iu, nu );
    }
  }

  bool
  enforce_input_bounds( ControlTrajectory& u, const OCP& problem ) const
  {
    if( problem.input_lower_bounds && problem.input_upper_bounds )
    {
      clamp_controls( u, *problem.input_lower_bounds, *problem.input_upper_bounds );
      return true;
    }
    return false;
  }

  double
  augmented_merit( double cost ) const
  {
    double penalty = 0.0;
    for( size_t t = 0; t < defects.size(); ++t )
      penalty += multipliers[t].dot( defects[t] ) + 0.5 * penalty_parameter * defects[t].squaredNorm();
    return cost + penalty;
  }

  void
  update_multipliers()
  {
    for( size_t t = 0; t < multipliers.size(); ++t )
      multipliers[t] += penalty_parameter * defects[t];
  }

  double
  total_defect_norm() const
  {
    double norm = 0.0;
    for( const auto& defect : defects )
      norm += defect.squaredNorm();
    return std::sqrt( norm );
  }
};

} // namespace mas

