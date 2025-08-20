#pragma once

#include <algorithm>
#include <functional>
#include <cmath>

#include <Eigen/Dense>

#include "multi_agent_solver/integrator.hpp"
#include "multi_agent_solver/types.hpp"

namespace mas
{

//================================================================
// Finite Differences for the Overall Trajectory Cost
//================================================================
inline ControlGradient
finite_differences_gradient( const State& initial_state, const ControlTrajectory& controls, const MotionModel& dynamics,
                             const ObjectiveFunction& objective_function, double dt )
{
  ControlGradient   gradients      = ControlGradient::Zero( controls.rows(), controls.cols() );
  ControlTrajectory controls_minus = controls;
  ControlTrajectory controls_plus  = controls;

  StateTrajectory trajectory_minus = integrate_horizon( initial_state, controls_minus, dt, dynamics, integrate_rk4 );
  double          cost_minus       = to_double( objective_function( trajectory_minus, controls_minus ) );

  for( int t = 0; t < controls.cols(); ++t )
  {
    for( int i = 0; i < controls.rows(); ++i )
    {
      // Use raw double value for epsilon to avoid issues with autodiff types
      double epsilon = std::max( 1e-6, 1e-8 * std::abs( to_double( controls( i, t ) ) ) );

      controls_plus                    = controls;
      controls_plus( i, t )           += epsilon;
      StateTrajectory trajectory_plus  = integrate_horizon( initial_state, controls_plus, dt, dynamics, integrate_rk4 );
      double          cost_plus        = to_double( objective_function( trajectory_plus, controls_plus ) );

      controls_minus          = controls;
      controls_minus( i, t ) -= epsilon;
      trajectory_minus        = integrate_horizon( initial_state, controls_minus, dt, dynamics, integrate_rk4 );
      cost_minus              = to_double( objective_function( trajectory_minus, controls_minus ) );

      gradients( i, t ) = ( cost_plus - cost_minus ) / ( 2 * epsilon );
    }
  }
  return gradients;
}

//================================================================
// Finite Differences for the Dynamics
//================================================================
inline Eigen::MatrixXd
compute_dynamics_state_jacobian( const MotionModel& dynamics, const State& x, const Control& u )
{
  const int       state_dim = x.size();
  const double    epsilon   = 1e-6;
  Eigen::MatrixXd A         = Eigen::MatrixXd::Zero( state_dim, state_dim );

  for( int i = 0; i < state_dim; ++i )
  {
    State dx = State::Zero( state_dim );
    dx( i )  = epsilon;

    State f_plus  = dynamics( x + dx, u );
    State f_minus = dynamics( x - dx, u );

    A.col( i ) = ( ( f_plus - f_minus ) / ( 2 * epsilon ) ).unaryExpr( []( const Scalar& s ) { return to_double( s ); } );
  }
  return A;
}

inline Eigen::MatrixXd
compute_dynamics_control_jacobian( const MotionModel& dynamics, const State& x, const Control& u )
{
  const int       state_dim   = x.size();
  const int       control_dim = u.size();
  const double    epsilon     = 1e-6;
  Eigen::MatrixXd B           = Eigen::MatrixXd::Zero( state_dim, control_dim );

  for( int i = 0; i < control_dim; ++i )
  {
    Control du = Control::Zero( control_dim );
    du( i )    = epsilon;

    State f_plus  = dynamics( x, u + du );
    State f_minus = dynamics( x, u - du );

    B.col( i ) = ( ( f_plus - f_minus ) / ( 2 * epsilon ) ).unaryExpr( []( const Scalar& s ) { return to_double( s ); } );
  }
  return B;
}

// Safe evaluation wrapper
inline double
safe_eval( const StageCostFunction& stage_cost, const State& x, const Control& u, size_t time_idx )
{
  double value = to_double( stage_cost( x, u, time_idx ) );
  return std::isfinite( value ) ? value : 0.0;
}

// Cost derivatives
inline Eigen::VectorXd
compute_cost_state_gradient( const StageCostFunction& stage_cost, const State& x, const Control& u, size_t time_idx )
{
  Eigen::VectorXd grad    = Eigen::VectorXd::Zero( x.size() );
  const double    epsilon = 1e-6;
  for( int i = 0; i < x.size(); ++i )
  {
    State dx  = State::Zero( x.size() );
    dx( i )   = epsilon;
    grad( i ) = ( to_double( stage_cost( x + dx, u, time_idx ) ) - to_double( stage_cost( x - dx, u, time_idx ) ) ) / ( 2 * epsilon );
  }
  return grad;
}

inline Eigen::VectorXd
compute_cost_control_gradient( const StageCostFunction& stage_cost, const State& x, const Control& u, size_t time_idx )
{
  Eigen::VectorXd grad    = Eigen::VectorXd::Zero( u.size() );
  const double    epsilon = 1e-6;
  for( int i = 0; i < u.size(); ++i )
  {
    Control du = Control::Zero( u.size() );
    du( i )    = epsilon;
    grad( i )  = ( to_double( stage_cost( x, u + du, time_idx ) ) - to_double( stage_cost( x, u - du, time_idx ) ) ) / ( 2 * epsilon );
  }
  return grad;
}

inline Eigen::MatrixXd
compute_cost_state_hessian( const StageCostFunction& stage_cost, const State& x, const Control& u, size_t time_idx )
{
  const int       n       = x.size();
  Eigen::MatrixXd H       = Eigen::MatrixXd::Zero( n, n );
  const double    epsilon = 1e-5;

  for( int i = 0; i < n; ++i )
  {
    State dx       = State::Zero( n );
    dx( i )        = epsilon;
    double f_plus  = safe_eval( stage_cost, x + dx, u, time_idx );
    double f       = safe_eval( stage_cost, x, u, time_idx );
    double f_minus = safe_eval( stage_cost, x - dx, u, time_idx );
    H( i, i )      = ( f_plus - 2 * f + f_minus ) / ( epsilon * epsilon );
  }

  for( int i = 0; i < n; ++i )
  {
    for( int j = 0; j < n; ++j )
    {
      if( i != j )
      {
        State dx_i = State::Zero( n ), dx_j = State::Zero( n );
        dx_i( i )   = epsilon;
        dx_j( j )   = epsilon;
        double f_pp = safe_eval( stage_cost, x + dx_i + dx_j, u, time_idx );
        double f_pm = safe_eval( stage_cost, x + dx_i - dx_j, u, time_idx );
        double f_mp = safe_eval( stage_cost, x - dx_i + dx_j, u, time_idx );
        double f_mm = safe_eval( stage_cost, x - dx_i - dx_j, u, time_idx );
        H( i, j )   = ( f_pp - f_pm - f_mp + f_mm ) / ( 4 * epsilon * epsilon );
      }
    }
  }
  return H;
}

inline Eigen::MatrixXd
compute_cost_control_hessian( const StageCostFunction& stage_cost, const State& x, const Control& u, size_t time_idx )
{
  const int       m       = u.size();
  Eigen::MatrixXd H       = Eigen::MatrixXd::Zero( m, m );
  const double    epsilon = 1e-5;

  for( int i = 0; i < m; ++i )
  {
    Control du     = Control::Zero( m );
    du( i )        = epsilon;
    double f_plus  = safe_eval( stage_cost, x, u + du, time_idx );
    double f       = safe_eval( stage_cost, x, u, time_idx );
    double f_minus = safe_eval( stage_cost, x, u - du, time_idx );
    H( i, i )      = ( f_plus - 2 * f + f_minus ) / ( epsilon * epsilon );
  }

  for( int i = 0; i < m; ++i )
  {
    for( int j = 0; j < m; ++j )
    {
      if( i != j )
      {
        Control du_i = Control::Zero( m ), du_j = Control::Zero( m );
        du_i( i )   = epsilon;
        du_j( j )   = epsilon;
        double f_pp = safe_eval( stage_cost, x, u + du_i + du_j, time_idx );
        double f_pm = safe_eval( stage_cost, x, u + du_i - du_j, time_idx );
        double f_mp = safe_eval( stage_cost, x, u - du_i + du_j, time_idx );
        double f_mm = safe_eval( stage_cost, x, u - du_i - du_j, time_idx );
        H( i, j )   = ( f_pp - f_pm - f_mp + f_mm ) / ( 4 * epsilon * epsilon );
      }
    }
  }
  return H;
}

inline Eigen::MatrixXd
compute_cost_cross_term( const StageCostFunction& stage_cost, const State& x, const Control& u, size_t time_idx )
{
  const int       m       = u.size();
  const int       n       = x.size();
  Eigen::MatrixXd H       = Eigen::MatrixXd::Zero( m, n );
  const double    epsilon = 1e-6;

  for( int i = 0; i < m; ++i )
  {
    for( int j = 0; j < n; ++j )
    {
      Control du  = Control::Zero( m );
      State   dx  = State::Zero( n );
      du( i )     = epsilon;
      dx( j )     = epsilon;
      double f_pp = safe_eval( stage_cost, x + dx, u + du, time_idx );
      double f_pm = safe_eval( stage_cost, x - dx, u + du, time_idx );
      double f_mp = safe_eval( stage_cost, x + dx, u - du, time_idx );
      double f_mm = safe_eval( stage_cost, x - dx, u - du, time_idx );
      H( i, j )   = ( f_pp - f_pm - f_mp + f_mm ) / ( 4 * epsilon * epsilon );
    }
  }
  return H;
}
} // namespace mas
