#pragma once

#include <cmath>

#include <algorithm>
#include <functional>

#include <Eigen/Dense>

#include "multi_agent_solver/integrator.hpp"
#include "multi_agent_solver/types.hpp"

namespace mas
{

//================================================================
// Finite Differences for the Overall Trajectory Cost
//================================================================
template<typename Scalar>
inline ControlGradientT<Scalar>
finite_differences_gradient( const StateT<Scalar>& initial_state, const ControlTrajectoryT<Scalar>& controls,
                             const MotionModelT<Scalar>& dynamics, const ObjectiveFunctionT<Scalar>& objective_function, Scalar dt )
{
  ControlGradientT<Scalar>   gradients      = ControlGradientT<Scalar>::Zero( controls.rows(), controls.cols() );
  ControlTrajectoryT<Scalar> controls_minus = controls;
  ControlTrajectoryT<Scalar> controls_plus  = controls;

  StateTrajectoryT<Scalar> trajectory_minus = integrate_horizon<Scalar>( initial_state, controls_minus, dt, dynamics,
                                                                         integrate_rk4<Scalar> );
  Scalar                   cost_minus       = objective_function( trajectory_minus, controls_minus );

  for( int t = 0; t < controls.cols(); ++t )
  {
    for( int i = 0; i < controls.rows(); ++i )
    {
      const Scalar epsilon = std::max( static_cast<Scalar>( 1e-6 ), static_cast<Scalar>( 1e-8 ) * std::abs( controls( i, t ) ) );

      controls_plus                             = controls;
      controls_plus( i, t )                    += epsilon;
      StateTrajectoryT<Scalar> trajectory_plus  = integrate_horizon<Scalar>( initial_state, controls_plus, dt, dynamics,
                                                                             integrate_rk4<Scalar> );
      Scalar                   cost_plus        = objective_function( trajectory_plus, controls_plus );

      controls_minus          = controls;
      controls_minus( i, t ) -= epsilon;
      trajectory_minus        = integrate_horizon<Scalar>( initial_state, controls_minus, dt, dynamics, integrate_rk4<Scalar> );
      cost_minus              = objective_function( trajectory_minus, controls_minus );

      gradients( i, t ) = ( cost_plus - cost_minus ) / ( static_cast<Scalar>( 2 ) * epsilon );
    }
  }
  return gradients;
}

inline ControlGradient
finite_differences_gradient( const State& initial_state, const ControlTrajectory& controls, const MotionModel& dynamics,
                             const ObjectiveFunction& objective_function, double dt )
{
  return finite_differences_gradient<double>( initial_state, controls, dynamics, objective_function, dt );
}

//================================================================
// Finite Differences for the Dynamics
//================================================================
template<typename Scalar>
inline Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
compute_dynamics_state_jacobian( const MotionModelT<Scalar>& dynamics, const StateT<Scalar>& x, const ControlT<Scalar>& u )
{
  const int                                             state_dim = static_cast<int>( x.size() );
  const Scalar                                          epsilon   = static_cast<Scalar>( 1e-6 );
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> A         = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Zero( state_dim,
                                                                                                                                 state_dim );

  for( int i = 0; i < state_dim; ++i )
  {
    StateT<Scalar> dx = StateT<Scalar>::Zero( state_dim );
    dx( i )           = epsilon;

    StateT<Scalar> f_plus  = dynamics( x + dx, u );
    StateT<Scalar> f_minus = dynamics( x - dx, u );

    A.col( i ) = ( f_plus - f_minus ) / ( static_cast<Scalar>( 2 ) * epsilon );
  }
  return A;
}

inline Eigen::MatrixXd
compute_dynamics_state_jacobian( const MotionModel& dynamics, const State& x, const Control& u )
{
  return compute_dynamics_state_jacobian<double>( dynamics, x, u );
}

template<typename Scalar>
inline Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
compute_dynamics_control_jacobian( const MotionModelT<Scalar>& dynamics, const StateT<Scalar>& x, const ControlT<Scalar>& u )
{
  const int                                             state_dim   = static_cast<int>( x.size() );
  const int                                             control_dim = static_cast<int>( u.size() );
  const Scalar                                          epsilon     = static_cast<Scalar>( 1e-6 );
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> B = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Zero( state_dim,
                                                                                                                         control_dim );

  for( int i = 0; i < control_dim; ++i )
  {
    ControlT<Scalar> du = ControlT<Scalar>::Zero( control_dim );
    du( i )             = epsilon;

    StateT<Scalar> f_plus  = dynamics( x, u + du );
    StateT<Scalar> f_minus = dynamics( x, u - du );

    B.col( i ) = ( f_plus - f_minus ) / ( static_cast<Scalar>( 2 ) * epsilon );
  }
  return B;
}

inline Eigen::MatrixXd
compute_dynamics_control_jacobian( const MotionModel& dynamics, const State& x, const Control& u )
{
  return compute_dynamics_control_jacobian<double>( dynamics, x, u );
}

// Safe evaluation wrapper
template<typename Scalar>
inline Scalar
safe_eval( const StageCostFunctionT<Scalar>& stage_cost, const StateT<Scalar>& x, const ControlT<Scalar>& u, size_t time_idx )
{
  const Scalar value = stage_cost( x, u, time_idx );
  return std::isfinite( static_cast<double>( value ) ) ? value : static_cast<Scalar>( 0 );
}

template<typename Scalar>
inline Scalar
safe_eval_terminal( const TerminalCostFunctionT<Scalar>& terminal_cost, const StateT<Scalar>& x )
{
  const Scalar value = terminal_cost( x );
  return std::isfinite( static_cast<double>( value ) ) ? value : static_cast<Scalar>( 0 );
}

inline double
safe_eval( const StageCostFunction& stage_cost, const State& x, const Control& u, size_t time_idx )
{
  return safe_eval<double>( stage_cost, x, u, time_idx );
}

inline double
safe_eval_terminal( const TerminalCostFunction& terminal_cost, const State& x )
{
  return safe_eval_terminal<double>( terminal_cost, x );
}

// Cost derivatives
template<typename Scalar>
inline StateT<Scalar>
compute_cost_state_gradient( const StageCostFunctionT<Scalar>& stage_cost, const StateT<Scalar>& x, const ControlT<Scalar>& u,
                             size_t time_idx )
{
  StateT<Scalar> grad    = StateT<Scalar>::Zero( x.size() );
  const Scalar   epsilon = static_cast<Scalar>( 1e-6 );
  for( int i = 0; i < x.size(); ++i )
  {
    StateT<Scalar> dx = StateT<Scalar>::Zero( x.size() );
    dx( i )           = epsilon;
    grad( i )         = ( stage_cost( x + dx, u, time_idx ) - stage_cost( x - dx, u, time_idx ) ) / ( static_cast<Scalar>( 2 ) * epsilon );
  }
  return grad;
}

inline Eigen::VectorXd
compute_cost_state_gradient( const StageCostFunction& stage_cost, const State& x, const Control& u, size_t time_idx )
{
  return compute_cost_state_gradient<double>( stage_cost, x, u, time_idx );
}

template<typename Scalar>
inline ControlT<Scalar>
compute_cost_control_gradient( const StageCostFunctionT<Scalar>& stage_cost, const StateT<Scalar>& x, const ControlT<Scalar>& u,
                               size_t time_idx )
{
  ControlT<Scalar> grad    = ControlT<Scalar>::Zero( u.size() );
  const Scalar     epsilon = static_cast<Scalar>( 1e-6 );
  for( int i = 0; i < u.size(); ++i )
  {
    ControlT<Scalar> du = ControlT<Scalar>::Zero( u.size() );
    du( i )             = epsilon;
    grad( i ) = ( stage_cost( x, u + du, time_idx ) - stage_cost( x, u - du, time_idx ) ) / ( static_cast<Scalar>( 2 ) * epsilon );
  }
  return grad;
}

inline Eigen::VectorXd
compute_cost_control_gradient( const StageCostFunction& stage_cost, const State& x, const Control& u, size_t time_idx )
{
  return compute_cost_control_gradient<double>( stage_cost, x, u, time_idx );
}

template<typename Scalar>
inline Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
compute_cost_state_hessian( const StageCostFunctionT<Scalar>& stage_cost, const StateT<Scalar>& x, const ControlT<Scalar>& u,
                            size_t time_idx )
{
  const int                                             n       = static_cast<int>( x.size() );
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> H       = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Zero( n, n );
  const Scalar                                          epsilon = static_cast<Scalar>( 1e-5 );

  for( int i = 0; i < n; ++i )
  {
    StateT<Scalar> dx = StateT<Scalar>::Zero( n );
    dx( i )           = epsilon;
    Scalar f_plus     = safe_eval<Scalar>( stage_cost, x + dx, u, time_idx );
    Scalar f          = safe_eval<Scalar>( stage_cost, x, u, time_idx );
    Scalar f_minus    = safe_eval<Scalar>( stage_cost, x - dx, u, time_idx );
    H( i, i )         = ( f_plus - static_cast<Scalar>( 2 ) * f + f_minus ) / ( epsilon * epsilon );
  }

  for( int i = 0; i < n; ++i )
  {
    for( int j = 0; j < n; ++j )
    {
      if( i != j )
      {
        StateT<Scalar> dx_i = StateT<Scalar>::Zero( n );
        StateT<Scalar> dx_j = StateT<Scalar>::Zero( n );
        dx_i( i )           = epsilon;
        dx_j( j )           = epsilon;
        const Scalar f_pp   = safe_eval<Scalar>( stage_cost, x + dx_i + dx_j, u, time_idx );
        const Scalar f_pm   = safe_eval<Scalar>( stage_cost, x + dx_i - dx_j, u, time_idx );
        const Scalar f_mp   = safe_eval<Scalar>( stage_cost, x - dx_i + dx_j, u, time_idx );
        const Scalar f_mm   = safe_eval<Scalar>( stage_cost, x - dx_i - dx_j, u, time_idx );
        H( i, j )           = ( f_pp - f_pm - f_mp + f_mm ) / ( static_cast<Scalar>( 4 ) * epsilon * epsilon );
      }
    }
  }
  return H;
}

inline Eigen::MatrixXd
compute_cost_state_hessian( const StageCostFunction& stage_cost, const State& x, const Control& u, size_t time_idx )
{
  return compute_cost_state_hessian<double>( stage_cost, x, u, time_idx );
}

template<typename Scalar>
inline Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
compute_cost_control_hessian( const StageCostFunctionT<Scalar>& stage_cost, const StateT<Scalar>& x, const ControlT<Scalar>& u,
                              size_t time_idx )
{
  const int                                             m       = static_cast<int>( u.size() );
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> H       = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Zero( m, m );
  const Scalar                                          epsilon = static_cast<Scalar>( 1e-5 );

  for( int i = 0; i < m; ++i )
  {
    ControlT<Scalar> du = ControlT<Scalar>::Zero( m );
    du( i )             = epsilon;
    Scalar f_plus       = safe_eval<Scalar>( stage_cost, x, u + du, time_idx );
    Scalar f            = safe_eval<Scalar>( stage_cost, x, u, time_idx );
    Scalar f_minus      = safe_eval<Scalar>( stage_cost, x, u - du, time_idx );
    H( i, i )           = ( f_plus - static_cast<Scalar>( 2 ) * f + f_minus ) / ( epsilon * epsilon );
  }

  for( int i = 0; i < m; ++i )
  {
    for( int j = 0; j < m; ++j )
    {
      if( i != j )
      {
        ControlT<Scalar> du_i = ControlT<Scalar>::Zero( m );
        ControlT<Scalar> du_j = ControlT<Scalar>::Zero( m );
        du_i( i )             = epsilon;
        du_j( j )             = epsilon;
        const Scalar f_pp     = safe_eval<Scalar>( stage_cost, x, u + du_i + du_j, time_idx );
        const Scalar f_pm     = safe_eval<Scalar>( stage_cost, x, u + du_i - du_j, time_idx );
        const Scalar f_mp     = safe_eval<Scalar>( stage_cost, x, u - du_i + du_j, time_idx );
        const Scalar f_mm     = safe_eval<Scalar>( stage_cost, x, u - du_i - du_j, time_idx );
        H( i, j )             = ( f_pp - f_pm - f_mp + f_mm ) / ( static_cast<Scalar>( 4 ) * epsilon * epsilon );
      }
    }
  }
  return H;
}

inline Eigen::MatrixXd
compute_cost_control_hessian( const StageCostFunction& stage_cost, const State& x, const Control& u, size_t time_idx )
{
  return compute_cost_control_hessian<double>( stage_cost, x, u, time_idx );
}

template<typename Scalar>
inline Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
compute_terminal_cost_hessian( const TerminalCostFunctionT<Scalar>& terminal_cost, const StateT<Scalar>& x )
{
  const int                                             n       = static_cast<int>( x.size() );
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> H       = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Zero( n, n );
  const Scalar                                          epsilon = static_cast<Scalar>( 1e-5 );

  for( int i = 0; i < n; ++i )
  {
    StateT<Scalar> dx = StateT<Scalar>::Zero( n );
    dx( i )           = epsilon;
    Scalar f_plus     = safe_eval_terminal<Scalar>( terminal_cost, x + dx );
    Scalar f          = safe_eval_terminal<Scalar>( terminal_cost, x );
    Scalar f_minus    = safe_eval_terminal<Scalar>( terminal_cost, x - dx );
    H( i, i )         = ( f_plus - static_cast<Scalar>( 2 ) * f + f_minus ) / ( epsilon * epsilon );
  }

  for( int i = 0; i < n; ++i )
  {
    for( int j = 0; j < n; ++j )
    {
      if( i != j )
      {
        StateT<Scalar> dx_i = StateT<Scalar>::Zero( n );
        StateT<Scalar> dx_j = StateT<Scalar>::Zero( n );
        dx_i( i )           = epsilon;
        dx_j( j )           = epsilon;
        const Scalar f_pp   = safe_eval_terminal<Scalar>( terminal_cost, x + dx_i + dx_j );
        const Scalar f_pm   = safe_eval_terminal<Scalar>( terminal_cost, x + dx_i - dx_j );
        const Scalar f_mp   = safe_eval_terminal<Scalar>( terminal_cost, x - dx_i + dx_j );
        const Scalar f_mm   = safe_eval_terminal<Scalar>( terminal_cost, x - dx_i - dx_j );
        H( i, j )           = ( f_pp - f_pm - f_mp + f_mm ) / ( static_cast<Scalar>( 4 ) * epsilon * epsilon );
      }
    }
  }
  return H;
}

template<typename Scalar>
inline StateT<Scalar>
compute_terminal_cost_gradient( const TerminalCostFunctionT<Scalar>& terminal_cost, const StateT<Scalar>& x )
{
  StateT<Scalar> grad    = StateT<Scalar>::Zero( x.size() );
  const Scalar   epsilon = static_cast<Scalar>( 1e-6 );
  for( int i = 0; i < x.size(); ++i )
  {
    StateT<Scalar> dx = StateT<Scalar>::Zero( x.size() );
    dx( i )           = epsilon;
    grad( i )         = ( terminal_cost( x + dx ) - terminal_cost( x - dx ) ) / ( static_cast<Scalar>( 2 ) * epsilon );
  }
  return grad;
}

inline Eigen::VectorXd
compute_terminal_cost_gradient( const TerminalCostFunction& terminal_cost, const State& x )
{
  return compute_terminal_cost_gradient<double>( terminal_cost, x );
}

inline Eigen::MatrixXd
compute_terminal_cost_hessian( const TerminalCostFunction& terminal_cost, const State& x )
{
  return compute_terminal_cost_hessian<double>( terminal_cost, x );
}

template<typename Scalar>
inline Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
compute_cost_cross_term( const StageCostFunctionT<Scalar>& stage_cost, const StateT<Scalar>& x, const ControlT<Scalar>& u, size_t time_idx )
{
  const int                                             m       = static_cast<int>( u.size() );
  const int                                             n       = static_cast<int>( x.size() );
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> H       = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Zero( m, n );
  const Scalar                                          epsilon = static_cast<Scalar>( 1e-6 );

  for( int i = 0; i < m; ++i )
  {
    for( int j = 0; j < n; ++j )
    {
      ControlT<Scalar> du = ControlT<Scalar>::Zero( m );
      StateT<Scalar>   dx = StateT<Scalar>::Zero( n );
      du( i )             = epsilon;
      dx( j )             = epsilon;
      const Scalar f_pp   = safe_eval<Scalar>( stage_cost, x + dx, u + du, time_idx );
      const Scalar f_pm   = safe_eval<Scalar>( stage_cost, x - dx, u + du, time_idx );
      const Scalar f_mp   = safe_eval<Scalar>( stage_cost, x + dx, u - du, time_idx );
      const Scalar f_mm   = safe_eval<Scalar>( stage_cost, x - dx, u - du, time_idx );
      H( i, j )           = ( f_pp - f_pm - f_mp + f_mm ) / ( static_cast<Scalar>( 4 ) * epsilon * epsilon );
    }
  }
  return H;
}

inline Eigen::MatrixXd
compute_cost_cross_term( const StageCostFunction& stage_cost, const State& x, const Control& u, size_t time_idx )
{
  return compute_cost_cross_term<double>( stage_cost, x, u, time_idx );
}

template<typename Scalar>
inline ConstraintsJacobianT<Scalar>
compute_constraints_state_jacobian( const ConstraintsFunctionT<Scalar>& constraint, const StateT<Scalar>& x, const ControlT<Scalar>& u )
{
  if( !constraint )
    return ConstraintsJacobianT<Scalar>{};

  const ConstraintViolationsT<Scalar> base = constraint( x, u );
  const int                           m    = static_cast<int>( base.size() );
  if( m == 0 )
    return ConstraintsJacobianT<Scalar>{};

  const int                    n       = static_cast<int>( x.size() );
  const Scalar                 epsilon = static_cast<Scalar>( 1e-6 );
  ConstraintsJacobianT<Scalar> J       = ConstraintsJacobianT<Scalar>::Zero( m, n );

  for( int i = 0; i < n; ++i )
  {
    StateT<Scalar> dx = StateT<Scalar>::Zero( n );
    dx( i )           = epsilon;

    ConstraintViolationsT<Scalar> f_plus  = constraint( x + dx, u );
    ConstraintViolationsT<Scalar> f_minus = constraint( x - dx, u );

    J.col( i ) = ( f_plus - f_minus ) / ( static_cast<Scalar>( 2 ) * epsilon );
  }

  return J;
}

inline ConstraintsJacobian
compute_constraints_state_jacobian( const ConstraintsFunction& constraint, const State& x, const Control& u )
{
  return compute_constraints_state_jacobian<double>( constraint, x, u );
}

template<typename Scalar>
inline ConstraintsJacobianT<Scalar>
compute_constraints_control_jacobian( const ConstraintsFunctionT<Scalar>& constraint, const StateT<Scalar>& x, const ControlT<Scalar>& u )
{
  if( !constraint )
    return ConstraintsJacobianT<Scalar>{};

  const ConstraintViolationsT<Scalar> base = constraint( x, u );
  const int                           m    = static_cast<int>( base.size() );
  if( m == 0 )
    return ConstraintsJacobianT<Scalar>{};

  const int                    p       = static_cast<int>( u.size() );
  const Scalar                 epsilon = static_cast<Scalar>( 1e-6 );
  ConstraintsJacobianT<Scalar> J       = ConstraintsJacobianT<Scalar>::Zero( m, p );

  for( int i = 0; i < p; ++i )
  {
    ControlT<Scalar> du = ControlT<Scalar>::Zero( p );
    du( i )             = epsilon;

    ConstraintViolationsT<Scalar> f_plus  = constraint( x, u + du );
    ConstraintViolationsT<Scalar> f_minus = constraint( x, u - du );

    J.col( i ) = ( f_plus - f_minus ) / ( static_cast<Scalar>( 2 ) * epsilon );
  }

  return J;
}

inline ConstraintsJacobian
compute_constraints_control_jacobian( const ConstraintsFunction& constraint, const State& x, const Control& u )
{
  return compute_constraints_control_jacobian<double>( constraint, x, u );
}

} // namespace mas
