#pragma once

#include "integrator.hpp"
#include "types.hpp"
#pragma once
#include <algorithm>
#include <functional>

#include <Eigen/Dense>

#include "integrator.hpp"
#include "types.hpp"

//================================================================
// Finite Differences for the Overall Trajectory Cost
// (This computes the gradient of the full ObjectiveFunction,
//  which in your design is typically set to a lambda wrapping
//  compute_trajectory_cost(stage_cost, terminal_cost).)
//================================================================

// Finite Differences GradientComputer (for the full trajectory)
inline ControlGradient
finite_differences_gradient( const State& initial_state, const ControlTrajectory& controls, const MotionModel& dynamics,
                             const ObjectiveFunction& objective_function, double dt )
{
  bool            forward_only = false; // TODO: Consider making this a parameter
  ControlGradient gradients    = ControlGradient::Zero( controls.rows(), controls.cols() );

  ControlTrajectory controls_minus = controls;
  ControlTrajectory controls_plus  = controls;

  StateTrajectory trajectory_minus = integrate_horizon( initial_state, controls_minus, dt, dynamics, integrate_rk4 );
  double          cost_minus       = objective_function( trajectory_minus, controls_minus );

  // Iterate over each control input
  for( int t = 0; t < controls.cols(); ++t )
  {
    for( int i = 0; i < controls.rows(); ++i )
    {
      double epsilon = std::max( 1e-6, 1e-8 * std::abs( controls( i, t ) ) );

      // Perturb positively
      controls_plus                    = controls;
      controls_plus( i, t )           += epsilon;
      StateTrajectory trajectory_plus  = integrate_horizon( initial_state, controls_plus, dt, dynamics, integrate_rk4 );
      double          cost_plus        = objective_function( trajectory_plus, controls_plus );

      if( !forward_only )
      {
        // Perturb negatively
        controls_minus          = controls;
        controls_minus( i, t ) -= epsilon;
        trajectory_minus        = integrate_horizon( initial_state, controls_minus, dt, dynamics, integrate_rk4 );
        cost_minus              = objective_function( trajectory_minus, controls_minus );
      }

      // Central difference approximation (or forward if forward_only)
      gradients( i, t ) = ( cost_plus - cost_minus ) / ( forward_only ? epsilon : 2 * epsilon );
    }
  }
  return gradients;
}

//================================================================
// Finite Differences for the Dynamics (unchanged)
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

    A.col( i ) = ( f_plus - f_minus ) / ( 2 * epsilon );
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

    B.col( i ) = ( f_plus - f_minus ) / ( 2 * epsilon );
  }
  return B;
}

//----------------------------------------------------------
// Compute cost state gradient (first derivative wrt state)
//----------------------------------------------------------
inline Eigen::VectorXd
compute_cost_state_gradient( const StageCostFunction& stage_cost, const State& x, const Control& u )
{
  Eigen::VectorXd grad    = Eigen::VectorXd::Zero( x.size() );
  const double    epsilon = 1e-6;
  for( int i = 0; i < x.size(); ++i )
  {
    State dx       = State::Zero( x.size() );
    dx( i )        = epsilon;
    double f_plus  = stage_cost( x + dx, u );
    double f_minus = stage_cost( x - dx, u );
    grad( i )      = ( f_plus - f_minus ) / ( 2 * epsilon );
  }
  return grad;
}

//----------------------------------------------------------
// Compute cost control gradient (first derivative wrt control)
//----------------------------------------------------------
inline Eigen::VectorXd
compute_cost_control_gradient( const StageCostFunction& stage_cost, const State& x, const Control& u )
{
  Eigen::VectorXd grad    = Eigen::VectorXd::Zero( u.size() );
  const double    epsilon = 1e-6;
  for( int i = 0; i < u.size(); ++i )
  {
    Control du     = Control::Zero( u.size() );
    du( i )        = epsilon;
    double f_plus  = stage_cost( x, u + du );
    double f_minus = stage_cost( x, u - du );
    grad( i )      = ( f_plus - f_minus ) / ( 2 * epsilon );
  }
  return grad;
}

//------------------------------------------------------------------------------
// Helper function: safe_eval
// Calls the stage_cost function and checks if the returned value is finite.
// If not, it prints a warning and returns 0.0 as a fallback.
//------------------------------------------------------------------------------
inline double
safe_eval( const StageCostFunction& stage_cost, const State& x, const Control& u )
{
  double value = stage_cost( x, u );
  if( !std::isfinite( value ) )
  {
    return 0.0;
  }
  return value;
}

//------------------------------------------------------------------------------
// Compute cost state Hessian (second derivative wrt state)
//------------------------------------------------------------------------------
inline Eigen::MatrixXd
compute_cost_state_hessian( const StageCostFunction& stage_cost, const State& x, const Control& u )
{
  const int       n       = x.size();
  Eigen::MatrixXd H       = Eigen::MatrixXd::Zero( n, n );
  const double    epsilon = 1e-5; // Slightly larger for second derivatives

  // Diagonal entries using the standard second-order central difference:
  for( int i = 0; i < n; ++i )
  {
    State dx       = State::Zero( n );
    dx( i )        = epsilon;
    double f_plus  = safe_eval( stage_cost, x + dx, u );
    double f       = safe_eval( stage_cost, x, u );
    double f_minus = safe_eval( stage_cost, x - dx, u );
    // If any value is not finite (shouldn't happen because of safe_eval), we get 0.
    H( i, i ) = ( f_plus - 2 * f + f_minus ) / ( epsilon * epsilon );
  }

  // Off-diagonal entries using a four-point formula:
  for( int i = 0; i < n; ++i )
  {
    for( int j = 0; j < n; ++j )
    {
      if( i != j )
      {
        State dx_i  = State::Zero( n );
        State dx_j  = State::Zero( n );
        dx_i( i )   = epsilon;
        dx_j( j )   = epsilon;
        double f_pp = safe_eval( stage_cost, x + dx_i + dx_j, u );
        double f_pm = safe_eval( stage_cost, x + dx_i - dx_j, u );
        double f_mp = safe_eval( stage_cost, x - dx_i + dx_j, u );
        double f_mm = safe_eval( stage_cost, x - dx_i - dx_j, u );
        H( i, j )   = ( f_pp - f_pm - f_mp + f_mm ) / ( 4 * epsilon * epsilon );
      }
    }
  }

  return H;
}

//------------------------------------------------------------------------------
// Compute cost control Hessian (second derivative wrt control)
//------------------------------------------------------------------------------
inline Eigen::MatrixXd
compute_cost_control_hessian( const StageCostFunction& stage_cost, const State& x, const Control& u )
{
  const int       m       = u.size();
  Eigen::MatrixXd H       = Eigen::MatrixXd::Zero( m, m );
  const double    epsilon = 1e-5;

  // Diagonal entries using the standard second-order central difference:
  for( int i = 0; i < m; ++i )
  {
    Control du     = Control::Zero( m );
    du( i )        = epsilon;
    double f_plus  = safe_eval( stage_cost, x, u + du );
    double f       = safe_eval( stage_cost, x, u );
    double f_minus = safe_eval( stage_cost, x, u - du );
    H( i, i )      = ( f_plus - 2 * f + f_minus ) / ( epsilon * epsilon );
  }

  // Off-diagonal entries using a four-point formula:
  for( int i = 0; i < m; ++i )
  {
    for( int j = 0; j < m; ++j )
    {
      if( i != j )
      {
        Control du_i = Control::Zero( m );
        Control du_j = Control::Zero( m );
        du_i( i )    = epsilon;
        du_j( j )    = epsilon;
        double f_pp  = safe_eval( stage_cost, x, u + du_i + du_j );
        double f_pm  = safe_eval( stage_cost, x, u + du_i - du_j );
        double f_mp  = safe_eval( stage_cost, x, u - du_i + du_j );
        double f_mm  = safe_eval( stage_cost, x, u - du_i - du_j );
        H( i, j )    = ( f_pp - f_pm - f_mp + f_mm ) / ( 4 * epsilon * epsilon );
      }
    }
  }

  return H;
}

//------------------------------------------------------------------------------
// Compute cost cross term Hessian (mixed derivative: control vs. state)
//------------------------------------------------------------------------------
inline Eigen::MatrixXd
compute_cost_cross_term( const StageCostFunction& stage_cost, const State& x, const Control& u )
{
  const int       m       = u.size();
  const int       n       = x.size();
  Eigen::MatrixXd H       = Eigen::MatrixXd::Zero( m, n );
  const double    epsilon = 1e-6; // Use 1e-6; adjust if needed

  // Use a four-point formula for mixed derivatives.
  for( int i = 0; i < m; ++i )
  {
    for( int j = 0; j < n; ++j )
    {
      Control du  = Control::Zero( m );
      State   dx  = State::Zero( n );
      du( i )     = epsilon;
      dx( j )     = epsilon;
      double f_pp = safe_eval( stage_cost, x + dx, u + du );
      double f_pm = safe_eval( stage_cost, x - dx, u + du );
      double f_mp = safe_eval( stage_cost, x + dx, u - du );
      double f_mm = safe_eval( stage_cost, x - dx, u - du );
      H( i, j )   = ( f_pp - f_pm - f_mp + f_mm ) / ( 4 * epsilon * epsilon );
    }
  }

  return H;
}
