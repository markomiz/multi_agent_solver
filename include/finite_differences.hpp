#pragma once

#include "integrator.hpp"
#include "types.hpp"

// Finite Differences GradientComputer
inline ControlGradient
finite_differences_gradient( const State& initial_state, const ControlTrajectory& controls, const MotionModel& dynamics,
                             const ObjectiveFunction& objective_function, double dt )
{
  bool            forward_only = false; // TODO take out of function
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

      // Create perturbed control trajectories
      controls_plus                    = controls;
      controls_plus( i, t )           += epsilon;
      StateTrajectory trajectory_plus  = integrate_horizon( initial_state, controls_plus, dt, dynamics, integrate_rk4 );
      double          cost_plus        = objective_function( trajectory_plus, controls_plus );

      if( !forward_only )
      {
        controls_minus          = controls;
        controls_minus( i, t ) -= epsilon;
        trajectory_minus        = integrate_horizon( initial_state, controls_minus, dt, dynamics, integrate_rk4 );
        cost_minus              = objective_function( trajectory_minus, controls_minus );
      }

      // Compute central difference gradient
      gradients( i, t ) = ( cost_plus - cost_minus ) / ( forward_only ? epsilon : 2 * epsilon );
    }
  }

  return gradients;
}

// Compute dynamics state Jacobian using finite differences or analytical methods
inline Eigen::MatrixXd
compute_dynamics_state_jacobian( const MotionModel& dynamics, const State& x, const Control& u )
{
  Eigen::MatrixXd A;
  const int       state_dim = x.size();
  const double    epsilon   = 1e-6;
  A                         = Eigen::MatrixXd::Zero( state_dim, state_dim );

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

// Compute dynamics control Jacobian using finite differences or analytical methods
inline Eigen::MatrixXd
compute_dynamics_control_jacobian( const MotionModel& dynamics, const State& x, const Control& u )
{
  Eigen::MatrixXd B;
  const int       state_dim   = x.size();
  const int       control_dim = u.size();
  const double    epsilon     = 1e-6;
  B                           = Eigen::MatrixXd::Zero( state_dim, control_dim );

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

// Compute cost state gradient using finite differences or analytical methods
inline Eigen::VectorXd
compute_cost_state_gradient( const ObjectiveFunction& objective_function, const State& x, const Control& u )
{
  Eigen::VectorXd l_x;

  const int    state_dim = x.size();
  const double epsilon   = 1e-6;
  l_x                    = Eigen::VectorXd::Zero( state_dim );

  for( int i = 0; i < state_dim; ++i )
  {
    State dx = State::Zero( state_dim );
    dx( i )  = epsilon;

    double cost_plus  = objective_function( x + dx, u );
    double cost_minus = objective_function( x - dx, u );

    l_x( i ) = ( cost_plus - cost_minus ) / ( 2 * epsilon );
  }
  return l_x;
}

// Compute cost control gradient using finite differences or analytical methods
inline Eigen::VectorXd
compute_cost_control_gradient( const ObjectiveFunction& objective_function, const State& x, const Control& u )
{
  Eigen::VectorXd l_u;
  const int       control_dim = u.size();
  const double    epsilon     = 1e-6;
  l_u                         = Eigen::VectorXd::Zero( control_dim );

  for( int i = 0; i < control_dim; ++i )
  {
    Control du = Control::Zero( control_dim );
    du( i )    = epsilon;

    double cost_plus  = objective_function( x, u + du );
    double cost_minus = objective_function( x, u - du );

    l_u( i ) = ( cost_plus - cost_minus ) / ( 2 * epsilon );
  }
  return l_u;
}

// Compute cost state Hessian using finite differences or analytical methods
inline Eigen::MatrixXd
compute_cost_state_hessian( const ObjectiveFunction& objective_function, const State& x, const Control& u )
{
  Eigen::MatrixXd l_xx;

  const int    state_dim = x.size();
  const double epsilon   = 1e-6;
  l_xx                   = Eigen::MatrixXd::Zero( state_dim, state_dim );

  for( int i = 0; i < state_dim; ++i )
  {
    for( int j = 0; j < state_dim; ++j )
    {
      State dx1 = State::Zero( state_dim );
      State dx2 = State::Zero( state_dim );
      dx1( i )  = epsilon;
      dx2( j )  = epsilon;

      double cost_plus  = objective_function( x + dx1 + dx2, u );
      double cost_minus = objective_function( x + dx1 - dx2, u );

      l_xx( i, j ) = ( cost_plus - cost_minus ) / ( 4 * epsilon * epsilon );
    }
  }
  return l_xx;
}

// Compute cost control Hessian using finite differences or analytical methods
inline Eigen::MatrixXd
compute_cost_control_hessian( const ObjectiveFunction& objective_function, const State& x, const Control& u )
{
  Eigen::MatrixXd l_uu;
  const int       control_dim = u.size();
  const double    epsilon     = 1e-6;
  l_uu                        = Eigen::MatrixXd::Zero( control_dim, control_dim );

  for( int i = 0; i < control_dim; ++i )
  {
    for( int j = 0; j < control_dim; ++j )
    {
      Control du1 = Control::Zero( control_dim );
      Control du2 = Control::Zero( control_dim );
      du1( i )    = epsilon;
      du2( j )    = epsilon;

      double cost_plus  = objective_function( x, u + du1 + du2 );
      double cost_minus = objective_function( x, u + du1 - du2 );

      l_uu( i, j ) = ( cost_plus - cost_minus ) / ( 4 * epsilon * epsilon );
    }
  }
  return l_uu;
}

// Compute cost cross term Hessian using finite differences or analytical methods
inline Eigen::MatrixXd
compute_cost_cross_term( const ObjectiveFunction& objective_function, const State& x, const Control& u )
{
  Eigen::MatrixXd l_ux;
  const int       state_dim   = x.size();
  const int       control_dim = u.size();
  const double    epsilon     = 1e-6;
  l_ux                        = Eigen::MatrixXd::Zero( control_dim, state_dim );

  for( int i = 0; i < control_dim; ++i )
  {
    for( int j = 0; j < state_dim; ++j )
    {
      Control du = Control::Zero( control_dim );
      State   dx = State::Zero( state_dim );
      du( i )    = epsilon;
      dx( j )    = epsilon;

      double cost_plus  = objective_function( x + dx, u + du );
      double cost_minus = objective_function( x - dx, u + du );

      l_ux( i, j ) = ( cost_plus - cost_minus ) / ( 4 * epsilon * epsilon );
    }
  }
  return l_ux;
}