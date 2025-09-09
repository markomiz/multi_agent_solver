#pragma once
#include <Eigen/Dense>

#include "multi_agent_solver/types.hpp"

/**
 * @brief Single-track kinematic bicycle model **with time as a state**.
 *
 * States (x):
 *   x(0) = X   [m]      - Global X position
 *   x(1) = Y   [m]      - Global Y position
 *   x(2) = psi [rad]    - Heading angle
 *   x(3) = v   [m/s]    - Velocity
 *
 * Controls (u):
 *   u(0) = delta [rad]  - Steering angle
 *   u(1) = a     [m/s²] - Acceleration
 *
 *
 */
namespace mas
{
inline StateDerivative
single_track_model( const State& x, const Control& u )
{
  double psi = x( 2 );
  double v   = x( 3 );

  // Unpack controls
  double delta = u( 0 );
  double a     = u( 1 );

  // Vehicle parameters
  const double L = 2.5; // Wheelbase length [m]

  // Compute derivatives
  StateDerivative dxdt( 4 );
  dxdt( 0 ) = v * std::cos( psi );       // X_dot
  dxdt( 1 ) = v * std::sin( psi );       // Y_dot
  dxdt( 2 ) = v * std::tan( delta ) / L; // Psi_dot
  dxdt( 3 ) = a;                         // v_dot

  return dxdt;
}

/**
 * @brief Compute the state Jacobian A = ∂f/∂x for the single-track model.
 */
inline Eigen::MatrixXd
single_track_state_jacobian( const State& x, const Control& u )
{
  double       psi   = x( 2 );
  double       v     = x( 3 );
  double       delta = u( 0 );
  const double L     = 2.5; // Wheelbase

  Eigen::MatrixXd A = Eigen::MatrixXd::Zero( 4, 4 );
  A( 0, 2 )         = -v * std::sin( psi );  // ∂X_dot / ∂psi
  A( 0, 3 )         = std::cos( psi );       // ∂X_dot / ∂v
  A( 1, 2 )         = v * std::cos( psi );   // ∂Y_dot / ∂psi
  A( 1, 3 )         = std::sin( psi );       // ∂Y_dot / ∂v
  A( 2, 3 )         = std::tan( delta ) / L; // ∂psi_dot / ∂v

  return A;
}

/**
 * @brief Compute the control Jacobian B = ∂f/∂u for the single-track model.
 */
inline Eigen::MatrixXd
single_track_control_jacobian( const State& x, const Control& u )
{
  double       v     = x( 3 );
  double       delta = u( 0 );
  const double L     = 2.5;

  Eigen::MatrixXd B = Eigen::MatrixXd::Zero( 4, 2 );
  B( 2, 0 )         = v / ( L * std::cos( delta ) * std::cos( delta ) ); // ∂psi_dot / ∂delta
  B( 3, 1 )         = 1.0;                                               // ∂v_dot / ∂a

  return B;
}
} // namespace mas