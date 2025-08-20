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
  Scalar psi = x( 2 );
  Scalar v   = x( 3 );

  // Unpack controls
  Scalar delta = u( 0 );
  Scalar a     = u( 1 );

  // Vehicle parameters
  const double L = 2.5; // Wheelbase length [m]

  // Compute derivatives
  StateDerivative dxdt( 4 );
#ifdef MAS_USE_AUTODIFF
  dxdt( 0 ) = v * autodiff::detail::cos( psi );       // X_dot
  dxdt( 1 ) = v * autodiff::detail::sin( psi );       // Y_dot
  dxdt( 2 ) = v * autodiff::detail::tan( delta ) / L; // Psi_dot
#else
  dxdt( 0 ) = v * std::cos( psi );       // X_dot
  dxdt( 1 ) = v * std::sin( psi );       // Y_dot
  dxdt( 2 ) = v * std::tan( delta ) / L; // Psi_dot
#endif
  dxdt( 3 ) = a;                         // v_dot

  return dxdt;
}

/**
 * @brief Compute the state Jacobian A = ∂f/∂x for the single-track model.
 */
inline Eigen::MatrixXd
single_track_state_jacobian( const State& x, const Control& u )
{
  double       psi   = to_double( x( 2 ) );
  double       v     = to_double( x( 3 ) );
  double       delta = to_double( u( 0 ) );
  const double L     = 2.5; // Wheelbase

  Eigen::MatrixXd A = Eigen::MatrixXd::Zero( 4, 4 );
  A( 0, 2 )
#ifdef MAS_USE_AUTODIFF
    = -v * autodiff::detail::sin( psi );
  A( 0, 3 ) = autodiff::detail::cos( psi );
  A( 1, 2 ) = v * autodiff::detail::cos( psi );
  A( 1, 3 ) = autodiff::detail::sin( psi );
  A( 2, 3 ) = autodiff::detail::tan( delta ) / L;
#else
    = -v * std::sin( psi );
  A( 0, 3 ) = std::cos( psi );
  A( 1, 2 ) = v * std::cos( psi );
  A( 1, 3 ) = std::sin( psi );
  A( 2, 3 ) = std::tan( delta ) / L;
#endif

  return A;
}

/**
 * @brief Compute the control Jacobian B = ∂f/∂u for the single-track model.
 */
inline Eigen::MatrixXd
single_track_control_jacobian( const State& x, const Control& u )
{
  double       v     = to_double( x( 3 ) );
  double       delta = to_double( u( 0 ) );
  const double L     = 2.5;

  Eigen::MatrixXd B = Eigen::MatrixXd::Zero( 4, 2 );
  B( 2, 0 )
#ifdef MAS_USE_AUTODIFF
    = v / ( L * autodiff::detail::cos( delta ) * autodiff::detail::cos( delta ) );
#else
    = v / ( L * std::cos( delta ) * std::cos( delta ) );
#endif // MAS_USE_AUTODIFF
  B( 3, 1 )         = 1.0;                                               // ∂v_dot / ∂a

  return B;
}
} // namespace mas
