#pragma once

#include <cmath>

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
 */
namespace mas
{

template<typename Scalar>
inline StateDerivativeT<Scalar>
single_track_model( const StateT<Scalar>& x, const ControlT<Scalar>& u )
{
  const Scalar psi = x( 2 );
  const Scalar v   = x( 3 );

  // Unpack controls
  const Scalar delta = u( 0 );
  const Scalar a     = u( 1 );

  // Vehicle parameters
  const Scalar L = static_cast<Scalar>( 2.5 ); // Wheelbase length [m]

  // Compute derivatives
  StateDerivativeT<Scalar> dxdt( 4 );
  dxdt( 0 ) = v * std::cos( psi );       // X_dot
  dxdt( 1 ) = v * std::sin( psi );       // Y_dot
  dxdt( 2 ) = v * std::tan( delta ) / L; // Psi_dot
  dxdt( 3 ) = a;                         // v_dot

  return dxdt;
}

inline StateDerivative
single_track_model( const State& x, const Control& u )
{
  return single_track_model<double>( x, u );
}

/**
 * @brief Compute the state Jacobian A = ∂f/∂x for the single-track model.
 */
template<typename Scalar>
inline Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
single_track_state_jacobian( const StateT<Scalar>& x, const ControlT<Scalar>& u )
{
  const Scalar psi   = x( 2 );
  const Scalar v     = x( 3 );
  const Scalar delta = u( 0 );
  const Scalar L     = static_cast<Scalar>( 2.5 ); // Wheelbase

  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> A
    = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Zero( 4, 4 );
  A( 0, 2 ) = -v * std::sin( psi );  // ∂X_dot / ∂psi
  A( 0, 3 ) = std::cos( psi );       // ∂X_dot / ∂v
  A( 1, 2 ) = v * std::cos( psi );   // ∂Y_dot / ∂psi
  A( 1, 3 ) = std::sin( psi );       // ∂Y_dot / ∂v
  A( 2, 3 ) = std::tan( delta ) / L; // ∂psi_dot / ∂v

  return A;
}

inline Eigen::MatrixXd
single_track_state_jacobian( const State& x, const Control& u )
{
  return single_track_state_jacobian<double>( x, u );
}

/**
 * @brief Compute the control Jacobian B = ∂f/∂u for the single-track model.
 */
template<typename Scalar>
inline Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
single_track_control_jacobian( const StateT<Scalar>& x, const ControlT<Scalar>& u )
{
  const Scalar v     = x( 3 );
  const Scalar delta = u( 0 );
  const Scalar L     = static_cast<Scalar>( 2.5 );

  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> B
    = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Zero( 4, 2 );
  const Scalar cos_delta = std::cos( delta );
  B( 2, 0 )              = v / ( L * cos_delta * cos_delta ); // ∂psi_dot / ∂delta
  B( 3, 1 )              = static_cast<Scalar>( 1.0 );        // ∂v_dot / ∂a

  return B;
}

inline Eigen::MatrixXd
single_track_control_jacobian( const State& x, const Control& u )
{
  return single_track_control_jacobian<double>( x, u );
}

} // namespace mas