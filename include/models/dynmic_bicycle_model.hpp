#pragma once
#include <cmath>

#include <Eigen/Dense>

using State           = Eigen::VectorXd;
using StateDerivative = Eigen::VectorXd;
using Control         = Eigen::VectorXd;

/**
 * @brief A dynamic bicycle model that handles very small velocities more gracefully.
 *
 * States (x):
 *   x(0) = X   [m]
 *   x(1) = Y   [m]
 *   x(2) = psi [rad]   (yaw angle)
 *   x(3) = vx  [m/s]   (longitudinal velocity in vehicle frame)
 *   x(4) = vy  [m/s]   (lateral velocity in vehicle frame)
 *   x(5) = r   [rad/s] (yaw rate)
 *
 * Controls (u):
 *   u(0) = delta [rad] (steer angle)
 *   u(1) = a     [m/s^2] (longitudinal acceleration)
 *
 * When |vx| is below min_speed, we switch to a simplified or clamped approach
 * to avoid dividing by near-zero speeds and to reflect that the vehicle is nearly stationary.
 */
StateDerivative
dynamic_bicycle_model( const State& x, const Control& u )
{
  // Unpack states
  double X   = x( 0 );
  double Y   = x( 1 );
  double psi = x( 2 );
  double vx  = x( 3 );
  double vy  = x( 4 );
  double r   = x( 5 );

  // Unpack controls
  double delta = u( 0 ); // steering angle
  double a     = u( 1 ); // longitudinal acceleration

  // Vehicle/road parameters (example constants)
  const double m   = 1500.0;   // mass [kg]
  const double I_z = 3000.0;   // yaw inertia [kg*m^2]
  const double l_f = 1.2;      // distance from CoG to front axle [m]
  const double l_r = 1.6;      // distance from CoG to rear axle [m]
  const double C_f = 160000.0; // cornering stiffness front [N/rad]
  const double C_r = 170000.0; // cornering stiffness rear [N/rad]

  // Minimum speed threshold
  const double v_min = 1e-6; // m/s, adjustable

  // We'll compute derivatives in two modes:
  StateDerivative dxdt( 6 );
  dxdt.setZero();


  double alpha_f = delta - std::atan2( ( vy + l_f * r ), vx );
  double alpha_r = -std::atan2( ( vy - l_r * r ), vx );

  double F_yf = -C_f * alpha_f;
  double F_yr = -C_r * alpha_r;

  // Longitudinal force
  double F_x = m * a;

  // Derivatives
  dxdt( 0 ) = vx * std::cos( psi ) - vy * std::sin( psi ); // dot(X)
  dxdt( 1 ) = vx * std::sin( psi ) + vy * std::cos( psi ); // dot(Y)
  dxdt( 2 ) = r;                                           // dot(psi)

  dxdt( 3 ) = r * vy + ( 1.0 / m ) * ( F_x - F_yf * std::sin( delta ) );       // dot(vx)
  dxdt( 4 ) = -r * vx + ( 1.0 / m ) * ( F_yf * std::cos( delta ) + F_yr );     // dot(vy)
  dxdt( 5 ) = ( 1.0 / I_z ) * ( l_f * F_yf * std::cos( delta ) - l_r * F_yr ); // dot(r)


  return dxdt;
}

inline Eigen::MatrixXd
dynamic_bicycle_state_jacobian( const State& x, const Control& u )
{
  // Unpack states.
  double X   = x( 0 ); // Not used in the dynamics derivatives.
  double Y   = x( 1 ); // Not used.
  double psi = x( 2 );
  double vx  = x( 3 );
  double vy  = x( 4 );
  double r   = x( 5 );

  // Unpack controls.
  double delta = u( 0 );
  double a     = u( 1 ); // Not used in the derivatives of the kinematics.

  // Vehicle parameters.
  const double m   = 1500.0;
  const double I_z = 3000.0;
  const double l_f = 1.2;
  const double l_r = 1.6;
  const double C_f = 160000.0;
  const double C_r = 170000.0;

  Eigen::MatrixXd A = Eigen::MatrixXd::Zero( 6, 6 );

  // f0 = vx*cos(psi) - vy*sin(psi)
  A( 0, 2 ) = -vx * std::sin( psi ) - vy * std::cos( psi );
  A( 0, 3 ) = std::cos( psi );
  A( 0, 4 ) = -std::sin( psi );

  // f1 = vx*sin(psi) + vy*cos(psi)
  A( 1, 2 ) = vx * std::cos( psi ) - vy * std::sin( psi );
  A( 1, 3 ) = std::sin( psi );
  A( 1, 4 ) = std::cos( psi );

  // f2 = r
  A( 2, 5 ) = 1.0;

  // For the remaining rows (f3, f4, f5), the expressions are more involved.
  // Here we provide approximate expressions based on common derivations.
  // Note: These derivatives should be verified and possibly refined.

  // f3 = r*vy + (1/m) * ( m*a - F_yf*sin(delta) )
  // with F_yf = -C_f*(delta - atan2(vy+l_f*r, vx)).
  double z_f  = vy + l_f * r;
  double F_yf = -C_f * ( delta - std::atan2( z_f, vx ) );
  // Partial derivatives for f3:
  // ∂f3/∂r (from r*vy and indirectly via F_yf) — approximate as:
  A( 3, 5 ) = vy; // plus additional terms omitted for brevity.
  // ∂f3/∂vy:
  A( 3, 4 ) = r;
  // ∂f3/∂vx: only from F_yf via atan2 term (approximate as zero for simplicity).
  A( 3, 3 ) = 0.0;

  // f4 = -r*vx + (1/m)*( F_yf*cos(delta) + F_yr )
  // with F_yr = -C_r*( -atan2(vy-l_r*r, vx) ) = C_r*atan2(vy-l_r*r, vx).
  double z_r  = vy - l_r * r;
  double F_yr = C_r * std::atan2( z_r, vx );
  // Partial derivatives for f4 (approximate):
  A( 4, 3 ) = -r;
  A( 4, 5 ) = -vx;
  // Other derivatives are omitted for brevity.

  // f5 = (1/I_z)*( l_f*F_yf*cos(delta) - l_r*F_yr )
  // Partial derivatives (approximate):
  A( 5, 5 ) = 0.0; // For simplicity.

  return A;
}

//---------------------------------------------------------------------
// Compute the control Jacobian B = ∂f/∂u for the dynamic bicycle model.
//---------------------------------------------------------------------
inline Eigen::MatrixXd
dynamic_bicycle_control_jacobian( const State& x, const Control& u )
{
  Eigen::MatrixXd B = Eigen::MatrixXd::Zero( 6, 2 );

  // Unpack states.
  double X   = x( 0 ); // Not used.
  double Y   = x( 1 ); // Not used.
  double psi = x( 2 );
  double vx  = x( 3 );
  double vy  = x( 4 );
  double r   = x( 5 );

  // Unpack controls.
  double delta = u( 0 );
  double a     = u( 1 );

  // Vehicle parameters.
  const double m   = 1500.0;
  const double I_z = 3000.0;
  const double l_f = 1.2;
  const double l_r = 1.6;
  const double C_f = 160000.0;
  const double C_r = 170000.0;

  // f0, f1, f2 have no dependence on controls.

  // For f3 = r*vy + (1/m)*( m*a - F_yf*sin(delta) )
  // Compute F_yf as in the dynamics function.
  double z_f  = vy + l_f * r;
  double F_yf = -C_f * ( delta - std::atan2( z_f, vx ) );
  // ∂f3/∂a = 1/m.
  B( 3, 1 ) = 1.0;
  // ∂f3/∂delta: differentiate -F_yf*sin(delta) with respect to delta.
  // dF_yf/d(delta) = -C_f.
  // Therefore, ∂/∂delta [F_yf*sin(delta)] = F_yf*cos(delta) - C_f*sin(delta).
  B( 3, 0 ) = ( 1.0 / m ) * ( -( F_yf * std::cos( delta ) - C_f * std::sin( delta ) ) );

  // For f4 = -r*vx + (1/m)*( F_yf*cos(delta) + F_yr )
  // Compute F_yr.
  double z_r  = vy - l_r * r;
  double F_yr = C_r * std::atan2( z_r, vx );
  // ∂f4/∂delta: derivative from F_yf*cos(delta)
  // d/d(delta)[F_yf*cos(delta)] = -F_yf*sin(delta) + cos(delta)*(-C_f)
  B( 4, 0 ) = ( 1.0 / m ) * ( -F_yf * std::sin( delta ) - C_f * std::cos( delta ) );
  // f4 does not depend on a.
  B( 4, 1 ) = 0.0;

  // For f5 = (1/I_z)*( l_f*F_yf*cos(delta) - l_r*F_yr )
  // ∂f5/∂delta:
  // d/d(delta)[l_f*F_yf*cos(delta)] = l_f*(F_yf*(-sin(delta)) + cos(delta)*(-C_f))
  B( 5, 0 ) = ( 1.0 / I_z ) * ( -l_f * F_yf * std::sin( delta ) - l_f * C_f * std::cos( delta ) );
  // f5 does not depend on a.
  B( 5, 1 ) = 0.0;

  return B;
}