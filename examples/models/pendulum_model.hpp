#pragma once

#include <cmath>

#include <Eigen/Dense>

#include "multi_agent_solver/types.hpp"

namespace mas
{

template<typename Scalar>
inline StateDerivativeT<Scalar>
pendulum_dynamics( const StateT<Scalar>& x, const ControlT<Scalar>& u )
{
  const Scalar          g    = static_cast<Scalar>( 9.81 ); // [m/s^2]
  const Scalar          l    = static_cast<Scalar>( 1.0 );  // [m]
  const Scalar          m    = static_cast<Scalar>( 1.0 );  // [kg]
  StateDerivativeT<Scalar> dxdt( 2 );
  dxdt( 0 ) = x( 1 );
  dxdt( 1 ) = ( g / l ) * std::sin( x( 0 ) ) + u( 0 ) / ( m * l * l );
  return dxdt;
}

inline StateDerivative
pendulum_dynamics( const State& x, const Control& u )
{
  return pendulum_dynamics<double>( x, u );
}

template<typename Scalar>
inline Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
pendulum_state_jacobian( const StateT<Scalar>& x, const ControlT<Scalar>& )
{
  const Scalar g = static_cast<Scalar>( 9.81 );
  const Scalar l = static_cast<Scalar>( 1.0 );
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> A = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Zero( 2, 2 );
  A( 0, 1 ) = static_cast<Scalar>( 1.0 );
  A( 1, 0 ) = ( g / l ) * std::cos( x( 0 ) );
  return A;
}

inline Eigen::MatrixXd
pendulum_state_jacobian( const State& x, const Control& u )
{
  return pendulum_state_jacobian<double>( x, u );
}

template<typename Scalar>
inline Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
pendulum_control_jacobian( const StateT<Scalar>&, const ControlT<Scalar>& )
{
  const Scalar m = static_cast<Scalar>( 1.0 );
  const Scalar l = static_cast<Scalar>( 1.0 );
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> B = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Zero( 2, 1 );
  B( 1, 0 ) = static_cast<Scalar>( 1.0 ) / ( m * l * l );
  return B;
}

inline Eigen::MatrixXd
pendulum_control_jacobian( const State& x, const Control& u )
{
  return pendulum_control_jacobian<double>( x, u );
}

} // namespace mas