#pragma once
#include <Eigen/Dense>

#include "multi_agent_solver/types.hpp"

namespace mas
{
inline StateDerivative
pendulum_dynamics( const State& x, const Control& u )
{
  const double    g = 9.81; // [m/s^2]
  const double    l = 1.0;  // [m]
  const double    m = 1.0;  // [kg]
  StateDerivative dxdt( 2 );
  dxdt( 0 ) = x( 1 );
  dxdt( 1 ) = ( g / l ) * std::sin( x( 0 ) ) + u( 0 ) / ( m * l * l );
  return dxdt;
}

inline Eigen::MatrixXd
pendulum_state_jacobian( const State& x, const Control& )
{
  const double    g = 9.81;
  const double    l = 1.0;
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero( 2, 2 );
  A( 0, 1 )         = 1.0;
  A( 1, 0 )         = ( g / l ) * std::cos( x( 0 ) );
  return A;
}

inline Eigen::MatrixXd
pendulum_control_jacobian( const State&, const Control& )
{
  const double    m = 1.0;
  const double    l = 1.0;
  Eigen::MatrixXd B = Eigen::MatrixXd::Zero( 2, 1 );
  B( 1, 0 )         = 1.0 / ( m * l * l );
  return B;
}
} // namespace mas