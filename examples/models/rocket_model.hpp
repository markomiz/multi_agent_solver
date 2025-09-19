#pragma once

#include <algorithm>

#include <Eigen/Dense>

#include "multi_agent_solver/types.hpp"

namespace mas
{

struct RocketParameters
{
  double initial_mass     = 1.0;  ///< Initial vehicle mass [kg]
  double gravity          = 9.81; ///< Gravity [m/s^2]
  double exhaust_velocity = 25.0; ///< Effective exhaust velocity [m/s]
};

inline mas::State
rocket_dynamics( const RocketParameters& params, const mas::State& state, const mas::Control& control )
{
  mas::State derivative = mas::State::Zero( 3 );

  const double mass   = std::max( state( 2 ), 1e-6 );
  const double thrust = control( 0 );

  derivative( 0 ) = state( 1 );
  derivative( 1 ) = thrust / mass - params.gravity;
  derivative( 2 ) = -thrust / params.exhaust_velocity;

  return derivative;
}

inline mas::MotionModel
make_rocket_dynamics( const RocketParameters& params )
{
  return [params]( const mas::State& state, const mas::Control& control ) { return rocket_dynamics( params, state, control ); };
}

inline Eigen::MatrixXd
rocket_state_jacobian( const RocketParameters&, const mas::State& state, const mas::Control& control )
{
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero( 3, 3 );
  A( 0, 1 )         = 1.0;

  const double thrust = control( 0 );
  const double mass   = std::max( state( 2 ), 1e-6 );

  A( 1, 2 ) = -thrust / ( mass * mass );

  return A;
}

inline Eigen::MatrixXd
rocket_control_jacobian( const RocketParameters& params, const mas::State& state, const mas::Control& )
{
  Eigen::MatrixXd B    = Eigen::MatrixXd::Zero( 3, 1 );
  const double    mass = std::max( state( 2 ), 1e-6 );

  B( 1, 0 ) = 1.0 / mass;
  B( 2, 0 ) = -1.0 / params.exhaust_velocity;
  return B;
}

} // namespace mas