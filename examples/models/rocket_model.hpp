#pragma once

#include <algorithm>

#include <Eigen/Dense>

#include "multi_agent_solver/types.hpp"

namespace mas
{

template<typename Scalar = double>
struct RocketParametersT
{
  Scalar initial_mass     = static_cast<Scalar>( 1.0 );  ///< Initial vehicle mass [kg]
  Scalar gravity          = static_cast<Scalar>( 9.81 ); ///< Gravity [m/s^2]
  Scalar exhaust_velocity = static_cast<Scalar>( 25.0 ); ///< Effective exhaust velocity [m/s]
};

using RocketParameters  = RocketParametersT<double>;
using RocketParametersf = RocketParametersT<float>;

template<typename Scalar>
inline StateT<Scalar>
rocket_dynamics( const RocketParametersT<Scalar>& params, const StateT<Scalar>& state, const ControlT<Scalar>& control )
{
  StateT<Scalar> derivative = StateT<Scalar>::Zero( 3 );

  const Scalar mass   = std::max( state( 2 ), static_cast<Scalar>( 1e-6 ) );
  const Scalar thrust = mass > static_cast<Scalar>( 0 ) ? control( 0 ) : static_cast<Scalar>( 0 );

  derivative( 0 ) = state( 1 );
  derivative( 1 ) = thrust / mass - params.gravity;
  derivative( 2 ) = -thrust / params.exhaust_velocity;

  return derivative;
}

inline mas::State
rocket_dynamics( const RocketParameters& params, const mas::State& state, const mas::Control& control )
{
  return rocket_dynamics<double>( params, state, control );
}

template<typename Scalar>
inline MotionModelT<Scalar>
make_rocket_dynamics( const RocketParametersT<Scalar>& params )
{
  return [params]( const StateT<Scalar>& state, const ControlT<Scalar>& control ) {
    return rocket_dynamics<Scalar>( params, state, control );
  };
}

inline mas::MotionModel
make_rocket_dynamics( const RocketParameters& params )
{
  return make_rocket_dynamics<double>( params );
}

template<typename Scalar>
inline Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
rocket_state_jacobian( const RocketParametersT<Scalar>&, const StateT<Scalar>& state, const ControlT<Scalar>& control )
{
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> A
    = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Zero( 3, 3 );
  A( 0, 1 ) = static_cast<Scalar>( 1.0 );

  const Scalar thrust = control( 0 );
  const Scalar mass   = std::max( state( 2 ), static_cast<Scalar>( 1e-6 ) );

  A( 1, 2 ) = -thrust / ( mass * mass );

  return A;
}

inline Eigen::MatrixXd
rocket_state_jacobian( const RocketParameters& params, const mas::State& state, const mas::Control& control )
{
  return rocket_state_jacobian<double>( params, state, control );
}

template<typename Scalar>
inline Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
rocket_control_jacobian( const RocketParametersT<Scalar>& params, const StateT<Scalar>& state, const ControlT<Scalar>& )
{
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> B
    = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Zero( 3, 1 );
  const Scalar mass = std::max( state( 2 ), static_cast<Scalar>( 1e-6 ) );

  B( 1, 0 ) = static_cast<Scalar>( 1.0 ) / mass;
  B( 2, 0 ) = -static_cast<Scalar>( 1.0 ) / params.exhaust_velocity;
  return B;
}

inline Eigen::MatrixXd
rocket_control_jacobian( const RocketParameters& params, const mas::State& state, const mas::Control& control )
{
  return rocket_control_jacobian<double>( params, state, control );
}

} // namespace mas