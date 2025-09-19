#pragma once

#include <Eigen/Dense>

#include "multi_agent_solver/types.hpp"

namespace mas::examples
{

struct RocketParameters
{
  double mass     = 1.0;  ///< Vehicle mass [kg]
  double gravity  = 9.81; ///< Gravity [m/s^2]
};

inline mas::State
rocket_dynamics( const RocketParameters& params, const mas::State& state, const mas::Control& control )
{
  mas::State derivative = mas::State::Zero( 2 );
  derivative( 0 )       = state( 1 );
  derivative( 1 )       = control( 0 ) / params.mass - params.gravity;
  return derivative;
}

inline mas::MotionModel
make_rocket_dynamics( const RocketParameters& params )
{
  return [params]( const mas::State& state, const mas::Control& control ) {
    return rocket_dynamics( params, state, control );
  };
}

inline Eigen::MatrixXd
rocket_state_jacobian( const RocketParameters&, const mas::State&, const mas::Control& )
{
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero( 2, 2 );
  A( 0, 1 )         = 1.0;
  return A;
}

inline Eigen::MatrixXd
rocket_control_jacobian( const RocketParameters& params, const mas::State&, const mas::Control& )
{
  Eigen::MatrixXd B = Eigen::MatrixXd::Zero( 2, 1 );
  B( 1, 0 )         = 1.0 / params.mass;
  return B;
}

} // namespace mas::examples
