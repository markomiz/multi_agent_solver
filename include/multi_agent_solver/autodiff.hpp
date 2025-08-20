#pragma once

#ifdef MAS_USE_AUTODIFF

  #include "multi_agent_solver/integrator.hpp"
  #include "multi_agent_solver/types.hpp"
  #include <autodiff/forward/dual.hpp>
  #include <autodiff/forward/dual/eigen.hpp>

namespace mas
{
inline ControlGradient
autodiff_gradient( const State& initial_state, const ControlTrajectory& controls, const MotionModel& dynamics,
                   const ObjectiveFunction& objective_function, double dt )
{
  using autodiff::dual;
  using autodiff::VectorXdual;
  using MatrixXdual = Eigen::Matrix<dual, Eigen::Dynamic, Eigen::Dynamic>;

  // Flatten controls into a single vector for autodiff
  VectorXdual u = Eigen::Map<const VectorXdual>( controls.data(), controls.size() );

  auto f = [&]( const VectorXdual& uv ) -> dual {
    MatrixXdual     U = Eigen::Map<const MatrixXdual>( uv.data(), controls.rows(), controls.cols() );
    StateTrajectory X = integrate_horizon( initial_state, U, dt, dynamics, integrate_rk4 );
    return objective_function( X, U );
  };

  dual            y;
  Eigen::VectorXd g;
  autodiff::gradient( f, autodiff::wrt( u ), autodiff::at( u ), y, g );

  return Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>( g.data(), controls.rows(), controls.cols() );
}
} // namespace mas

#endif // MAS_USE_AUTODIFF
