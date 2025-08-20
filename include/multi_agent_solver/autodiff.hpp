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

inline Eigen::MatrixXd
autodiff_dynamics_state_jacobian( const MotionModel& dynamics, const State& x, const Control& u )
{
  using autodiff::dual;
  using VectorXdual = Eigen::Matrix<dual, Eigen::Dynamic, 1>;

  VectorXdual x_ad = x;
  VectorXdual u_ad = u;
  VectorXdual y;

  auto f = [&]( const VectorXdual& xv ) { return dynamics( xv, u_ad ); };
  return autodiff::jacobian( f, autodiff::wrt( x_ad ), autodiff::at( x_ad ), y );
}

inline Eigen::MatrixXd
autodiff_dynamics_control_jacobian( const MotionModel& dynamics, const State& x, const Control& u )
{
  using autodiff::dual;
  using VectorXdual = Eigen::Matrix<dual, Eigen::Dynamic, 1>;

  VectorXdual x_ad = x;
  VectorXdual u_ad = u;
  VectorXdual y;

  auto f = [&]( const VectorXdual& uv ) { return dynamics( x_ad, uv ); };
  return autodiff::jacobian( f, autodiff::wrt( u_ad ), autodiff::at( u_ad ), y );
}

inline Eigen::VectorXd
autodiff_cost_state_gradient( const StageCostFunction& stage_cost, const State& x, const Control& u, size_t time_idx )
{
  using autodiff::dual;
  using VectorXdual = Eigen::Matrix<dual, Eigen::Dynamic, 1>;

  VectorXdual x_ad = x;
  VectorXdual u_ad = u;
  dual         y;
  Eigen::VectorXd g;

  auto f = [&]( const VectorXdual& xv ) -> dual { return stage_cost( xv, u_ad, time_idx ); };
  autodiff::gradient( f, autodiff::wrt( x_ad ), autodiff::at( x_ad ), y, g );
  return g;
}

inline Eigen::VectorXd
autodiff_cost_control_gradient( const StageCostFunction& stage_cost, const State& x, const Control& u, size_t time_idx )
{
  using autodiff::dual;
  using VectorXdual = Eigen::Matrix<dual, Eigen::Dynamic, 1>;

  VectorXdual x_ad = x;
  VectorXdual u_ad = u;
  dual         y;
  Eigen::VectorXd g;

  auto f = [&]( const VectorXdual& uv ) -> dual { return stage_cost( x_ad, uv, time_idx ); };
  autodiff::gradient( f, autodiff::wrt( u_ad ), autodiff::at( u_ad ), y, g );
  return g;
}

inline Eigen::MatrixXd
autodiff_cost_state_hessian( const StageCostFunction& stage_cost, const State& x, const Control& u, size_t time_idx )
{
  using autodiff::dual;
  using VectorXdual = Eigen::Matrix<dual, Eigen::Dynamic, 1>;

  VectorXdual x_ad = x;

  auto grad_x = [&]( VectorXdual xv ) -> VectorXdual {
    dual       y;
    VectorXdual g;
    auto f = [&]( const VectorXdual& xin ) -> dual { return stage_cost( xin, u, time_idx ); };
    autodiff::gradient( f, autodiff::wrt( xv ), autodiff::at( xv ), y, g );
    return g;
  };

  VectorXdual g_out;
  return autodiff::jacobian( grad_x, autodiff::wrt( x_ad ), autodiff::at( x_ad ), g_out );
}

inline Eigen::MatrixXd
autodiff_cost_control_hessian( const StageCostFunction& stage_cost, const State& x, const Control& u, size_t time_idx )
{
  using autodiff::dual;
  using VectorXdual = Eigen::Matrix<dual, Eigen::Dynamic, 1>;

  VectorXdual u_ad = u;

  auto grad_u = [&]( VectorXdual uv ) -> VectorXdual {
    dual       y;
    VectorXdual g;
    auto f = [&]( const VectorXdual& uin ) -> dual { return stage_cost( x, uin, time_idx ); };
    autodiff::gradient( f, autodiff::wrt( uv ), autodiff::at( uv ), y, g );
    return g;
  };

  VectorXdual g_out;
  return autodiff::jacobian( grad_u, autodiff::wrt( u_ad ), autodiff::at( u_ad ), g_out );
}

inline Eigen::MatrixXd
autodiff_cost_cross_term( const StageCostFunction& stage_cost, const State& x, const Control& u, size_t time_idx )
{
  using autodiff::dual;
  using VectorXdual = Eigen::Matrix<dual, Eigen::Dynamic, 1>;

  VectorXdual x_ad = x;
  VectorXdual u_ad = u;

  auto grad_u_given_x = [&]( VectorXdual xv ) -> VectorXdual {
    dual       y;
    VectorXdual g;
    auto f = [&]( const VectorXdual& uv ) -> dual { return stage_cost( xv, uv, time_idx ); };
    autodiff::gradient( f, autodiff::wrt( u_ad ), autodiff::at( u_ad ), y, g );
    return g;
  };

  VectorXdual g_out;
  // Jacobian of grad_u w.r.t x gives matrix of size (m x n)
  return autodiff::jacobian( grad_u_given_x, autodiff::wrt( x_ad ), autodiff::at( x_ad ), g_out );
}
} // namespace mas

#endif // MAS_USE_AUTODIFF
