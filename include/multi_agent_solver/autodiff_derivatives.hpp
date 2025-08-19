#pragma once
#include <vector>

#include <cppad/cppad.hpp>
#include <Eigen/Dense>

#include "multi_agent_solver/integrator.hpp"
#include "multi_agent_solver/types.hpp"

namespace mas
{
// Dynamics jacobians -------------------------------------------------
inline Eigen::MatrixXd
autodiff_dynamics_state_jacobian( const MotionModel<>& dynamics, const State<>& x,
                                  const Control<>& u )
{
  using ADScalar = CppAD::AD<double>;
  const int nx   = x.size();
  const int nu   = u.size();
  std::vector<ADScalar> x_ad( nx );
  for( int i = 0; i < nx; ++i )
    x_ad[i] = x( i );
  CppAD::Independent( x_ad );

  State<ADScalar>  x_eig( nx );
  Control<ADScalar> u_eig( nu );
  for( int i = 0; i < nx; ++i )
    x_eig( i ) = x_ad[i];
  for( int i = 0; i < nu; ++i )
    u_eig( i ) = ADScalar( u( i ) );

  State<ADScalar> y = dynamics( x_eig, u_eig );
  std::vector<ADScalar> y_ad( nx );
  for( int i = 0; i < nx; ++i )
    y_ad[i] = y( i );

  CppAD::ADFun<double> f( x_ad, y_ad );
  std::vector<double>   x_val( nx );
  for( int i = 0; i < nx; ++i )
    x_val[i] = x( i );
  std::vector<double> jac = f.Jacobian( x_val );
  Eigen::MatrixXd     J( nx, nx );
  for( int i = 0; i < nx; ++i )
    for( int j = 0; j < nx; ++j )
      J( i, j ) = jac[i * nx + j];
  return J;
}

inline Eigen::MatrixXd
autodiff_dynamics_control_jacobian( const MotionModel<>& dynamics, const State<>& x,
                                    const Control<>& u )
{
  using ADScalar = CppAD::AD<double>;
  const int nx   = x.size();
  const int nu   = u.size();
  std::vector<ADScalar> u_ad( nu );
  for( int i = 0; i < nu; ++i )
    u_ad[i] = u( i );
  CppAD::Independent( u_ad );

  State<ADScalar>  x_eig = x.cast<ADScalar>();
  Control<ADScalar> u_eig( nu );
  for( int i = 0; i < nu; ++i )
    u_eig( i ) = u_ad[i];

  State<ADScalar> y = dynamics( x_eig, u_eig );
  std::vector<ADScalar> y_ad( nx );
  for( int i = 0; i < nx; ++i )
    y_ad[i] = y( i );

  CppAD::ADFun<double> f( u_ad, y_ad );
  std::vector<double>   u_val( nu );
  for( int i = 0; i < nu; ++i )
    u_val[i] = u( i );
  std::vector<double> jac = f.Jacobian( u_val );
  Eigen::MatrixXd     J( nx, nu );
  for( int i = 0; i < nx; ++i )
    for( int j = 0; j < nu; ++j )
      J( i, j ) = jac[i * nu + j];
  return J;
}

// Cost derivatives ----------------------------------------------------
inline Eigen::VectorXd
autodiff_cost_state_gradient( const StageCostFunction<>& cost, const State<>& x, const Control<>& u,
                              size_t t )
{
  using ADScalar = CppAD::AD<double>;
  const int nx   = x.size();
  std::vector<ADScalar> x_ad( nx );
  for( int i = 0; i < nx; ++i )
    x_ad[i] = x( i );
  CppAD::Independent( x_ad );

  State<ADScalar>  x_eig( nx );
  Control<ADScalar> u_eig = u.cast<ADScalar>();
  for( int i = 0; i < nx; ++i )
    x_eig( i ) = x_ad[i];

  std::vector<ADScalar> y( 1 );
  y[0] = cost( x_eig, u_eig, t );

  CppAD::ADFun<double> f( x_ad, y );
  std::vector<double>   x_val( nx );
  for( int i = 0; i < nx; ++i )
    x_val[i] = x( i );
  std::vector<double> jac = f.Jacobian( x_val );
  Eigen::VectorXd     g( nx );
  for( int i = 0; i < nx; ++i )
    g( i ) = jac[i];
  return g;
}

inline Eigen::VectorXd
autodiff_cost_control_gradient( const StageCostFunction<>& cost, const State<>& x, const Control<>& u,
                                size_t t )
{
  using ADScalar = CppAD::AD<double>;
  const int nu   = u.size();
  std::vector<ADScalar> u_ad( nu );
  for( int i = 0; i < nu; ++i )
    u_ad[i] = u( i );
  CppAD::Independent( u_ad );

  State<ADScalar>  x_eig = x.cast<ADScalar>();
  Control<ADScalar> u_eig( nu );
  for( int i = 0; i < nu; ++i )
    u_eig( i ) = u_ad[i];

  std::vector<ADScalar> y( 1 );
  y[0] = cost( x_eig, u_eig, t );

  CppAD::ADFun<double> f( u_ad, y );
  std::vector<double>   u_val( nu );
  for( int i = 0; i < nu; ++i )
    u_val[i] = u( i );
  std::vector<double> jac = f.Jacobian( u_val );
  Eigen::VectorXd     g( nu );
  for( int i = 0; i < nu; ++i )
    g( i ) = jac[i];
  return g;
}

inline Eigen::MatrixXd
autodiff_cost_state_hessian( const StageCostFunction<>& cost, const State<>& x, const Control<>& u,
                               size_t t )
{
  using ADScalar = CppAD::AD<double>;
  const int nx   = x.size();
  std::vector<ADScalar> x_ad( nx );
  for( int i = 0; i < nx; ++i )
    x_ad[i] = x( i );
  CppAD::Independent( x_ad );

  State<ADScalar>  x_eig( nx );
  Control<ADScalar> u_eig = u.cast<ADScalar>();
  for( int i = 0; i < nx; ++i )
    x_eig( i ) = x_ad[i];

  std::vector<ADScalar> y( 1 );
  y[0] = cost( x_eig, u_eig, t );
  CppAD::ADFun<double> f( x_ad, y );

  std::vector<double> x_val( nx );
  for( int i = 0; i < nx; ++i )
    x_val[i] = x( i );
  std::vector<double> h = f.Hessian( x_val, 0 );
  Eigen::MatrixXd     H( nx, nx );
  for( int i = 0; i < nx; ++i )
    for( int j = 0; j < nx; ++j )
      H( i, j ) = h[i * nx + j];
  return H;
}

inline Eigen::MatrixXd
autodiff_cost_control_hessian( const StageCostFunction<>& cost, const State<>& x, const Control<>& u,
                                 size_t t )
{
  using ADScalar = CppAD::AD<double>;
  const int nu   = u.size();
  std::vector<ADScalar> u_ad( nu );
  for( int i = 0; i < nu; ++i )
    u_ad[i] = u( i );
  CppAD::Independent( u_ad );

  State<ADScalar>  x_eig = x.cast<ADScalar>();
  Control<ADScalar> u_eig( nu );
  for( int i = 0; i < nu; ++i )
    u_eig( i ) = u_ad[i];

  std::vector<ADScalar> y( 1 );
  y[0] = cost( x_eig, u_eig, t );
  CppAD::ADFun<double> f( u_ad, y );

  std::vector<double> u_val( nu );
  for( int i = 0; i < nu; ++i )
    u_val[i] = u( i );
  std::vector<double> h = f.Hessian( u_val, 0 );
  Eigen::MatrixXd     H( nu, nu );
  for( int i = 0; i < nu; ++i )
    for( int j = 0; j < nu; ++j )
      H( i, j ) = h[i * nu + j];
  return H;
}

inline Eigen::MatrixXd
autodiff_cost_cross_term( const StageCostFunction<>& cost, const State<>& x, const Control<>& u, size_t t )
{
  using ADScalar = CppAD::AD<double>;
  const int nx   = x.size();
  const int nu   = u.size();
  const int n    = nx + nu;

  std::vector<ADScalar> xu_ad( n );
  for( int i = 0; i < nx; ++i )
    xu_ad[i] = x( i );
  for( int i = 0; i < nu; ++i )
    xu_ad[nx + i] = u( i );
  CppAD::Independent( xu_ad );

  State<ADScalar>  x_eig( nx );
  Control<ADScalar> u_eig( nu );
  for( int i = 0; i < nx; ++i )
    x_eig( i ) = xu_ad[i];
  for( int i = 0; i < nu; ++i )
    u_eig( i ) = xu_ad[nx + i];

  std::vector<ADScalar> y( 1 );
  y[0] = cost( x_eig, u_eig, t );
  CppAD::ADFun<double> f( xu_ad, y );

  std::vector<double> xu_val( n );
  for( int i = 0; i < n; ++i )
    xu_val[i] = ( i < nx ) ? x( i ) : u( i - nx );
  std::vector<double> h = f.Hessian( xu_val, 0 );
  Eigen::MatrixXd     H = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>( h.data(), n, n );
  return H.block( nx, 0, nu, nx );
}

// Gradient of trajectory cost wrt control sequence
template <typename Dynamics, typename Stage, typename Terminal>
inline ControlGradient
autodiff_trajectory_gradient( const State<>& x0, const ControlTrajectory& controls, const Dynamics& dynamics,
                              const Stage& stage_cost, const Terminal& terminal_cost, double dt )
{
  using ADScalar = CppAD::AD<double>;
  const int nu   = controls.rows();
  const int T    = controls.cols();
  const int nvar = nu * T;

  std::vector<ADScalar> u_ad( nvar );
  for( int t = 0; t < T; ++t )
    for( int i = 0; i < nu; ++i )
      u_ad[t * nu + i] = controls( i, t );
  CppAD::Independent( u_ad );

  ControlTrajectoryT<ADScalar> U( nu, T );
  for( int t = 0; t < T; ++t )
    for( int i = 0; i < nu; ++i )
      U( i, t ) = u_ad[t * nu + i];

  State<ADScalar> x_init = x0.cast<ADScalar>();
  StateTrajectoryT<ADScalar> X
    = integrate_horizon<ADScalar>( x_init, U, dt, dynamics, integrate_rk4<ADScalar> );

  ADScalar cost = 0.0;
  for( int t = 0; t < T; ++t )
    cost += stage_cost( X.col( t ), U.col( t ), t );
  cost += terminal_cost( X.col( T ) );

  std::vector<ADScalar> y( 1 );
  y[0] = cost;
  CppAD::ADFun<double> f( u_ad, y );

  std::vector<double> u_val( nvar );
  for( int t = 0; t < T; ++t )
    for( int i = 0; i < nu; ++i )
      u_val[t * nu + i] = controls( i, t );
  std::vector<double> grad = f.Jacobian( u_val );

  ControlGradient G( nu, T );
  for( int t = 0; t < T; ++t )
    for( int i = 0; i < nu; ++i )
      G( i, t ) = grad[t * nu + i];
  return G;
}
} // namespace mas
