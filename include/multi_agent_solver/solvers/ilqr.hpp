// iLQR solver with optional primal-dual augmented Lagrangian handling
#pragma once

#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

#include <Eigen/Dense>

#include "multi_agent_solver/constraint_helpers.hpp"
#include "multi_agent_solver/integrator.hpp"
#include "multi_agent_solver/ocp.hpp"
#include "multi_agent_solver/types.hpp"

namespace mas
{

/**
 * @brief Augmented-Lagrangian iLQR solver supporting path equality and inequality constraints.
 */
template<typename Scalar = double>
class iLQR
{
public:
  using ScalarType             = Scalar;
  using State                  = StateT<Scalar>;
  using Control                = ControlT<Scalar>;
  using StateTrajectory        = StateTrajectoryT<Scalar>;
  using ControlTrajectory      = ControlTrajectoryT<Scalar>;
  using MotionModel            = MotionModelT<Scalar>;
  using ObjectiveFunction      = ObjectiveFunctionT<Scalar>;
  using ConstraintViolations   = ConstraintViolationsT<Scalar>;
  using SolverParams           = SolverParamsT<Scalar>;
  using Problem                = OCP<Scalar>;
  using Matrix                 = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  using Vector                 = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  using Array                  = Eigen::Array<Scalar, Eigen::Dynamic, 1>;
  using LLT                    = Eigen::LLT<Matrix>;

  iLQR()
    : max_iterations( 50 )
    , tolerance( static_cast<Scalar>( 1e-6 ) )
    , max_ms( std::numeric_limits<Scalar>::infinity() )
    , debug( false )
    , penalty_parameter( static_cast<Scalar>( 10 ) )
    , penalty_increase( static_cast<Scalar>( 5 ) )
    , constraint_tolerance( static_cast<Scalar>( 1e-4 ) )
    , inequality_activation_tolerance( static_cast<Scalar>( 1e-6 ) )
    , equality_dim( 0 )
    , inequality_dim( 0 )
  {}

  void
  set_params( const SolverParams& params )
  {
    max_iterations = static_cast<int>( params.at( "max_iterations" ) );
    tolerance      = params.at( "tolerance" );
    max_ms         = params.at( "max_ms" );
    debug          = params.count( "debug" ) && params.at( "debug" ) > static_cast<Scalar>( 0.5 );

    if( auto it = params.find( "penalty" ); it != params.end() )
      penalty_parameter = it->second;
    if( auto it = params.find( "penalty_increase" ); it != params.end() )
      penalty_increase = it->second;
    if( auto it = params.find( "constraint_tolerance" ); it != params.end() )
      constraint_tolerance = it->second;
    if( auto it = params.find( "inequality_activation_tolerance" ); it != params.end() )
      inequality_activation_tolerance = it->second;
  }

  //------------------------------- API ----------------------------------//
  void
  solve( Problem& problem )
  {
    using clock      = std::chrono::high_resolution_clock;
    const auto start = clock::now();

    resize_buffers( problem );

    const int    T  = problem.horizon_steps;
    const int    nx = problem.state_dim;
    const int    nu = problem.control_dim;
    const Scalar dt = problem.dt;

    StateTrajectory&   x    = problem.best_states;
    ControlTrajectory& u    = problem.best_controls;
    Scalar&            cost = problem.best_cost;

    x    = integrate_horizon<Scalar>( problem.initial_state, u, dt, problem.dynamics, integrate_rk4<Scalar> );
    cost = problem.objective_function( x, u );

    Scalar current_merit = compute_merit( problem, x, u );
    if( debug )
      std::cout << "iLQR initial cost=" << cost << " merit=" << current_merit << '\n';

    for( int iter = 0; iter < max_iterations; ++iter )
    {
      const double elapsed_ms
        = static_cast<double>( std::chrono::duration_cast<std::chrono::milliseconds>( clock::now() - start ).count() );
      if( elapsed_ms > static_cast<double>( max_ms ) )
      {
        if( debug )
          std::cout << "iLQR time limit hit after " << elapsed_ms << " ms / " << max_ms << " ms\n";
        break;
      }

      if( problem.terminal_cost_gradient )
        v_x = problem.terminal_cost_gradient( problem.terminal_cost, x.col( T ) );
      else
        v_x.setZero();

      if( problem.terminal_cost_hessian )
        v_xx = problem.terminal_cost_hessian( problem.terminal_cost, x.col( T ) );
      else
        v_xx.setZero();

      v_xx = static_cast<Scalar>( 0.5 ) * ( v_xx + v_xx.transpose() );

      for( int t = T - 1; t >= 0; --t )
      {
        a_step[t] = problem.dynamics_state_jacobian( problem.dynamics, x.col( t ), u.col( t ) );
        b_step[t] = problem.dynamics_control_jacobian( problem.dynamics, x.col( t ), u.col( t ) );

        l_x_step[t]  = problem.cost_state_gradient( problem.stage_cost, x.col( t ), u.col( t ), t );
        l_u_step[t]  = problem.cost_control_gradient( problem.stage_cost, x.col( t ), u.col( t ), t );
        l_xx_step[t] = problem.cost_state_hessian( problem.stage_cost, x.col( t ), u.col( t ), t );
        l_uu_step[t] = problem.cost_control_hessian( problem.stage_cost, x.col( t ), u.col( t ), t );
        l_ux_step[t] = problem.cost_cross_term( problem.stage_cost, x.col( t ), u.col( t ), t );

        q_x_step[t]  = l_x_step[t] + a_step[t].transpose() * v_x;
        q_u_step[t]  = l_u_step[t] + b_step[t].transpose() * v_x;
        q_xx_step[t] = l_xx_step[t] + a_step[t].transpose() * v_xx * a_step[t];
        q_ux_step[t] = l_ux_step[t] + b_step[t].transpose() * v_xx * a_step[t];
        q_uu_step[t] = l_uu_step[t] + b_step[t].transpose() * v_xx * b_step[t];

        if( equality_dim > 0 && problem.equality_constraints )
        {
          eq_residuals[t] = problem.equality_constraints( x.col( t ), u.col( t ) );
          eq_jacobian_x[t]
            = problem.equality_constraints_state_jacobian
                ? problem.equality_constraints_state_jacobian( x.col( t ), u.col( t ) )
                : compute_constraints_state_jacobian<Scalar>( problem.equality_constraints, x.col( t ), u.col( t ) );
          eq_jacobian_u[t]
            = problem.equality_constraints_control_jacobian
                ? problem.equality_constraints_control_jacobian( x.col( t ), u.col( t ) )
                : compute_constraints_control_jacobian<Scalar>( problem.equality_constraints, x.col( t ), u.col( t ) );

          const Vector dual = eq_multipliers[t] + penalty_parameter * eq_residuals[t];
          q_x_step[t] += eq_jacobian_x[t].transpose() * dual;
          q_u_step[t] += eq_jacobian_u[t].transpose() * dual;

          q_xx_step[t] += penalty_parameter * eq_jacobian_x[t].transpose() * eq_jacobian_x[t];
          q_ux_step[t] += penalty_parameter * eq_jacobian_u[t].transpose() * eq_jacobian_x[t];
          q_uu_step[t] += penalty_parameter * eq_jacobian_u[t].transpose() * eq_jacobian_u[t];
        }

        if( inequality_dim > 0 && problem.inequality_constraints )
        {
          ineq_residuals[t] = problem.inequality_constraints( x.col( t ), u.col( t ) );
          ineq_jacobian_x[t]
            = problem.inequality_constraints_state_jacobian
                ? problem.inequality_constraints_state_jacobian( x.col( t ), u.col( t ) )
                : compute_constraints_state_jacobian<Scalar>( problem.inequality_constraints, x.col( t ), u.col( t ) );
          ineq_jacobian_u[t]
            = problem.inequality_constraints_control_jacobian
                ? problem.inequality_constraints_control_jacobian( x.col( t ), u.col( t ) )
                : compute_constraints_control_jacobian<Scalar>( problem.inequality_constraints, x.col( t ), u.col( t ) );

          const Vector slack  = ineq_residuals[t].cwiseMax( static_cast<Scalar>( 0 ) );
          const Array  active = ( ineq_residuals[t].array() > -inequality_activation_tolerance ).template cast<Scalar>();
          const Vector dual   = ineq_multipliers[t].array() * active + penalty_parameter * slack.array() * active;

          q_x_step[t] += ineq_jacobian_x[t].transpose() * dual;
          q_u_step[t] += ineq_jacobian_u[t].transpose() * dual;

          if( active.any() )
          {
            const Matrix active_diag = active.matrix().asDiagonal();
            q_xx_step[t] += penalty_parameter * ineq_jacobian_x[t].transpose() * active_diag * ineq_jacobian_x[t];
            q_ux_step[t] += penalty_parameter * ineq_jacobian_u[t].transpose() * active_diag * ineq_jacobian_x[t];
            q_uu_step[t] += penalty_parameter * ineq_jacobian_u[t].transpose() * active_diag * ineq_jacobian_u[t];
          }
        }

        q_uu_reg_step[t] = q_uu_step[t];
        auto&  llt       = llt_step[t];
        Scalar reg       = static_cast<Scalar>( 1e-6 );
        while( true )
        {
          llt.compute( q_uu_reg_step[t] );
          if( llt.info() == Eigen::Success )
            break;
          q_uu_reg_step[t] += reg * identity_nu;
          reg              *= static_cast<Scalar>( 10 );
        }
        q_uu_inv_step[t] = llt.solve( identity_nu );

        k[t]        = -q_uu_inv_step[t] * q_u_step[t];
        k_matrix[t] = -q_uu_inv_step[t] * q_ux_step[t];

        v_x = q_x_step[t] + k_matrix[t].transpose() * q_u_step[t] + q_ux_step[t].transpose() * k[t]
            + k_matrix[t].transpose() * q_uu_step[t] * k[t];
        v_xx = q_xx_step[t] + k_matrix[t].transpose() * q_ux_step[t] + q_ux_step[t].transpose() * k_matrix[t]
             + k_matrix[t].transpose() * q_uu_step[t] * k_matrix[t];
        v_xx = static_cast<Scalar>( 0.5 ) * ( v_xx + v_xx.transpose() );
      }

      x_trial.setZero();
      u_trial.setZero();
      x_trial.col( 0 ) = problem.initial_state;

      const Scalar amin = static_cast<Scalar>( 1e-3 );
      Scalar       alpha = static_cast<Scalar>( 1 );

      Scalar            best_merit = current_merit;
      StateTrajectory   best_x     = x;
      ControlTrajectory best_u     = u;

      while( alpha >= amin )
      {
        for( int t = 0; t < T; ++t )
        {
          const Vector dx = x_trial.col( t ) - x.col( t );
          u_trial.col( t )         = u.col( t ) + alpha * k[t] + k_matrix[t] * dx;

          if( problem.input_lower_bounds && problem.input_upper_bounds )
            clamp_controls<Scalar>( u_trial, *problem.input_lower_bounds, *problem.input_upper_bounds );

          x_trial.col( t + 1 ) = integrate_rk4<Scalar>( x_trial.col( t ), u_trial.col( t ), dt, problem.dynamics );
        }

        const Scalar trial_merit = compute_merit( problem, x_trial, u_trial );
        if( trial_merit < best_merit )
        {
          best_merit = trial_merit;
          best_x     = x_trial;
          best_u     = u_trial;
          break;
        }
        alpha *= static_cast<Scalar>( 0.5 );
      }

      const Scalar improvement = current_merit - best_merit;
      x    = best_x;
      u    = best_u;
      cost = problem.objective_function( x, u );
      current_merit = best_merit;

      Scalar eq_violation_norm   = static_cast<Scalar>( 0 );
      Scalar ineq_violation_norm = static_cast<Scalar>( 0 );

      for( int t = 0; t < T; ++t )
      {
        if( equality_dim > 0 && problem.equality_constraints )
        {
          const Vector residual = problem.equality_constraints( x.col( t ), u.col( t ) );
          eq_multipliers[t] += penalty_parameter * residual;
          eq_violation_norm += residual.squaredNorm();
        }
        if( inequality_dim > 0 && problem.inequality_constraints )
        {
          const Vector residual = problem.inequality_constraints( x.col( t ), u.col( t ) );
          const Vector positive = residual.cwiseMax( static_cast<Scalar>( 0 ) );
          ineq_multipliers[t]            = ( ineq_multipliers[t] + penalty_parameter * positive ).cwiseMax( static_cast<Scalar>( 0 ) );
          ineq_violation_norm += positive.squaredNorm();
        }
      }

      eq_violation_norm
        = static_cast<Scalar>( std::sqrt( static_cast<double>( eq_violation_norm ) ) );
      ineq_violation_norm
        = static_cast<Scalar>( std::sqrt( static_cast<double>( ineq_violation_norm ) ) );

      if( eq_violation_norm > constraint_tolerance || ineq_violation_norm > constraint_tolerance )
        penalty_parameter *= penalty_increase;

      if( debug )
      {
        std::cout << "iLQR iter " << iter << ": cost=" << cost << " merit=" << current_merit << " d_merit="
                  << improvement << " eq_violation=" << eq_violation_norm << " ineq_violation=" << ineq_violation_norm
                  << '\n';
      }

      if( static_cast<double>( std::abs( improvement ) ) < static_cast<double>( tolerance )
          && eq_violation_norm < constraint_tolerance && ineq_violation_norm < constraint_tolerance )
        break;
    }
  }

private:
  //---------------- buffer management ----------------------------------//
  void
  resize_buffers( const Problem& problem )
  {
    const int T  = problem.horizon_steps;
    const int nx = problem.state_dim;
    const int nu = problem.control_dim;

    auto resize_mat_vec = [&]( auto& container, auto&& prototype ) {
      if( static_cast<int>( container.size() ) != T )
        container.assign( T, prototype );
      else
        for( auto& m : container )
          m.setZero();
    };

    resize_mat_vec( k, Vector::Zero( nu ) );
    resize_mat_vec( k_matrix, Matrix::Zero( nu, nx ) );

    resize_mat_vec( a_step, Matrix::Zero( nx, nx ) );
    resize_mat_vec( b_step, Matrix::Zero( nx, nu ) );
    resize_mat_vec( l_x_step, Vector::Zero( nx ) );
    resize_mat_vec( l_u_step, Vector::Zero( nu ) );
    resize_mat_vec( l_xx_step, Matrix::Zero( nx, nx ) );
    resize_mat_vec( l_uu_step, Matrix::Zero( nu, nu ) );
    resize_mat_vec( l_ux_step, Matrix::Zero( nu, nx ) );

    resize_mat_vec( q_x_step, Vector::Zero( nx ) );
    resize_mat_vec( q_u_step, Vector::Zero( nu ) );
    resize_mat_vec( q_xx_step, Matrix::Zero( nx, nx ) );
    resize_mat_vec( q_ux_step, Matrix::Zero( nu, nx ) );
    resize_mat_vec( q_uu_step, Matrix::Zero( nu, nu ) );
    resize_mat_vec( q_uu_reg_step, Matrix::Zero( nu, nu ) );
    resize_mat_vec( q_uu_inv_step, Matrix::Zero( nu, nu ) );

    if( static_cast<int>( llt_step.size() ) != T )
      llt_step.assign( T, LLT( nu ) );

    Control default_control = Control::Zero( nu );
    if( problem.initial_controls.cols() == T )
      default_control = problem.initial_controls.col( 0 );

    equality_dim   = 0;
    inequality_dim = 0;
    if( problem.equality_constraints )
      equality_dim = static_cast<int>( problem.equality_constraints( problem.initial_state, default_control ).size() );
    if( problem.inequality_constraints )
      inequality_dim = static_cast<int>( problem.inequality_constraints( problem.initial_state, default_control ).size() );

    if( equality_dim > 0 )
    {
      resize_mat_vec( eq_residuals, Vector::Zero( equality_dim ) );
      resize_mat_vec( eq_jacobian_x, Matrix::Zero( equality_dim, nx ) );
      resize_mat_vec( eq_jacobian_u, Matrix::Zero( equality_dim, nu ) );

      if( static_cast<int>( eq_multipliers.size() ) != T )
        eq_multipliers.assign( T, Vector::Zero( equality_dim ) );
      else
        for( auto& m : eq_multipliers )
        {
          if( m.size() != equality_dim )
            m = Vector::Zero( equality_dim );
        }
    }
    else
    {
      eq_residuals.clear();
      eq_jacobian_x.clear();
      eq_jacobian_u.clear();
      eq_multipliers.clear();
    }

    if( inequality_dim > 0 )
    {
      resize_mat_vec( ineq_residuals, Vector::Zero( inequality_dim ) );
      resize_mat_vec( ineq_jacobian_x, Matrix::Zero( inequality_dim, nx ) );
      resize_mat_vec( ineq_jacobian_u, Matrix::Zero( inequality_dim, nu ) );

      if( static_cast<int>( ineq_multipliers.size() ) != T )
        ineq_multipliers.assign( T, Vector::Zero( inequality_dim ) );
      else
        for( auto& m : ineq_multipliers )
        {
          if( m.size() != inequality_dim )
            m = Vector::Zero( inequality_dim );
        }
    }
    else
    {
      ineq_residuals.clear();
      ineq_jacobian_x.clear();
      ineq_jacobian_u.clear();
      ineq_multipliers.clear();
    }

    x_trial.resize( nx, T + 1 );
    u_trial.resize( nu, T );

    v_x.resize( nx );
    v_xx.resize( nx, nx );
    identity_nu = Matrix::Identity( nu, nu );
  }

  Scalar
  compute_merit( const Problem& problem, const StateTrajectory& states, const ControlTrajectory& controls ) const
  {
    const int T = problem.horizon_steps;
    Scalar merit = problem.objective_function ? problem.objective_function( states, controls ) : problem.best_cost;

    for( int t = 0; t < T; ++t )
    {
      if( equality_dim > 0 && problem.equality_constraints )
      {
        const Vector residual = problem.equality_constraints( states.col( t ), controls.col( t ) );
        merit += eq_multipliers[t].dot( residual )
                 + static_cast<Scalar>( 0.5 ) * penalty_parameter * residual.squaredNorm();
      }
      if( inequality_dim > 0 && problem.inequality_constraints )
      {
        const Vector residual = problem.inequality_constraints( states.col( t ), controls.col( t ) );
        const Vector slack    = residual.cwiseMax( static_cast<Scalar>( 0 ) );
        const Array  active
          = ( residual.array() > -inequality_activation_tolerance ).template cast<Scalar>();
        const Vector active_slack      = slack.array() * active;
        const Vector weighted_multiply = ineq_multipliers[t].array() * active;
        merit += weighted_multiply.dot( active_slack );
        merit += static_cast<Scalar>( 0.5 ) * penalty_parameter * active_slack.squaredNorm();
      }
    }

    return merit;
  }

  //---------------- data members ---------------------------------------//
  int    max_iterations;
  Scalar tolerance;
  Scalar max_ms;
  bool   debug;

  Scalar penalty_parameter;
  Scalar penalty_increase;
  Scalar constraint_tolerance;
  Scalar inequality_activation_tolerance;

  int equality_dim;
  int inequality_dim;

  std::vector<Vector> k;
  std::vector<Matrix> k_matrix;

  std::vector<Matrix> a_step;
  std::vector<Matrix> b_step;

  std::vector<Vector> l_x_step;
  std::vector<Vector> l_u_step;
  std::vector<Matrix> l_xx_step;
  std::vector<Matrix> l_uu_step;
  std::vector<Matrix> l_ux_step;

  std::vector<Vector> q_x_step;
  std::vector<Vector> q_u_step;
  std::vector<Matrix> q_xx_step;
  std::vector<Matrix> q_ux_step;
  std::vector<Matrix> q_uu_step;
  std::vector<Matrix> q_uu_reg_step;
  std::vector<Matrix> q_uu_inv_step;

  std::vector<LLT> llt_step;

  std::vector<Vector> eq_residuals;
  std::vector<Matrix> eq_jacobian_x;
  std::vector<Matrix> eq_jacobian_u;
  std::vector<Vector> eq_multipliers;

  std::vector<Vector> ineq_residuals;
  std::vector<Matrix> ineq_jacobian_x;
  std::vector<Matrix> ineq_jacobian_u;
  std::vector<Vector> ineq_multipliers;

  StateTrajectory   x_trial;
  ControlTrajectory u_trial;

  Vector v_x;
  Matrix v_xx;
  Matrix identity_nu;
};

using iLQRd = iLQR<double>;
using iLQRf = iLQR<float>;

} // namespace mas
