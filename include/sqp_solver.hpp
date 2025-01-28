#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "integrator.hpp"
#include "ocp.hpp"
#include "solver_output.hpp"
#include <OsqpEigen/OsqpEigen.h>

namespace
{

// Helper to construct Hessian matrix
Eigen::SparseMatrix<double>
construct_hessian( const OCP& problem, const StateTrajectory& states, const ControlTrajectory& controls )
{
  int                                 n_u = problem.control_dim;
  int                                 n_x = problem.state_dim;
  int                                 T   = problem.horizon_steps;
  Eigen::SparseMatrix<double>         H( ( T + 1 ) * n_x + T * n_u, ( T + 1 ) * n_x + T * n_u );
  std::vector<Eigen::Triplet<double>> H_triplets;

  for( int t = 0; t < T; ++t )
  {
    Eigen::MatrixXd R = problem.cost_control_hessian( problem.objective_function, states.col( t ), controls.col( t ) );
    for( int i = 0; i < n_u; ++i )
    {
      H_triplets.emplace_back( ( T + 1 ) * n_x + t * n_u + i, ( T + 1 ) * n_x + t * n_u + i, R( i, i ) );
    }
  }

  for( int t = 0; t < T + 1; ++t )
  {
    Eigen::MatrixXd Q = problem.cost_state_hessian( problem.objective_function, states.col( t ), controls.col( std::min( t, T - 1 ) ) );
    for( int i = 0; i < n_x; ++i )
    {
      H_triplets.emplace_back( t * n_x + i, t * n_x + i, Q( i, i ) );
    }
  }

  H.setFromTriplets( H_triplets.begin(), H_triplets.end() );
  return H;
}

// Helper to construct gradient vector
Eigen::VectorXd
construct_gradient( const OCP& problem, const StateTrajectory& states, const ControlTrajectory& controls )
{
  int             n_x      = problem.state_dim;
  int             n_u      = problem.control_dim;
  int             T        = problem.horizon_steps;
  Eigen::VectorXd gradient = Eigen::VectorXd::Zero( ( T + 1 ) * n_x + T * n_u );

  for( int t = 0; t < T; ++t )
  {
    Eigen::VectorXd g_u = problem.cost_control_gradient( problem.objective_function, states.col( t ), controls.col( t ) );
    for( int i = 0; i < n_u; ++i )
    {
      gradient( ( T + 1 ) * n_x + t * n_u + i ) = g_u( i );
    }
  }

  for( int t = 0; t < T + 1; ++t )
  {
    Eigen::VectorXd g_x = problem.cost_state_gradient( problem.objective_function, states.col( t ), controls.col( std::min( t, T - 1 ) ) );
    for( int i = 0; i < n_x; ++i )
    {
      gradient( t * n_x + i ) = g_x( i );
    }
  }

  return gradient;
}

// Helper to construct constraint matrix
Eigen::SparseMatrix<double>
construct_constraints( const OCP& problem, const StateTrajectory& states, const ControlTrajectory& controls )
{
  int                                 n_x = problem.state_dim;
  int                                 n_u = problem.control_dim;
  int                                 T   = problem.horizon_steps;
  std::vector<Eigen::Triplet<double>> A_c_triplets;

  for( int t = 0; t < T; ++t )
  {
    for( int i = 0; i < n_x; ++i )
    {
      A_c_triplets.emplace_back( t * n_x + i, t * n_x + i, -1.0 );
    }

    Eigen::MatrixXd A = problem.dynamics_state_jacobian( problem.dynamics, states.col( t ), controls.col( t ) );
    for( int i = 0; i < n_x; ++i )
    {
      for( int j = 0; j < n_x; ++j )
      {
        if( A( i, j ) != 0.0 )
        {
          A_c_triplets.emplace_back( ( t + 1 ) * n_x + i, t * n_x + j, A( i, j ) );
        }
      }
    }

    Eigen::MatrixXd B = problem.dynamics_control_jacobian( problem.dynamics, states.col( t ), controls.col( t ) );
    for( int i = 0; i < n_x; ++i )
    {
      for( int j = 0; j < n_u; ++j )
      {
        if( B( i, j ) != 0.0 )
        {
          A_c_triplets.emplace_back( ( t + 1 ) * n_x + i, ( T + 1 ) * n_x + t * n_u + j, B( i, j ) );
        }
      }
    }
  }

  for( int t = 0; t < T + 1; ++t )
  {
    for( int i = 0; i < n_x; ++i )
    {
      A_c_triplets.emplace_back( ( T + t ) * n_x + i, t * n_x + i, 1.0 );
    }
  }

  for( int t = 0; t < T; ++t )
  {
    for( int i = 0; i < n_u; ++i )
    {
      A_c_triplets.emplace_back( ( T + T + t ) * n_u + i, ( T + 1 ) * n_x + t * n_u + i, 1.0 );
    }
  }

  Eigen::SparseMatrix<double> A_c( ( T + 1 ) * n_x + ( T + 1 ) * n_x + T * n_u, ( T + 1 ) * n_x + T * n_u );
  A_c.setFromTriplets( A_c_triplets.begin(), A_c_triplets.end() );
  return A_c;
}

// Helper to construct bounds
std::pair<Eigen::VectorXd, Eigen::VectorXd>
construct_bounds( const OCP& problem )
{
  int             n_x = problem.state_dim;
  int             n_u = problem.control_dim;
  int             T   = problem.horizon_steps;
  Eigen::VectorXd l   = Eigen::VectorXd::Zero( ( T + 1 ) * n_x + ( T + 1 ) * n_x + T * n_u );
  Eigen::VectorXd u   = Eigen::VectorXd::Zero( ( T + 1 ) * n_x + ( T + 1 ) * n_x + T * n_u );

  l.head( ( T + 1 ) * n_x ).setZero();
  u.head( ( T + 1 ) * n_x ).setZero();

  if( problem.state_lower_bounds.has_value() && problem.state_upper_bounds.has_value() )
  {
    for( int t = 0; t < T + 1; ++t )
    {
      l.segment( ( T + t ) * n_x, n_x ) = *problem.state_lower_bounds;
      u.segment( ( T + t ) * n_x, n_x ) = *problem.state_upper_bounds;
    }
  }

  if( problem.input_lower_bounds.has_value() && problem.input_upper_bounds.has_value() )
  {
    for( int t = 0; t < T; ++t )
    {
      l.segment( ( T + T + t ) * n_u, n_u ) = *problem.input_lower_bounds;
      u.segment( ( T + T + t ) * n_u, n_u ) = *problem.input_upper_bounds;
    }
  }

  return { l, u };
}

} // namespace

SolverOutput
osqp_solver( const OCP& problem, int max_iterations = 100, double tolerance = 1e-5 )
{
  const int horizon_steps = problem.horizon_steps;
  const int state_dim     = problem.state_dim;
  const int control_dim   = problem.control_dim;

  StateTrajectory   states   = StateTrajectory::Zero( state_dim, horizon_steps + 1 );
  ControlTrajectory controls = ControlTrajectory::Zero( control_dim, horizon_steps );
  states.col( 0 )            = problem.initial_state;

  double cost = problem.objective_function( states, controls );

  for( int iter = 0; iter < max_iterations; ++iter )
  {
    Eigen::SparseMatrix<double> H        = construct_hessian( problem, states, controls );
    Eigen::VectorXd             gradient = construct_gradient( problem, states, controls );
    Eigen::SparseMatrix<double> A_c      = construct_constraints( problem, states, controls );
    auto [l, u]                          = construct_bounds( problem );

    OsqpEigen::Solver solver;
    solver.data()->setNumberOfVariables( H.cols() );
    solver.data()->setNumberOfConstraints( A_c.rows() );

    if( !solver.data()->setHessianMatrix( H ) )
      throw std::runtime_error( "Failed to set Hessian." );
    if( !solver.data()->setGradient( gradient ) )
      throw std::runtime_error( "Failed to set gradient." );
    if( !solver.data()->setLinearConstraintsMatrix( A_c ) )
      throw std::runtime_error( "Failed to set constraint matrix." );
    if( !solver.data()->setLowerBound( l ) )
      throw std::runtime_error( "Failed to set lower bounds." );
    if( !solver.data()->setUpperBound( u ) )
      throw std::runtime_error( "Failed to set upper bounds." );

    if( !solver.initSolver() )
      throw std::runtime_error( "Failed to initialize OSQP solver." );

    solver.settings()->setWarmStart( true );

    if( solver.solveProblem() != OsqpEigen::ErrorExitFlag::NoError )
    {
      throw std::runtime_error( "OSQP solver failed." );
    }

    Eigen::VectorXd solution = solver.getSolution();

    for( int t = 0; t < horizon_steps; ++t )
    {
      controls.col( t ) = solution.segment( ( horizon_steps + 1 ) * state_dim + t * control_dim, control_dim );
    }

    states = integrate_horizon( problem.initial_state, controls, problem.dt, problem.dynamics, integrate_rk4 );

    double new_cost = problem.objective_function( states, controls );

    if( std::abs( cost - new_cost ) < tolerance )
    {
      break;
    }

    cost = new_cost;
  }

  SolverOutput solution_output;
  solution_output.cost       = cost;
  solution_output.trajectory = states;
  solution_output.controls   = controls;
  return solution_output;
}
