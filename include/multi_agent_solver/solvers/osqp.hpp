
#pragma once

#include <chrono>
#include <map>
#include <memory>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "multi_agent_solver/integrator.hpp"
#include "multi_agent_solver/ocp.hpp"
#include "multi_agent_solver/solvers/solver.hpp"
#include "multi_agent_solver/types.hpp"
#include <OsqpEigen/OsqpEigen.h>

namespace mas
{

/**
 * @brief Thin wrapper around OsqpEigen::Solver that reuses **everything**:
 *  - the internal OSQP workspace (held in `solver`)
 *  - sparse‐matrix patterns for Hessian & constraints (rebuilt only when
 *    problem dimensions change)
 *  - dense work vectors (gradient, bounds, step, etc.).
 *
 * The helper free-functions from the original implementation have been
 * turned into **private member functions** that overwrite pre-allocated
 * matrices instead of reallocating.
 *
 */
class OSQP
{
public:

  explicit OSQP()
  {
    solver = std::make_unique<OsqpEigen::Solver>();
  }

  void
  set_params( const SolverParams& params )
  {
    max_iterations = static_cast<int>( params.at( "max_iterations" ) );
    tolerance      = params.at( "tolerance" );
    max_ms         = params.at( "max_ms" );
    debug          = params.count( "debug" ) && params.at( "debug" ) > 0.5;
    solver->settings()->setWarmStart( true );
    solver->settings()->setVerbosity( false );
    solver->settings()->setAdaptiveRho( true );
    solver->settings()->setMaxIteration( 1000 );
    solver->settings()->setScaling( 10 );
    solver->settings()->setPolish( true );
    if( params.count( "rho" ) )
      solver->settings()->setRho( params.at( "rho" ) );
    if( params.count( "sigma" ) )
      solver->settings()->setSigma( params.at( "sigma" ) );
  }

  //--------------------------------------------------------------------//
  /**
   * @brief Solve an OCP with the already-configured OSQP instance.
   *        Reuses all pre-allocated memory across calls.
   */
  void
  solve( OCP& problem )
  {
    using clock      = std::chrono::high_resolution_clock;
    const auto start = clock::now();

    resize_buffers( problem );

    const int T  = problem.horizon_steps;
    const int nx = problem.state_dim;
    const int nu = problem.control_dim;
    const int Ns = T + 1;
    const int Nc = T;


    StateTrajectory&   states   = problem.best_states;
    ControlTrajectory& controls = problem.best_controls;
    double&            cost     = problem.best_cost;


    states = integrate_horizon( problem.initial_state, controls, problem.dt, problem.dynamics, integrate_rk4 );
    cost   = problem.objective_function( states, controls );

    double reg = 1e-6;

    if( ls_parameters.empty() )
      ls_parameters = {
        { "initial_step_size",  1.0 },
        {              "beta",  0.5 },
        {                "c1", 1e-6 }
      };

    solution.setZero( qp_dim );
    dual_solution.setZero( qp_data.A.rows() );

    for( int iter = 0; iter < max_iterations; ++iter )
    {
      const double elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>( clock::now() - start ).count();
      if( elapsed_ms > max_ms )
      {
        if( debug )
          std::cerr << "OSQP-collocation finished in " << elapsed_ms << " ms,  cost = " << problem.best_cost << '\n';
        break;
      }

      assemble_qp_data( problem, states, controls, reg );

      if( !solver_initialized )
      {
        solver->data()->setNumberOfVariables( qp_dim );
        solver->data()->setNumberOfConstraints( qp_data.A.rows() );
        if( !solver->data()->setHessianMatrix( qp_data.H ) || !solver->data()->setGradient( qp_data.gradient )
            || !solver->data()->setLinearConstraintsMatrix( qp_data.A ) || !solver->data()->setLowerBound( qp_data.lb )
            || !solver->data()->setUpperBound( qp_data.ub ) )
          throw std::runtime_error( "failed to set initial QP data" );

        if( !solver->initSolver() || !solver->isInitialized() )
          throw std::runtime_error( "failed to initialise OSQP" );
        solver_initialized = true;
      }
      else
      {
        if( !solver->updateHessianMatrix( qp_data.H ) || !solver->updateGradient( qp_data.gradient )
            || !solver->updateLinearConstraintsMatrix( qp_data.A ) || !solver->updateLowerBound( qp_data.lb )
            || !solver->updateUpperBound( qp_data.ub ) )
          throw std::runtime_error( "failed to update QP data" );
      }

      solver->warmStart( solution, dual_solution );

      if( solver->solveProblem() != OsqpEigen::ErrorExitFlag::NoError )
        throw std::runtime_error( "OSQP failed" );

      solution       = solver->getSolution();
      dual_solution  = solver->getDualSolution();

      for( int t = 0; t < Nc; ++t )
        u_candidate.col( t ) = solution.segment( t * nu, nu );

      d_u = controls - u_candidate;

      const double alpha = armijo_line_search( problem.initial_state, controls, d_u, problem.dynamics, problem.objective_function,
                                               problem.dt, ls_parameters );

      u_new                 = controls - alpha * d_u;
      states_new            = integrate_horizon( problem.initial_state, u_new, problem.dt, problem.dynamics, integrate_rk4 );
      const double cost_new = problem.objective_function( states_new, u_new );

      if( std::abs( cost - cost_new ) < tolerance )
      {
        controls = u_new;
        states   = states_new;
        cost     = cost_new;
        if( debug )
          std::cerr << "OSQP converged in " << iter + 1 << " SQP steps\n";
        break;
      }
      if( cost_new < cost )
      {
        controls = u_new;
        states   = states_new;
        cost     = cost_new;
      }
      else
      {
        if( debug )
          std::cerr << "OSQP converged in " << iter + 1 << " SQP steps\n";
        break;
      }
    }
  }

private:

  //--------------------------------------------------------------------//
  struct QPData
  {
    Eigen::SparseMatrix<double> H;
    Eigen::VectorXd             gradient;
    Eigen::SparseMatrix<double> A;
    Eigen::VectorXd             lb;
    Eigen::VectorXd             ub;
  } qp_data;

  std::unique_ptr<OsqpEigen::Solver> solver;
  bool                               solver_initialized = false;


  int qp_dim            = 0;
  int n_dyn_constraints = 0;
  int n_state_bounds    = 0;
  int n_control_bounds  = 0;

  Eigen::VectorXd   solution;
  Eigen::VectorXd   dual_solution;
  ControlTrajectory u_candidate;
  ControlTrajectory d_u;
  ControlTrajectory u_new;
  StateTrajectory   states_new;

  Eigen::MatrixXd H_dense;
  Eigen::MatrixXd A_dense;
  bool            sparsity_initialized = false;

  std::map<std::string, double> ls_parameters;
  
  //--------------------------------------------------------------------//
  void
  resize_buffers( const OCP& problem )
  {
    const int T  = problem.horizon_steps;
    const int nx = problem.state_dim;
    const int nu = problem.control_dim;

    qp_dim            = T * nu;
    n_dyn_constraints = 0;
    n_state_bounds    = ( T + 1 ) * nx;
    n_control_bounds  = T * nu;

    solution.resize( qp_dim );
    dual_solution.resize( n_state_bounds + n_control_bounds );
    u_candidate.resize( nu, T );
    d_u.resize( nu, T );
    u_new.resize( nu, T );
    states_new.resize( nx, T + 1 );

    qp_data.H.resize( qp_dim, qp_dim );
    qp_data.gradient.resize( qp_dim );
    qp_data.A.resize( n_state_bounds + n_control_bounds, qp_dim );
    qp_data.lb.resize( n_state_bounds + n_control_bounds );
    qp_data.ub.resize( n_state_bounds + n_control_bounds );

    H_dense.resize( qp_dim, qp_dim );
    A_dense.resize( n_state_bounds + n_control_bounds, qp_dim );

    sparsity_initialized = false;
  }

  //--------------------------------------------------------------------//
  void
  assemble_qp_data( const OCP& p, const StateTrajectory& X, const ControlTrajectory& U, double reg )
  {
    const int nx = p.state_dim;
    const int nu = p.control_dim;
    const int T  = p.horizon_steps;

    std::vector<Eigen::MatrixXd> A( T ), B( T );
    for( int t = 0; t < T; ++t )
    {
      A[t] = p.dynamics_state_jacobian( p.dynamics, X.col( t ), U.col( t ) );
      B[t] = p.dynamics_control_jacobian( p.dynamics, X.col( t ), U.col( t ) );
    }

    std::vector<std::vector<Eigen::MatrixXd>> G( T + 1 );
    G[0] = {};
    for( int t = 0; t < T; ++t )
    {
      G[t + 1].resize( t + 1 );
      for( int k = 0; k < t; ++k )
        G[t + 1][k] = A[t] * G[t][k];
      G[t + 1][t] = B[t];
    }

    std::vector<State> S( T + 1 );
    for( int t = 0; t <= T; ++t )
    {
      State s = X.col( t );
      for( int k = 0; k < t; ++k )
        s -= G[t][k] * U.col( k );
      S[t] = s;
    }

    H_dense.setZero();
    qp_data.gradient.setZero();
    A_dense.setZero();
    qp_data.lb.setZero();
    qp_data.ub.setZero();

    for( int t = 0; t < T; ++t )
    {
      const Eigen::VectorXd gx = p.cost_state_gradient( p.stage_cost, X.col( t ), U.col( t ), t );
      const Eigen::MatrixXd Q  = p.cost_state_hessian( p.stage_cost, X.col( t ), U.col( t ), t );
      for( int i = 0; i < t; ++i )
      {
        const Eigen::MatrixXd Gi = G[t][i];
        qp_data.gradient.segment( i * nu, nu ) += Gi.transpose() * gx;
        for( int j = 0; j < t; ++j )
          H_dense.block( i * nu, j * nu, nu, nu ) += Gi.transpose() * Q * G[t][j];
      }
      qp_data.gradient.segment( t * nu, nu )
        += p.cost_control_gradient( p.stage_cost, X.col( t ), U.col( t ), t );
      H_dense.block( t * nu, t * nu, nu, nu )
        += p.cost_control_hessian( p.stage_cost, X.col( t ), U.col( t ), t );
    }

    const Eigen::VectorXd gxT = p.cost_state_gradient( p.stage_cost, X.col( T ), U.col( T - 1 ), T );
    const Eigen::MatrixXd QT  = p.cost_state_hessian( p.stage_cost, X.col( T ), U.col( T - 1 ), T );
    for( int i = 0; i < T; ++i )
    {
      const Eigen::MatrixXd Gi = G[T][i];
      qp_data.gradient.segment( i * nu, nu ) += Gi.transpose() * gxT;
      for( int j = 0; j < T; ++j )
        H_dense.block( i * nu, j * nu, nu, nu ) += Gi.transpose() * QT * G[T][j];
    }

    H_dense.diagonal().array() += reg;

    int row = 0;
    for( int t = 0; t <= T; ++t )
      for( int i = 0; i < nx; ++i, ++row )
      {
        for( int k = 0; k < t; ++k )
          A_dense.block( row, k * nu, 1, nu ) = G[t][k].row( i );
        qp_data.lb( row ) = p.state_lower_bounds ? ( *p.state_lower_bounds )( i ) - S[t]( i ) : -OsqpEigen::INFTY;
        qp_data.ub( row ) = p.state_upper_bounds ? ( *p.state_upper_bounds )( i ) - S[t]( i ) : OsqpEigen::INFTY;
      }

    for( int t = 0; t < T; ++t )
      for( int i = 0; i < nu; ++i, ++row )
      {
        A_dense( row, t * nu + i ) = 1.0;
        qp_data.lb( row ) = p.input_lower_bounds ? ( *p.input_lower_bounds )( i ) : -OsqpEigen::INFTY;
        qp_data.ub( row ) = p.input_upper_bounds ? ( *p.input_upper_bounds )( i ) : OsqpEigen::INFTY;
      }

    if( !sparsity_initialized )
    {
      std::vector<Eigen::Triplet<double>> ht;
      ht.reserve( qp_dim * qp_dim );
      for( int j = 0; j < qp_dim; ++j )
        for( int i = 0; i < qp_dim; ++i )
          ht.emplace_back( i, j, H_dense( i, j ) );
      qp_data.H.setFromTriplets( ht.begin(), ht.end() );
      qp_data.H.makeCompressed();

      std::vector<Eigen::Triplet<double>> at;
      at.reserve( A_dense.rows() * A_dense.cols() );
      for( int j = 0; j < qp_dim; ++j )
        for( int i = 0; i < A_dense.rows(); ++i )
          at.emplace_back( i, j, A_dense( i, j ) );
      qp_data.A.setFromTriplets( at.begin(), at.end() );
      qp_data.A.makeCompressed();

      sparsity_initialized = true;
    }
    else
    {
      Eigen::Map<Eigen::VectorXd>( qp_data.H.valuePtr(), qp_data.H.nonZeros() )
        = Eigen::Map<const Eigen::VectorXd>( H_dense.data(), H_dense.size() );
      Eigen::Map<Eigen::VectorXd>( qp_data.A.valuePtr(), qp_data.A.nonZeros() )
        = Eigen::Map<const Eigen::VectorXd>( A_dense.data(), A_dense.size() );
    }
  }

  //--------------------------------------------------------------------//

  int    max_iterations;
  double tolerance;
  double max_ms;
  bool   debug = false;
};

} // namespace mas
