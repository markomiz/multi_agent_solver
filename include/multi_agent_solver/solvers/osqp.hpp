
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


    double reg = 0.0;
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

    //--------------------------------------------------------------------//

    if( ls_parameters.empty() )
      ls_parameters = {
        { "initial_step_size",  1.0 },
        {              "beta",  0.5 },
        {                "c1", 1e-6 }
      };

    //--------------------------------------------------------------------//
    for( int iter = 0; iter < max_iterations; ++iter )
    {
      const double elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>( clock::now() - start ).count();
      if( elapsed_ms > max_ms )
      {
        if( debug )
          std::cerr << "OSQP solver stopped: time limit \n";
        break;
      }

      //---------------- adaptive Hessian regularisation -----------------
      bool hess_ok = false;
      for( int attempt = 0; attempt < 10; ++attempt )
      {
        assemble_hessian( problem, states, controls, reg );
        if( solver->updateHessianMatrix( qp_data.H ) )
        {
          hess_ok = true;
          break;
        }
        reg = ( reg == 0.0 ? 1e-6 : reg * 10.0 );
      }
      if( !hess_ok )
        throw std::runtime_error( "Hessian update failed" );

      //---------------- update remaining QP pieces ----------------------
      assemble_gradient( problem, states, controls );
      assemble_constraints( problem, states, controls );
      assemble_bounds( problem );

      if( !solver->updateGradient( qp_data.gradient ) || !solver->updateLinearConstraintsMatrix( qp_data.A )
          || !solver->updateLowerBound( qp_data.lb ) || !solver->updateUpperBound( qp_data.ub ) )
        throw std::runtime_error( "failed to push QP updates" );

      //---------------- solve QP ---------------------------------------
      if( solver->solveProblem() != OsqpEigen::ErrorExitFlag::NoError )
        throw std::runtime_error( "OSQP failed" );

      solution = solver->getSolution();


      for( int t = 0; t < Nc; ++t )
        u_candidate.col( t ) = solution.segment( Ns * nx + t * nu, nu );

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
        break;
      }
      if( cost_new < cost )
      {
        controls = u_new;
        states   = states_new;
        cost     = cost_new;
      }
      else
        break;
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
  ControlTrajectory u_candidate;
  ControlTrajectory d_u;
  ControlTrajectory u_new;
  StateTrajectory   states_new;


  std::map<std::string, double> ls_parameters;

  //--------------------------------------------------------------------//
  void
  resize_buffers( const OCP& problem )
  {
    const int T  = problem.horizon_steps;
    const int nx = problem.state_dim;
    const int nu = problem.control_dim;

    qp_dim            = ( T + 1 ) * nx + T * nu;
    n_dyn_constraints = T * nx;
    n_state_bounds    = ( T + 1 ) * nx;
    n_control_bounds  = T * nu;


    solution.resize( qp_dim );
    u_candidate.resize( nu, T );
    d_u.resize( nu, T );
    u_new.resize( nu, T );
    states_new.resize( nx, T + 1 );


    qp_data.H.resize( qp_dim, qp_dim );
    qp_data.gradient.resize( qp_dim );
    qp_data.A.resize( n_dyn_constraints + n_state_bounds + n_control_bounds, qp_dim );
    qp_data.lb.resize( qp_data.A.rows() );
    qp_data.ub.resize( qp_data.A.rows() );
  }

  //--------------------------------------------------------------------//
  void
  assemble_qp_data( const OCP& p, const StateTrajectory& x, const ControlTrajectory& u, double reg )
  {
    assemble_hessian( p, x, u, reg );
    assemble_gradient( p, x, u );
    assemble_constraints( p, x, u );
    assemble_bounds( p );
  }

  //--------------------------------------------------------------------//
  void
  assemble_hessian( const OCP& p, const StateTrajectory& x, const ControlTrajectory& u, double reg )
  {


    std::vector<Eigen::Triplet<double>> tri;
    tri.reserve( qp_dim );
    const int nx = p.state_dim;
    const int nu = p.control_dim;
    const int T  = p.horizon_steps;


    for( int t = 0; t <= T; ++t )
    {
      const Eigen::MatrixXd Q = p.cost_state_hessian( p.stage_cost, x.col( t ), u.col( std::min( t, T - 1 ) ), t );
      for( int i = 0; i < nx; ++i )
      {
        const int idx = t * nx + i;
        tri.emplace_back( idx, idx, std::max( Q( i, i ) + reg, 1e-6 ) );
      }
    }

    for( int t = 0; t < T; ++t )
    {
      const Eigen::MatrixXd R = p.cost_control_hessian( p.stage_cost, x.col( t ), u.col( t ), t );
      for( int i = 0; i < nu; ++i )
      {
        const int idx = ( T + 1 ) * nx + t * nu + i;
        tri.emplace_back( idx, idx, std::max( R( i, i ) + reg, 1e-6 ) );
      }
    }
    qp_data.H.setFromTriplets( tri.begin(), tri.end() );
  }

  //--------------------------------------------------------------------//
  void
  assemble_gradient( const OCP& p, const StateTrajectory& x, const ControlTrajectory& u )
  {
    const int nx = p.state_dim;
    const int nu = p.control_dim;
    const int T  = p.horizon_steps;

    qp_data.gradient.setZero();
    for( int t = 0; t <= T; ++t )
      qp_data.gradient.segment( t * nx, nx ) = p.cost_state_gradient( p.stage_cost, x.col( t ), u.col( t ), t );

    for( int t = 0; t < T; ++t )
      qp_data.gradient.segment( ( T + 1 ) * nx + t * nu, nu ) = p.cost_control_gradient( p.stage_cost, x.col( t ), u.col( t ), t );
  }

  //--------------------------------------------------------------------//
  void
  assemble_constraints( const OCP& p, const StateTrajectory& x, const ControlTrajectory& u )
  {
    const int nx = p.state_dim;
    const int nu = p.control_dim;
    const int T  = p.horizon_steps;

    std::vector<Eigen::Triplet<double>> tri;
    tri.reserve( ( n_dyn_constraints + n_state_bounds + n_control_bounds ) * ( nx + nu ) );


    for( int t = 0; t < T; ++t )
    {
      const int row_off = t * nx;

      for( int i = 0; i < nx; ++i )
        tri.emplace_back( row_off + i, ( t + 1 ) * nx + i, 1.0 );

      const Eigen::MatrixXd A_t = p.dynamics_state_jacobian( p.dynamics, x.col( t ), u.col( t ) );
      const Eigen::MatrixXd B_t = p.dynamics_control_jacobian( p.dynamics, x.col( t ), u.col( t ) );

      for( int i = 0; i < nx; ++i )
        for( int j = 0; j < nx; ++j )
          tri.emplace_back( row_off + i, t * nx + j, -A_t( i, j ) );

      for( int i = 0; i < nx; ++i )
        for( int j = 0; j < nu; ++j )
          tri.emplace_back( row_off + i, ( T + 1 ) * nx + t * nu + j, -B_t( i, j ) );
    }


    const int sb_off = n_dyn_constraints;
    for( int t = 0; t <= T; ++t )
      for( int i = 0; i < nx; ++i )
        tri.emplace_back( sb_off + t * nx + i, t * nx + i, 1.0 );


    const int cb_off = n_dyn_constraints + n_state_bounds;
    for( int t = 0; t < T; ++t )
      for( int i = 0; i < nu; ++i )
        tri.emplace_back( cb_off + t * nu + i, ( T + 1 ) * nx + t * nu + i, 1.0 );

    qp_data.A.setFromTriplets( tri.begin(), tri.end() );
  }

  //--------------------------------------------------------------------//
  void
  assemble_bounds( const OCP& p )
  {
    const int nx = p.state_dim;
    const int nu = p.control_dim;
    const int T  = p.horizon_steps;

    qp_data.lb.setZero();
    qp_data.ub.setZero();


    const int sb_off = n_dyn_constraints;
    for( int t = 0; t <= T; ++t )
      for( int i = 0; i < nx; ++i )
      {
        const int idx     = sb_off + t * nx + i;
        qp_data.lb( idx ) = p.state_lower_bounds ? ( *p.state_lower_bounds )( i ) : -OsqpEigen::INFTY;
        qp_data.ub( idx ) = p.state_upper_bounds ? ( *p.state_upper_bounds )( i ) : OsqpEigen::INFTY;
      }


    const int cb_off = n_dyn_constraints + n_state_bounds;
    for( int t = 0; t < T; ++t )
      for( int i = 0; i < nu; ++i )
      {
        const int idx     = cb_off + t * nu + i;
        qp_data.lb( idx ) = p.input_lower_bounds ? ( *p.input_lower_bounds )( i ) : -OsqpEigen::INFTY;
        qp_data.ub( idx ) = p.input_upper_bounds ? ( *p.input_upper_bounds )( i ) : OsqpEigen::INFTY;
      }
  }

  //--------------------------------------------------------------------//

  int    max_iterations;
  double tolerance;
  double max_ms;
  bool   debug;
};

} // namespace mas
