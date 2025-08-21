#pragma once
/* ---------------  Standard / Eigen / OSQP  ------------------------ */
#include <cassert>
#include <chrono>
#include <iostream>
#include <memory>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <OsqpEigen/OsqpEigen.h>

/* ------------  project specific forward declarations -------------- */
#include "multi_agent_solver/ocp.hpp"
#include "multi_agent_solver/types.hpp"

namespace mas
{
/* =================================================================== */
/*                    Trapezoidal SQP with OSQP                        */
/* =================================================================== */
class OSQPCollocation
{
public:

  explicit OSQPCollocation() :
    solver{ std::make_unique<OsqpEigen::Solver>() }
  {}

  void set_params( const SolverParams& params );
  void solve( OCP& problem ); // main entry

private:

  /* ---------- helpers: decision-variable indices ----------------- */
  static inline int
  id_state( int t, int i, int nx )
  {
    return ( t - 1 ) * nx + i;
  }

  static inline int
  id_control( int t, int nx, int nu, int T )
  {
    return T * nx + t * nu;
  }

  /* ---------- phase 1: immutable sparsity pattern ---------------- */
  void prepare_structure( const OCP& );

  /* ---------- phase 2: overwrite numeric values ------------------ */
  void assemble_values( const OCP& p, const StateTrajectory& X, const ControlTrajectory& U, double reg );

  /* ---------- data ------------------------------------------------ */
  struct qpt
  {
    Eigen::SparseMatrix<double> H;      // Hessian
    Eigen::VectorXd             g;      // gradient
    Eigen::SparseMatrix<double> A;      // constraint matrix
    Eigen::VectorXd             lb, ub; // bounds
  } qp;

  /* OSQP */
  std::unique_ptr<OsqpEigen::Solver> solver;
  bool                               structure_ready{ false };

  /* problem dimensions */
  int T{}, nx{}, nu{}, qp_dim{};
  int n_dyn{}, n_state_bnd_{}, n_ctrl_bnd_{};

  /* index maps into value arrays (CSC order) */
  std::vector<Eigen::Index> H_idx;     // all mutable nnz in H
  std::vector<Eigen::Index> A_dyn_idx; // dynamics rows only

  /* settings */
  int    max_iter{ 20 };
  double tolerance{ 1e-4 };
  bool   debug{ false };

  double max_ms{ 1000.0 }; // max time in ms

  /* caching ----------------------------------------------------- */
  bool                         use_cache{ true };
  bool                         cache_valid{ false };
  StateTrajectory              X_prev;
  ControlTrajectory            U_prev;
  std::vector<Eigen::MatrixXd> Fx_cache; // size T+1
  std::vector<Eigen::MatrixXd> Fu_cache; // size T+1
  std::vector<Eigen::VectorXd> f_cache;  // size T+1
  std::vector<Eigen::MatrixXd> Q_cache;  // size T+1
  std::vector<Eigen::MatrixXd> R_cache;  // size T
};

/* =================================================================== */
/*                    Implementation details                            */
/* =================================================================== */
inline void
OSQPCollocation::set_params( const SolverParams& p )
{
  max_iter  = static_cast<int>( p.at( "max_iterations" ) );
  tolerance = p.at( "tolerance" );
  debug     = p.count( "debug" ) && p.at( "debug" ) > 0.5;
  max_ms    = p.count( "max_ms" ) ? p.at( "max_ms" ) : 1000.0;
  use_cache = p.count( "cache" ) ? p.at( "cache" ) > 0.5 : true;
  solver->settings()->setVerbosity( false );
  solver->settings()->setWarmStart( true );
  solver->settings()->setAdaptiveRho( true );
  solver->settings()->setScaling( 10 );
  solver->settings()->setPolish( true );
  solver->settings()->setMaxIteration( static_cast<int>( p.count( "osqpmax_iter" ) ? p.at( "osqpmax_iter" ) : 4000 ) );
  solver->settings()->setAbsoluteTolerance( p.count( "osqpabs_tol" ) ? p.at( "osqpabs_tol" ) : 1e-4 );
  solver->settings()->setRelativeTolerance( p.count( "osqprel_tol" ) ? p.at( "osqprel_tol" ) : 1e-4 );
}

/* ---------- phase 1: build fixed sparsity once -------------------- */
/* ------------------------------------------------------------------ */
/*  Phase-1: build immutable sparsity pattern and initialise OSQP     */
/* ------------------------------------------------------------------ */
inline void
OSQPCollocation::prepare_structure( const OCP& p )
{
  /* ---------- basic dimensions ---------------------------------- */
  T  = p.horizon_steps;
  nx = p.state_dim;
  nu = p.control_dim;

  qp_dim       = T * nx + T * nu; // decision-var length
  n_dyn        = T * nx;          // collocation equalities
  n_state_bnd_ = T * nx;          // state bound rows
  n_ctrl_bnd_  = T * nu;          // input bound rows

  /* ---------- allocate QP buffers (shape only) ------------------ */
  qp.H.resize( qp_dim, qp_dim );
  qp.g.resize( qp_dim );
  qp.A.resize( n_dyn + n_state_bnd_ + n_ctrl_bnd_, qp_dim );
  qp.lb.resize( qp.A.rows() );
  qp.ub.resize( qp.A.rows() );

  /* initialise them to zero so we can pass l-values to OSQP ------- */
  qp.g.setZero();
  qp.lb.setZero();
  qp.ub.setZero();

  /* ---------- build Hessian sparsity pattern -------------------- */
  std::vector<Eigen::Triplet<double>> H_tpl;
  H_tpl.reserve( qp_dim ); // diagonal reg

  /* Q-blocks (δx) */
  for( int t = 1; t <= T; ++t )
    for( int i = 0; i < nx; ++i )
      H_tpl.emplace_back( id_state( t, i, nx ), id_state( t, i, nx ), 0.0 );

  /* R-blocks (δu) */
  for( int t = 0; t < T; ++t )
    for( int i = 0; i < nu; ++i )
      H_tpl.emplace_back( id_control( t, nx, nu, T ) + i, id_control( t, nx, nu, T ) + i, 0.0 );

  /* small reg on every variable */
  for( int k = 0; k < qp_dim; ++k )
    H_tpl.emplace_back( k, k, 0.0 );

  qp.H.setFromTriplets( H_tpl.begin(), H_tpl.end() );
  qp.H.makeCompressed();

  /* keep CSC indices of *all* mutable Hessian entries */
  H_idx.assign( qp.H.innerIndexPtr(), qp.H.innerIndexPtr() + qp.H.nonZeros() );

  /* ---------- build constraint matrix sparsity pattern ---------- */
  std::vector<Eigen::Triplet<double>> A_tpl;
  A_tpl.reserve( n_dyn * ( 2 * nx + 2 * nu ) + n_state_bnd_ + n_ctrl_bnd_ );

  /* dynamics rows (trapezoidal) */
  for( int t = 0; t < T; ++t )
  {
    const int row0 = t * nx;
    for( int i = 0; i < nx; ++i )
    {
      /* δx_{t+1} */
      for( int j = 0; j < nx; ++j )
        A_tpl.emplace_back( row0 + i, id_state( t + 1, j, nx ), 0.0 );

      /* δx_t */
      if( t > 0 )
        for( int j = 0; j < nx; ++j )
          A_tpl.emplace_back( row0 + i, id_state( t, j, nx ), 0.0 );

      /* δu_t */
      for( int j = 0; j < nu; ++j )
        A_tpl.emplace_back( row0 + i, id_control( t, nx, nu, T ) + j, 0.0 );

      /* δu_{t+1} */
      if( t + 1 < T )
        for( int j = 0; j < nu; ++j )
          A_tpl.emplace_back( row0 + i, id_control( t + 1, nx, nu, T ) + j, 0.0 );
    }
  }

  /* bound identity rows */
  for( int r = n_dyn; r < qp.A.rows(); ++r )
    A_tpl.emplace_back( r, r - n_dyn, 1.0 );

  qp.A.setFromTriplets( A_tpl.begin(), A_tpl.end() );
  qp.A.makeCompressed();

  /* store CSC offsets of the dynamics entries (skip bound rows) */
  for( int col = 0; col < qp.A.outerSize(); ++col )
  {
    auto start = qp.A.outerIndexPtr()[col];
    auto end   = qp.A.outerIndexPtr()[col + 1];
    for( Eigen::Index idx = start; idx < end; ++idx )
      if( qp.A.innerIndexPtr()[idx] < n_dyn )
        A_dyn_idx.push_back( idx );
  }

  if( use_cache )
  {
    X_prev.setZero( nx, T + 1 );
    U_prev.setZero( nu, T );
    Fx_cache.assign( T + 1, Eigen::MatrixXd::Zero( nx, nx ) );
    Fu_cache.assign( T + 1, Eigen::MatrixXd::Zero( nx, nu ) );
    f_cache.assign( T + 1, Eigen::VectorXd::Zero( nx ) );
    Q_cache.assign( T + 1, Eigen::MatrixXd::Zero( nx, nx ) );
    R_cache.assign( T, Eigen::MatrixXd::Zero( nu, nu ) );
    cache_valid = false;
  }

  /* ---------- hand everything to OSQP once ---------------------- */
  solver->data()->setNumberOfVariables( qp_dim );
  solver->data()->setNumberOfConstraints( qp.A.rows() );

  if( !solver->data()->setHessianMatrix( qp.H ) || !solver->data()->setGradient( qp.g ) ||              // l-value
      !solver->data()->setLinearConstraintsMatrix( qp.A ) || !solver->data()->setLowerBound( qp.lb ) || // l-value
      !solver->data()->setUpperBound( qp.ub ) ||                                                        // l-value
      !solver->initSolver() )
  {
    throw std::runtime_error( "OSQP initialisation failed" );
  }

  structure_ready = true;
}

inline void
OSQPCollocation::assemble_values( const OCP& p, const StateTrajectory& X, const ControlTrajectory& U, double reg )
{
  qp.g.setZero();
  for( int t = 1; t <= T; ++t )
    qp.g.segment( id_state( t, 0, nx ), nx ) = p.cost_state_gradient( p.stage_cost, X.col( t ), U.col( std::min( t, T - 1 ) ), t );
  for( int t = 0; t < T; ++t )
    qp.g.segment( id_control( t, nx, nu, T ), nu ) = p.cost_control_gradient( p.stage_cost, X.col( t ), U.col( t ), t );

  double*     h_val = qp.H.valuePtr();
  std::size_t kH    = 0;

  constexpr double cache_eps = 1e-9;

  for( int t = 1; t <= T; ++t )
  {
    const auto x         = X.col( t );
    const auto u         = U.col( std::min( t, T - 1 ) );
    bool       recompute = !use_cache || !cache_valid || ( x - X_prev.col( t ) ).norm() > cache_eps
                  || ( u - U_prev.col( std::min( t, T - 1 ) ) ).norm() > cache_eps;

    if( recompute )
    {
      auto Q = p.cost_state_hessian( p.stage_cost, x, u, t );
      if( !Q.allFinite() )
        std::cerr << "[Warning] Non-finite Q at t=" << t << "\n";

      double min_diag = Q.diagonal().minCoeff();
      if( min_diag + reg < 0.0 )
      {
        double lambda         = std::abs( min_diag ) + reg;
        Q.diagonal().array() += lambda;
      }

      Q_cache[t] = Q;
    }

    for( int i = 0; i < nx; ++i )
      h_val[H_idx[kH++]] = Q_cache[t]( i, i );
  }

  for( int t = 0; t < T; ++t )
  {
    const auto x   = X.col( t );
    const auto u   = U.col( t );
    bool recompute = !use_cache || !cache_valid || ( x - X_prev.col( t ) ).norm() > cache_eps || ( u - U_prev.col( t ) ).norm() > cache_eps;

    if( recompute )
    {
      auto R = p.cost_control_hessian( p.stage_cost, x, u, t );
      if( !R.allFinite() )
        std::cerr << "[Warning] Non-finite R at t=" << t << "\n";

      double min_diag = R.diagonal().minCoeff();
      if( min_diag + reg < 0.0 )
      {
        double lambda         = std::abs( min_diag ) + reg;
        R.diagonal().array() += lambda;
      }

      R_cache[t] = R;
    }

    for( int i = 0; i < nu; ++i )
      h_val[H_idx[kH++]] = R_cache[t]( i, i );
  }

  for( ; kH < H_idx.size(); ++kH )
    h_val[H_idx[kH]] = reg;

  /* pre-compute dynamics Jacobians where needed ---------------- */
  for( int t = 0; t <= T; ++t )
  {
    const auto x         = X.col( t );
    const auto u         = U.col( std::min( t, T - 1 ) );
    bool       recompute = !use_cache || !cache_valid || ( x - X_prev.col( t ) ).norm() > cache_eps
                  || ( u - U_prev.col( std::min( t, T - 1 ) ) ).norm() > cache_eps;

    if( recompute )
    {
      Fx_cache[t] = p.dynamics_state_jacobian( p.dynamics, x, u );
      Fu_cache[t] = p.dynamics_control_jacobian( p.dynamics, x, u );
      f_cache[t]  = p.dynamics( x, u );
    }
  }

  for( int t = 0; t < T; ++t )
  {
    const int row0 = t * nx;

    const auto& Fx_t   = Fx_cache[t];
    const auto& Fu_t   = Fu_cache[t];
    const auto& Fx_tp1 = Fx_cache[t + 1];
    const auto& Fu_tp1 = Fu_cache[t + 1];
    const auto& f_t    = f_cache[t];
    const auto& f_tp1  = f_cache[t + 1];

    const auto defect         = X.col( t + 1 ) - X.col( t ) - 0.5 * p.dt * ( f_t + f_tp1 );
    qp.lb.segment( row0, nx ) = -defect;
    qp.ub.segment( row0, nx ) = -defect;

    for( int i = 0; i < nx; ++i )
    {
      for( int j = 0; j < nx; ++j )
        qp.A.coeffRef( row0 + i, id_state( t + 1, j, nx ) ) = ( i == j ? 1.0 : 0.0 ) - 0.5 * p.dt * Fx_tp1( i, j );

      if( t > 0 )
        for( int j = 0; j < nx; ++j )
          qp.A.coeffRef( row0 + i, id_state( t, j, nx ) ) = -( i == j ? 1.0 : 0.0 ) - 0.5 * p.dt * Fx_t( i, j );

      for( int j = 0; j < nu; ++j )
        qp.A.coeffRef( row0 + i, id_control( t, nx, nu, T ) + j ) = -0.5 * p.dt * Fu_t( i, j );

      if( t + 1 < T )
        for( int j = 0; j < nu; ++j )
          qp.A.coeffRef( row0 + i, id_control( t + 1, nx, nu, T ) + j ) = -0.5 * p.dt * Fu_tp1( i, j );
    }
  }

  if( use_cache )
  {
    X_prev      = X;
    U_prev      = U;
    cache_valid = true;
  }

  const int sb_off = n_dyn;
  const int cb_off = n_dyn + n_state_bnd_;

  for( int t = 1; t <= T; ++t )
  {
    const auto xr = X.col( t );
    for( int i = 0; i < nx; ++i )
    {
      int idx      = sb_off + ( t - 1 ) * nx + i;
      qp.lb( idx ) = p.state_lower_bounds ? ( *p.state_lower_bounds - xr )( i ) : -OsqpEigen::INFTY;
      qp.ub( idx ) = p.state_upper_bounds ? ( *p.state_upper_bounds - xr )( i ) : OsqpEigen::INFTY;
    }
  }

  for( int t = 0; t < T; ++t )
  {
    const auto ur = U.col( t );
    for( int i = 0; i < nu; ++i )
    {
      int idx      = cb_off + t * nu + i;
      qp.lb( idx ) = p.input_lower_bounds ? ( *p.input_lower_bounds - ur )( i ) : -OsqpEigen::INFTY;
      qp.ub( idx ) = p.input_upper_bounds ? ( *p.input_upper_bounds - ur )( i ) : OsqpEigen::INFTY;
    }
  }

  if( !qp.H.isCompressed() )
    std::cerr << "[Warning] Hessian matrix is not compressed\n";
  if( !qp.A.isCompressed() )
    std::cerr << "[Warning] Constraint matrix is not compressed\n";
  if( !qp.g.allFinite() )
    std::cerr << "[Warning] Gradient vector contains non-finite values\n";
  if( !qp.lb.allFinite() || !qp.ub.allFinite() )
    std::cerr << "[Warning] Bound vectors contain non-finite values\n";

  solver->updateHessianMatrix( qp.H );
  solver->updateGradient( qp.g );
  solver->updateLinearConstraintsMatrix( qp.A );
  if( !solver->updateBounds( qp.lb, qp.ub ) )
    throw std::runtime_error( "OSQP bounds update failed" );
}

/* ---------- main entry ------------------------------------------- */
inline void
OSQPCollocation::solve( OCP& problem )
{
  using clk = std::chrono::high_resolution_clock;

  const auto tic      = clk::now();
  const auto deadline = ( max_ms > 0.0 ) ? tic + std::chrono::milliseconds( static_cast<int>( max_ms ) )
                                         : clk::time_point::max(); // “infinite” if max_ms ≤ 0

  /* dimensions & warm-start guesses ----------------------------- */
  if( !structure_ready )
    prepare_structure( problem );

  auto& X = problem.best_states;
  auto& U = problem.best_controls;
  if( X.rows() != nx || X.cols() != T + 1 )
    X.setZero( nx, T + 1 );
  if( U.rows() != nu || U.cols() != T )
    U.setZero( nu, T );

  problem.initialize_problem();
  X          = problem.initial_states;
  U          = problem.initial_controls;
  X.col( 0 ) = problem.initial_state; // x₀ fixed

  constexpr double reg = 1e-6;

  cache_valid = false; // invalidate caches for fresh solve

  /* --------------- outer SQP loop ------------------------------ */
  for( int it = 0; it < max_iter; ++it )
  {
    if( clk::now() >= deadline )
    {
      if( debug )
        std::cerr << "OSQP Collocation stopped by time-out after " << it << " iterations\n";
      break; // keep best-so-far
    }

    assemble_values( problem, X, U, reg );

    if( solver->solveProblem() != OsqpEigen::ErrorExitFlag::NoError )
      throw std::runtime_error( "OSQP failed at iter " + std::to_string( it ) );

    const Eigen::VectorXd delta = solver->getSolution();

    /* apply step --------------------------------------------- */
    double step_norm2 = 0.0;
    for( int t = 0; t < T; ++t )
    {
      X.col( t + 1 ) += delta.segment( t * nx, nx );
      U.col( t )     += delta.segment( T * nx + t * nu, nu );
      step_norm2     += delta.segment( t * nx, nx ).squaredNorm() + delta.segment( T * nx + t * nu, nu ).squaredNorm();
    }

    if( std::sqrt( step_norm2 ) < tolerance )
    {
      if( debug )
        std::cerr << "OSQP Collocation  converged in " << it + 1 << " SQP steps\n";
      break;
    }
  }

  problem.best_cost = problem.objective_function( X, U );

  if( debug )
  {
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>( clk::now() - tic ).count();
    std::cerr << "OSQP-collocation finished in " << ms << " ms,  cost = " << problem.best_cost << '\n';
  }
}


} // namespace mas
