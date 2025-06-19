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
    solver_{ std::make_unique<OsqpEigen::Solver>() }
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
  struct qp_t
  {
    Eigen::SparseMatrix<double> H;      // Hessian
    Eigen::VectorXd             g;      // gradient
    Eigen::SparseMatrix<double> A;      // constraint matrix
    Eigen::VectorXd             lb, ub; // bounds
  } qp_;

  /* OSQP */
  std::unique_ptr<OsqpEigen::Solver> solver_;
  bool                               structure_ready_{ false };

  /* problem dimensions */
  int T_{}, nx_{}, nu_{}, qp_dim_{};
  int n_dyn_{}, n_state_bnd_{}, n_ctrl_bnd_{};

  /* index maps into value arrays (CSC order) */
  std::vector<Eigen::Index> H_idx_;     // all mutable nnz in H
  std::vector<Eigen::Index> A_dyn_idx_; // dynamics rows only

  /* settings */
  int    max_iter_{ 20 };
  double sqp_tol_{ 1e-4 };
  bool   debug_{ false };
};

/* =================================================================== */
/*                    Implementation details                            */
/* =================================================================== */
inline void
OSQPCollocation::set_params( const SolverParams& p )
{
  max_iter_ = static_cast<int>( p.at( "max_iterations" ) );
  sqp_tol_  = p.at( "tolerance" );
  debug_    = p.count( "debug" ) && p.at( "debug" ) > 0.5;

  solver_->settings()->setVerbosity( debug_ );
  solver_->settings()->setWarmStart( true );
  solver_->settings()->setAdaptiveRho( true );
  solver_->settings()->setScaling( 10 );
  solver_->settings()->setPolish( true );
  solver_->settings()->setMaxIteration( static_cast<int>( p.count( "osqp_max_iter" ) ? p.at( "osqp_max_iter" ) : 4000 ) );
  solver_->settings()->setAbsoluteTolerance( p.count( "osqp_abs_tol" ) ? p.at( "osqp_abs_tol" ) : 1e-4 );
  solver_->settings()->setRelativeTolerance( p.count( "osqp_rel_tol" ) ? p.at( "osqp_rel_tol" ) : 1e-4 );
}

/* ---------- phase 1: build fixed sparsity once -------------------- */
/* ------------------------------------------------------------------ */
/*  Phase-1: build immutable sparsity pattern and initialise OSQP     */
/* ------------------------------------------------------------------ */
inline void
OSQPCollocation::prepare_structure( const OCP& p )
{
  /* ---------- basic dimensions ---------------------------------- */
  T_  = p.horizon_steps;
  nx_ = p.state_dim;
  nu_ = p.control_dim;

  qp_dim_      = T_ * nx_ + T_ * nu_; // decision-var length
  n_dyn_       = T_ * nx_;            // collocation equalities
  n_state_bnd_ = T_ * nx_;            // state bound rows
  n_ctrl_bnd_  = T_ * nu_;            // input bound rows

  /* ---------- allocate QP buffers (shape only) ------------------ */
  qp_.H.resize( qp_dim_, qp_dim_ );
  qp_.g.resize( qp_dim_ );
  qp_.A.resize( n_dyn_ + n_state_bnd_ + n_ctrl_bnd_, qp_dim_ );
  qp_.lb.resize( qp_.A.rows() );
  qp_.ub.resize( qp_.A.rows() );

  /* initialise them to zero so we can pass l-values to OSQP ------- */
  qp_.g.setZero();
  qp_.lb.setZero();
  qp_.ub.setZero();

  /* ---------- build Hessian sparsity pattern -------------------- */
  std::vector<Eigen::Triplet<double>> H_tpl;
  H_tpl.reserve( qp_dim_ ); // diagonal reg

  /* Q-blocks (δx) */
  for( int t = 1; t <= T_; ++t )
    for( int i = 0; i < nx_; ++i )
      H_tpl.emplace_back( id_state( t, i, nx_ ), id_state( t, i, nx_ ), 0.0 );

  /* R-blocks (δu) */
  for( int t = 0; t < T_; ++t )
    for( int i = 0; i < nu_; ++i )
      H_tpl.emplace_back( id_control( t, nx_, nu_, T_ ) + i, id_control( t, nx_, nu_, T_ ) + i, 0.0 );

  /* small reg on every variable */
  for( int k = 0; k < qp_dim_; ++k )
    H_tpl.emplace_back( k, k, 0.0 );

  qp_.H.setFromTriplets( H_tpl.begin(), H_tpl.end() );
  qp_.H.makeCompressed();

  /* keep CSC indices of *all* mutable Hessian entries */
  H_idx_.assign( qp_.H.innerIndexPtr(), qp_.H.innerIndexPtr() + qp_.H.nonZeros() );

  /* ---------- build constraint matrix sparsity pattern ---------- */
  std::vector<Eigen::Triplet<double>> A_tpl;
  A_tpl.reserve( n_dyn_ * ( 2 * nx_ + 2 * nu_ ) + n_state_bnd_ + n_ctrl_bnd_ );

  /* dynamics rows (trapezoidal) */
  for( int t = 0; t < T_; ++t )
  {
    const int row0 = t * nx_;
    for( int i = 0; i < nx_; ++i )
    {
      /* δx_{t+1} */
      for( int j = 0; j < nx_; ++j )
        A_tpl.emplace_back( row0 + i, id_state( t + 1, j, nx_ ), 0.0 );

      /* δx_t */
      if( t > 0 )
        for( int j = 0; j < nx_; ++j )
          A_tpl.emplace_back( row0 + i, id_state( t, j, nx_ ), 0.0 );

      /* δu_t */
      for( int j = 0; j < nu_; ++j )
        A_tpl.emplace_back( row0 + i, id_control( t, nx_, nu_, T_ ) + j, 0.0 );

      /* δu_{t+1} */
      if( t + 1 < T_ )
        for( int j = 0; j < nu_; ++j )
          A_tpl.emplace_back( row0 + i, id_control( t + 1, nx_, nu_, T_ ) + j, 0.0 );
    }
  }

  /* bound identity rows */
  for( int r = n_dyn_; r < qp_.A.rows(); ++r )
    A_tpl.emplace_back( r, r - n_dyn_, 1.0 );

  qp_.A.setFromTriplets( A_tpl.begin(), A_tpl.end() );
  qp_.A.makeCompressed();

  /* store CSC offsets of the dynamics entries (skip bound rows) */
  for( int col = 0; col < qp_.A.outerSize(); ++col )
  {
    auto start = qp_.A.outerIndexPtr()[col];
    auto end   = qp_.A.outerIndexPtr()[col + 1];
    for( Eigen::Index idx = start; idx < end; ++idx )
      if( qp_.A.innerIndexPtr()[idx] < n_dyn_ )
        A_dyn_idx_.push_back( idx );
  }

  /* ---------- hand everything to OSQP once ---------------------- */
  solver_->data()->setNumberOfVariables( qp_dim_ );
  solver_->data()->setNumberOfConstraints( qp_.A.rows() );

  if( !solver_->data()->setHessianMatrix( qp_.H ) || !solver_->data()->setGradient( qp_.g ) ||              // l-value
      !solver_->data()->setLinearConstraintsMatrix( qp_.A ) || !solver_->data()->setLowerBound( qp_.lb ) || // l-value
      !solver_->data()->setUpperBound( qp_.ub ) ||                                                          // l-value
      !solver_->initSolver() )
  {
    throw std::runtime_error( "OSQP initialisation failed" );
  }

  structure_ready_ = true;
}

/* ---------- phase 2: overwrite numeric values -------------------- */
inline void
OSQPCollocation::assemble_values( const OCP& p, const StateTrajectory& X, const ControlTrajectory& U, double reg )
{
  /* ---------- gradient (stage cost) ----------------------------- */
  qp_.g.setZero();
  for( int t = 1; t <= T_; ++t )
    qp_.g.segment( id_state( t, 0, nx_ ), nx_ ) = p.cost_state_gradient( p.stage_cost, X.col( t ), U.col( std::min( t, T_ - 1 ) ), t );
  for( int t = 0; t < T_; ++t )
    qp_.g.segment( id_control( t, nx_, nu_, T_ ), nu_ ) = p.cost_control_gradient( p.stage_cost, X.col( t ), U.col( t ), t );

  /* ---------- Hessian values ------------------------------------ */
  double*     h_val = qp_.H.valuePtr();
  std::size_t kH    = 0;

  /* Q – state terms */
  for( int t = 1; t <= T_; ++t )
  {
    const auto Q = p.cost_state_hessian( p.stage_cost, X.col( t ), U.col( std::min( t, T_ - 1 ) ), t );
    for( int i = 0; i < nx_; ++i )
      h_val[H_idx_[kH++]] = Q( i, i ) + reg;
  }
  /* R – control terms */
  for( int t = 0; t < T_; ++t )
  {
    const auto R = p.cost_control_hessian( p.stage_cost, X.col( t ), U.col( t ), t );
    for( int i = 0; i < nu_; ++i )
      h_val[H_idx_[kH++]] = R( i, i ) + reg;
  }
  /* reg anywhere else */
  for( ; kH < H_idx_.size(); ++kH )
    h_val[H_idx_[kH]] = reg;

  /* ---------- dynamics rows ------------------------------------- */
  double*     a_val = qp_.A.valuePtr();
  std::size_t kA    = 0;

  for( int t = 0; t < T_; ++t )
  {
    const int row0 = t * nx_;

    const auto x_t   = X.col( t );
    const auto u_t   = U.col( t );
    const auto x_tp1 = X.col( t + 1 );
    const auto u_tp1 = ( t + 1 < T_ ) ? U.col( t + 1 ) : U.col( T_ - 1 );

    const auto Fx_t   = p.dynamics_state_jacobian( p.dynamics, x_t, u_t );
    const auto Fu_t   = p.dynamics_control_jacobian( p.dynamics, x_t, u_t );
    const auto Fx_tp1 = p.dynamics_state_jacobian( p.dynamics, x_tp1, u_tp1 );
    const auto Fu_tp1 = p.dynamics_control_jacobian( p.dynamics, x_tp1, u_tp1 );

    const auto f_t   = p.dynamics( x_t, u_t );
    const auto f_tp1 = p.dynamics( x_tp1, u_tp1 );

    const auto defect           = x_tp1 - x_t - 0.5 * p.dt * ( f_t + f_tp1 );
    qp_.lb.segment( row0, nx_ ) = -defect;
    qp_.ub.segment( row0, nx_ ) = -defect;

    for( int i = 0; i < nx_; ++i )
    {
      /* δx_{t+1} */
      for( int j = 0; j < nx_; ++j )
        qp_.A.coeffRef( row0 + i, id_state( t + 1, j, nx_ ) ) = ( i == j ? 1.0 : 0.0 ) - 0.5 * p.dt * Fx_tp1( i, j );

      if( t > 0 ) /* δx_t */
        for( int j = 0; j < nx_; ++j )
          qp_.A.coeffRef( row0 + i, id_state( t, j, nx_ ) ) = -( i == j ? 1.0 : 0.0 ) - 0.5 * p.dt * Fx_t( i, j );

      /* δu_t */
      for( int j = 0; j < nu_; ++j )
        qp_.A.coeffRef( row0 + i, id_control( t, nx_, nu_, T_ ) + j ) = -0.5 * p.dt * Fu_t( i, j );

      if( t + 1 < T_ ) /* δu_{t+1} */
        for( int j = 0; j < nu_; ++j )
          qp_.A.coeffRef( row0 + i, id_control( t + 1, nx_, nu_, T_ ) + j ) = -0.5 * p.dt * Fu_tp1( i, j );
    }
  }
  assert( kA == A_dyn_idx_.size() );

  /* ---------- bound rows ---------------------------------------- */
  const int sb_off = n_dyn_;
  const int cb_off = n_dyn_ + n_state_bnd_;

  for( int t = 1; t <= T_; ++t )
  {
    const auto xr = X.col( t );
    for( int i = 0; i < nx_; ++i )
    {
      int idx       = sb_off + ( t - 1 ) * nx_ + i;
      qp_.lb( idx ) = p.state_lower_bounds ? ( *p.state_lower_bounds - xr )( i ) : -OsqpEigen::INFTY;
      qp_.ub( idx ) = p.state_upper_bounds ? ( *p.state_upper_bounds - xr )( i ) : OsqpEigen::INFTY;
    }
  }
  for( int t = 0; t < T_; ++t )
  {
    const auto ur = U.col( t );
    for( int i = 0; i < nu_; ++i )
    {
      int idx       = cb_off + t * nu_ + i;
      qp_.lb( idx ) = p.input_lower_bounds ? ( *p.input_lower_bounds - ur )( i ) : -OsqpEigen::INFTY;
      qp_.ub( idx ) = p.input_upper_bounds ? ( *p.input_upper_bounds - ur )( i ) : OsqpEigen::INFTY;
    }
  }

  /* ---------- push numbers into OSQP ---------------------------- */
  solver_->updateHessianMatrix( qp_.H ); // value update only
  solver_->updateGradient( qp_.g );
  solver_->updateLinearConstraintsMatrix( qp_.A );
  /* single atomic bounds update – prevents intermediate l>u */
  if( !solver_->updateBounds( qp_.lb, qp_.ub ) )
    throw std::runtime_error( "OSQP bounds update failed" );
}

/* ---------- main entry ------------------------------------------- */
inline void
OSQPCollocation::solve( OCP& problem )
{
  using clk      = std::chrono::high_resolution_clock;
  const auto tic = clk::now();

  /* dimensions & warm-start guesses ------------------------------ */
  if( !structure_ready_ )
    prepare_structure( problem );

  auto& X = problem.best_states;
  auto& U = problem.best_controls;
  if( X.rows() != nx_ || X.cols() != T_ + 1 )
    X.setZero( nx_, T_ + 1 );
  if( U.rows() != nu_ || U.cols() != T_ )
    U.setZero( nu_, T_ );

  problem.initialize_problem();
  X          = problem.initial_states;
  U          = problem.initial_controls;
  X.col( 0 ) = problem.initial_state; // x0 fixed

  constexpr double reg = 1e-6;

  /* ------ outer SQP loop --------------------------------------- */
  for( int it = 0; it < max_iter_; ++it )
  {
    assemble_values( problem, X, U, reg );

    if( solver_->solveProblem() != OsqpEigen::ErrorExitFlag::NoError )
      throw std::runtime_error( "OSQP failed at iter " + std::to_string( it ) );

    const Eigen::VectorXd delta = solver_->getSolution();

    /* apply step */
    double step_norm2 = 0.0;
    for( int t = 0; t < T_; ++t )
    {
      X.col( t + 1 ) += delta.segment( t * nx_, nx_ );
      U.col( t )     += delta.segment( T_ * nx_ + t * nu_, nu_ );
      step_norm2     += delta.segment( t * nx_, nx_ ).squaredNorm() + delta.segment( T_ * nx_ + t * nu_, nu_ ).squaredNorm();
    }

    if( std::sqrt( step_norm2 ) < sqp_tol_ )
    {
      if( debug_ )
        std::cout << "converged in " << it + 1 << " SQP steps\n";
      break;
    }
  }

  problem.best_cost = problem.objective_function( X, U );

  if( debug_ )
  {
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>( clk::now() - tic ).count();
    std::cout << "OSQP-collocation finished in " << ms << " ms,  cost=" << problem.best_cost << '\n';
  }
}

} // namespace mas
