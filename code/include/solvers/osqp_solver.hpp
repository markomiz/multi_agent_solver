#pragma once

#include <cmath>

#include <algorithm>
#include <stdexcept>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "integrator.hpp"    // must provide integrate_horizon() and integrate_rk4()
#include "ocp.hpp"           // defines OCP, State, Control, StateTrajectory, ControlTrajectory, etc.
#include "solver_output.hpp" // defines SolverOutput
#include <OsqpEigen/OsqpEigen.h>

//====================================================================
// 1. QP Data Structure and Helper Functions in a Namespace
//====================================================================

namespace osqp_solver_ns
{

// QPData structure collects the matrices and vectors for the QP
struct QPData
{
  Eigen::SparseMatrix<double> H;        // Hessian matrix (P)
  Eigen::VectorXd             gradient; // Linear cost term (q)
  Eigen::SparseMatrix<double> A;        // Constraint matrix
  Eigen::VectorXd             lb;       // Lower bounds for constraints
  Eigen::VectorXd             ub;       // Upper bounds for constraints
};

//---------------------------------------------------------------------
// construct_hessian:
// Build the Hessian matrix for the QP by stacking the (diagonal)
// contributions from the state and control cost Hessians.
// For each time step, we use the cost Hessian computed (via finite differences)
// from the stage cost. Negative curvature is clamped to 0 and a small
// regularization "reg" is added.
// qp_dim = (T+1)*n_x + T*n_u.
//---------------------------------------------------------------------
inline Eigen::SparseMatrix<double>
construct_hessian( const OCP &problem, const StateTrajectory &states, const ControlTrajectory &controls, double reg )
{
  int n_x    = problem.state_dim;
  int n_u    = problem.control_dim;
  int T      = problem.horizon_steps;
  int qp_dim = ( T + 1 ) * n_x + T * n_u;

  std::vector<Eigen::Triplet<double>> triplets;
  // Reserve space (estimate): each block contributes one entry per variable.
  triplets.reserve( ( T + 1 ) * n_x + T * n_u );

  // --- State Blocks ---
  for( int t = 0; t < T + 1; ++t )
  {
    // For terminal time t==T, if a dedicated terminal cost is not available, use the stage cost Hessian.
    Eigen::MatrixXd Q;
    if( t == T )
      Q = problem.cost_state_hessian( problem.stage_cost, states.col( t ), controls.col( std::min( t, T - 1 ) ) );
    else
      Q = problem.cost_state_hessian( problem.stage_cost, states.col( t ), controls.col( t ) );
    for( int i = 0; i < n_x; ++i )
    {
      int    idx   = t * n_x + i;
      double diag  = Q( i, i );
      double value = std::max( diag, 0.0 ) + reg;
      triplets.push_back( Eigen::Triplet<double>( idx, idx, value ) );
    }
  }

  // --- Control Blocks ---
  for( int t = 0; t < T; ++t )
  {
    Eigen::MatrixXd R = problem.cost_control_hessian( problem.stage_cost, states.col( t ), controls.col( t ) );
    for( int i = 0; i < n_u; ++i )
    {
      int    idx   = ( T + 1 ) * n_x + t * n_u + i;
      double diag  = R( i, i );
      double value = std::max( diag, 0.0 ) + reg;
      triplets.push_back( Eigen::Triplet<double>( idx, idx, value ) );
    }
  }

  Eigen::SparseMatrix<double> H( qp_dim, qp_dim );
  H.setFromTriplets( triplets.begin(), triplets.end() );
  return H;
}

//---------------------------------------------------------------------
// construct_gradient:
// Build the gradient vector q by stacking the cost gradients for each
// state and control. The state gradients are computed using cost_state_gradient,
// and similarly for controls.
//---------------------------------------------------------------------
inline Eigen::VectorXd
construct_gradient( const OCP &problem, const StateTrajectory &states, const ControlTrajectory &controls )
{
  int n_x    = problem.state_dim;
  int n_u    = problem.control_dim;
  int T      = problem.horizon_steps;
  int qp_dim = ( T + 1 ) * n_x + T * n_u;

  Eigen::VectorXd q = Eigen::VectorXd::Zero( qp_dim );
  // State gradients:
  for( int t = 0; t < T + 1; ++t )
  {
    Eigen::VectorXd g;
    if( t == T )
      g = problem.cost_state_gradient( problem.stage_cost, states.col( t ), controls.col( std::min( t, T - 1 ) ) );
    else
      g = problem.cost_state_gradient( problem.stage_cost, states.col( t ), controls.col( t ) );
    q.segment( t * n_x, n_x ) = g;
  }
  // Control gradients:
  for( int t = 0; t < T; ++t )
  {
    Eigen::VectorXd g                           = problem.cost_control_gradient( problem.stage_cost, states.col( t ), controls.col( t ) );
    q.segment( ( T + 1 ) * n_x + t * n_u, n_u ) = g;
  }
  return q;
}

//---------------------------------------------------------------------
// construct_constraints:
// Build the constraint matrix A for the dynamics constraints only.
// For each time step t = 0,...,T-1, we enforce:
//   x_{t+1} - A_t*x_t - B_t*u_t = 0,
// where A_t and B_t are the dynamics Jacobians at time t.
// The resulting A is of size (T*n_x) x qp_dim.
//---------------------------------------------------------------------
inline Eigen::SparseMatrix<double>
construct_constraints( const OCP &problem, const StateTrajectory &states, const ControlTrajectory &controls )
{
  int n_x             = problem.state_dim;
  int n_u             = problem.control_dim;
  int T               = problem.horizon_steps;
  int num_constraints = T * n_x;
  int qp_dim          = ( T + 1 ) * n_x + T * n_u;

  std::vector<Eigen::Triplet<double>> triplets;
  // Reserve an estimated number of nonzeros.
  triplets.reserve( T * ( n_x + n_x * n_x + n_x * n_u ) );

  for( int t = 0; t < T; ++t )
  {
    int row_offset = t * n_x;
    // For x_{t+1}: coefficient +I.
    for( int i = 0; i < n_x; ++i )
    {
      int col = ( t + 1 ) * n_x + i;
      triplets.push_back( Eigen::Triplet<double>( row_offset + i, col, 1.0 ) );
    }
    // For x_t: coefficient -A_t.
    Eigen::MatrixXd A_mat = problem.dynamics_state_jacobian( problem.dynamics, states.col( t ), controls.col( t ) );
    for( int i = 0; i < n_x; ++i )
    {
      for( int j = 0; j < n_x; ++j )
      {
        int col = t * n_x + j;
        triplets.push_back( Eigen::Triplet<double>( row_offset + i, col, -A_mat( i, j ) ) );
      }
    }
    // For u_t: coefficient -B_t.
    Eigen::MatrixXd B_mat = problem.dynamics_control_jacobian( problem.dynamics, states.col( t ), controls.col( t ) );
    for( int i = 0; i < n_x; ++i )
    {
      for( int j = 0; j < n_u; ++j )
      {
        int col = ( T + 1 ) * n_x + t * n_u + j;
        triplets.push_back( Eigen::Triplet<double>( row_offset + i, col, -B_mat( i, j ) ) );
      }
    }
  }

  Eigen::SparseMatrix<double> A( num_constraints, qp_dim );
  A.setFromTriplets( triplets.begin(), triplets.end() );
  return A;
}

//---------------------------------------------------------------------
// construct_bounds:
// For the dynamics constraints, the equality is enforced by setting
// l = u = 0. (If there is a constant offset c, it would appear here.)
//---------------------------------------------------------------------
inline std::pair<Eigen::VectorXd, Eigen::VectorXd>
construct_bounds( const OCP &problem )
{
  int             n_x             = problem.state_dim;
  int             T               = problem.horizon_steps;
  int             num_constraints = T * n_x;
  Eigen::VectorXd lb              = Eigen::VectorXd::Zero( num_constraints );
  Eigen::VectorXd ub              = Eigen::VectorXd::Zero( num_constraints );
  return { lb, ub };
}

//---------------------------------------------------------------------
// constructQPData:
// Combines the Hessian, gradient, constraints, and bounds into a QPData struct.
// reg is the regularization parameter for the Hessian.
//---------------------------------------------------------------------
inline QPData
constructQPData( const OCP &problem, const StateTrajectory &states, const ControlTrajectory &controls, double reg = 0.0 )
{
  QPData qpData;
  qpData.H        = construct_hessian( problem, states, controls, reg );
  qpData.gradient = construct_gradient( problem, states, controls );
  qpData.A        = construct_constraints( problem, states, controls );
  auto bounds     = construct_bounds( problem );
  qpData.lb       = bounds.first;
  qpData.ub       = bounds.second;
  return qpData;
}

} // end namespace osqp_solver_ns

//====================================================================
// 2. OSQP Solver Function
//====================================================================
//
// This function sets up an OSQP problem and then iteratively updates it.
// At each iteration, it:
//  - Recomputes the QP data (objective and constraints) based on the current
//    state and control trajectories,
//  - Attempts to update the Hessian with adaptive regularization if necessary,
//  - Solves the QP,
//  - Extracts the new control trajectory,
//  - Propagates the state trajectory via forward integration,
//  - Checks for convergence.
//
// The function returns a SolverOutput that contains the final cost,
// state trajectory, and control trajectory.
//
inline SolverOutput
osqp_solver( const OCP &problem, int max_iterations = 100, double tolerance = 1e-5 )
{
  using namespace osqp_solver_ns;
  int T           = problem.horizon_steps;
  int n_x         = problem.state_dim;
  int n_u         = problem.control_dim;
  int numStates   = T + 1;
  int numControls = T;
  int qp_dim      = numStates * n_x + numControls * n_u;

  // Initialize state and control trajectories.
  StateTrajectory   states   = StateTrajectory::Zero( n_x, numStates );
  ControlTrajectory controls = ControlTrajectory::Zero( n_u, numControls );
  states.col( 0 )            = problem.initial_state;
  // Generate an initial trajectory (e.g. with zero controls):
  states      = integrate_horizon( problem.initial_state, controls, problem.dt, problem.dynamics, integrate_rk4 );
  double cost = problem.objective_function( states, controls );

  // Create OSQP solver instance.
  std::unique_ptr<OsqpEigen::Solver> solver      = std::make_unique<OsqpEigen::Solver>();
  double                             initial_reg = 0.0;
  QPData                             qpData      = constructQPData( problem, states, controls, initial_reg );

  solver->data()->setNumberOfVariables( qp_dim );
  solver->data()->setNumberOfConstraints( qpData.A.rows() );

  if( !solver->data()->setHessianMatrix( qpData.H ) )
    throw std::runtime_error( "Failed to set Hessian." );
  if( !solver->data()->setGradient( qpData.gradient ) )
    throw std::runtime_error( "Failed to set gradient." );
  if( !solver->data()->setLinearConstraintsMatrix( qpData.A ) )
    throw std::runtime_error( "Failed to set constraint matrix." );
  if( !solver->data()->setLowerBound( qpData.lb ) )
    throw std::runtime_error( "Failed to set lower bounds." );
  if( !solver->data()->setUpperBound( qpData.ub ) )
    throw std::runtime_error( "Failed to set upper bounds." );

  solver->settings()->setWarmStart( true );
  solver->settings()->setVerbosity( false );
  if( !solver->initSolver() )
    throw std::runtime_error( "Failed to initialize OSQP solver->" );

  if( !solver->isInitialized() )
    throw std::runtime_error( "Failed to initialize OSQP solver->" );
  // Prepare parameters for the Armijo line search.
  std::map<std::string, double> ls_parameters = {
    { "initial_step_size",  1.0 },
    {              "beta",  0.5 },
    {                "c1", 1e-6 }
  };

  // Main iterative loop.
  for( int iter = 0; iter < max_iterations; ++iter )
  {
    // Adaptive Hessian regularization.
    double                      current_reg     = 0.0;
    const int                   max_attempts    = 10;
    bool                        hessian_updated = false;
    Eigen::SparseMatrix<double> H_temp;
    for( int attempt = 0; attempt < max_attempts; ++attempt )
    {
      H_temp = construct_hessian( problem, states, controls, current_reg );
      if( solver->updateHessianMatrix( H_temp ) )
      {
        hessian_updated = true;
        break;
      }
      current_reg = ( current_reg == 0.0 ) ? 1e-6 : current_reg * 10;
    }
    if( !hessian_updated )
      throw std::runtime_error( "Failed to update Hessian even with adaptive regularization." );

    qpData = constructQPData( problem, states, controls, current_reg );
    if( !solver->updateGradient( qpData.gradient ) )
      throw std::runtime_error( "Failed to update gradient." );
    if( !solver->updateLinearConstraintsMatrix( qpData.A ) )
      throw std::runtime_error( "Failed to update constraint matrix." );
    if( !solver->updateLowerBound( qpData.lb ) )
      throw std::runtime_error( "Failed to update lower bounds." );
    if( !solver->updateUpperBound( qpData.ub ) )
      throw std::runtime_error( "Failed to update upper bounds." );

    // Solve the QP.
    if( solver->solveProblem() != OsqpEigen::ErrorExitFlag::NoError )
      throw std::runtime_error( "OSQP solver failed." );

    // Extract the candidate controls from the solution.
    Eigen::VectorXd solution = Eigen::VectorXd( solver->getSolution() ); // Force copy

    ControlTrajectory u_candidate( n_u, numControls );
    for( int t = 0; t < numControls; ++t )
    {
      u_candidate.col( t ) = solution.segment( numStates * n_x + t * n_u, n_u );
    }

    // Compute the update direction (difference between candidate and current controls).
    ControlTrajectory d_u = u_candidate - controls;

    // Use the Armijo line search to determine the step length.
    double alpha = backtracking_line_search( problem.initial_state, controls, d_u, problem.dynamics, problem.objective_function, problem.dt,
                                             ls_parameters );

    // Update controls with damping.
    ControlTrajectory u_new = controls + alpha * d_u;

    // Forward integrate to obtain the new state trajectory.
    StateTrajectory new_states = integrate_horizon( problem.initial_state, u_new, problem.dt, problem.dynamics, integrate_rk4 );
    double          new_cost   = problem.objective_function( new_states, u_new );

    // Optionally, log the step.
    // std::cerr << "Iteration " << iter << ": cost = " << new_cost << ", step size alpha = " << alpha << std::endl;

    // Check for convergence.
    if( std::abs( cost - new_cost ) < tolerance )
    {
      controls = u_new;
      states   = new_states;
      cost     = new_cost;
      break;
    }
    else if( new_cost < cost )
    {
      controls = u_new;
      states   = new_states;
      cost     = new_cost;
    }
    else
      break;
  }
  solver->clearSolverVariables();
  solver->clearSolver();

  solver = nullptr;

  SolverOutput sol;
  sol.cost       = cost;
  sol.trajectory = states;
  sol.controls   = controls;
  return sol;
}