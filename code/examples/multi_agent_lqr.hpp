#pragma once
#include <chrono>
#include <iostream>
#include <map>
#include <memory>
#include <unordered_map>

#include "Eigen/Dense"

#include "multi_agent_aggregator.hpp"
#include "ocp.hpp"
#include "solvers/cgd.hpp"
#include "solvers/ilqr.hpp"
#include "solvers/osqp_solver.hpp"
#include "types.hpp"

//---------------------------------------------------------------------
// A helper function that creates a linear LQR OCP for a single agent.
// For simplicity, we use identity dynamics and quadratic costs.
//---------------------------------------------------------------------
OCP
create_linear_lqr_ocp( int state_dim, int control_dim, double dt, int horizon_steps )
{
  OCP ocp;
  ocp.state_dim     = state_dim;
  ocp.control_dim   = control_dim;
  ocp.dt            = dt;
  ocp.horizon_steps = horizon_steps;

  // Random initial state.
  ocp.initial_state = Eigen::VectorXd::Random( state_dim );

  // Define linear dynamics: x_{k+1} = A*x_k + B*u_k, with A and B as identity.
  Eigen::MatrixXd A = Eigen::MatrixXd::Identity( state_dim, state_dim );
  Eigen::MatrixXd B = Eigen::MatrixXd::Identity( state_dim, control_dim );

  ocp.dynamics = [A, B]( const State& x, const Control& u ) { return A * x + B * u; };

  // Quadratic cost: l(x,u) = xᵀQx + uᵀRu.
  Eigen::MatrixXd Q = Eigen::MatrixXd::Identity( state_dim, state_dim );
  Eigen::MatrixXd R = Eigen::MatrixXd::Identity( control_dim, control_dim );
  ocp.stage_cost    = [Q, R]( const State& x, const Control& u ) -> double {
    return ( x.transpose() * Q * x ).value() + ( u.transpose() * R * u ).value();
  };
  ocp.terminal_cost = []( const State& ) -> double { return 0.0; };

  // Set analytic derivatives.
  // Dynamics derivatives.
  ocp.dynamics_state_jacobian   = [A]( const MotionModel&, const State&, const Control& ) -> Eigen::MatrixXd { return A; };
  ocp.dynamics_control_jacobian = [B]( const MotionModel&, const State&, const Control& ) -> Eigen::MatrixXd { return B; };
  // Cost derivatives.
  ocp.cost_state_gradient   = [Q]( const StageCostFunction&, const State& x, const Control& ) -> Eigen::VectorXd { return 2.0 * Q * x; };
  ocp.cost_control_gradient = [R]( const StageCostFunction&, const State&, const Control& u ) -> Eigen::VectorXd { return 2.0 * R * u; };
  ocp.cost_state_hessian    = [Q]( const StageCostFunction&, const State&, const Control& ) -> Eigen::MatrixXd { return 2.0 * Q; };
  ocp.cost_control_hessian  = [R]( const StageCostFunction&, const State&, const Control& ) -> Eigen::MatrixXd { return 2.0 * R; };
  ocp.cost_cross_term       = [state_dim, control_dim]( const StageCostFunction&, const State&, const Control& ) -> Eigen::MatrixXd {
    // With a separable cost, cross derivatives are zero.
    return Eigen::MatrixXd::Zero( state_dim, control_dim );
  };

  ocp.initialize_problem();
  ocp.verify_problem();
  return ocp;
}

//---------------------------------------------------------------------
// Multi-agent LQR example using 4 agents.
// This example uses the MultiAgentAggregator class to create a global OCP,
// solves it using the iLQR solver, and then extracts per-agent solutions.
//---------------------------------------------------------------------

void
multi_agent_lqr_example()
{
  const int    num_agents    = 10;
  const int    state_dim     = 4;
  const int    control_dim   = 4;
  const double dt            = 0.1;
  const int    horizon_steps = 10;

  // Create an aggregator for multi-agent problems
  MultiAgentAggregator aggregator;

  // Create LQR OCP for each agent
  for( int i = 0; i < num_agents; ++i )
  {
    std::shared_ptr<OCP> agent_ocp = std::make_shared<OCP>( create_linear_lqr_ocp( state_dim, control_dim, dt, horizon_steps ) );
    aggregator.agent_ocps[i]       = agent_ocp;
  }

  // Compute offsets for multi-agent system
  aggregator.compute_offsets();

  SolverParams params;
  params["max_iterations"] = 100;
  params["tolerance"]      = 1e-5;

  // Solve in centralized mode
  auto   start                 = std::chrono::high_resolution_clock::now();
  double central_ilqr_cost     = aggregator.solve_centralized( ilqr_solver, params );
  auto   end                   = std::chrono::high_resolution_clock::now();
  double centralized_ilqr_time = std::chrono::duration<double, std::milli>( end - start ).count();
  aggregator.reset();

  start                        = std::chrono::high_resolution_clock::now();
  double central_osqp_cost     = aggregator.solve_centralized( osqp_solver, params );
  end                          = std::chrono::high_resolution_clock::now();
  double centralized_osqp_time = std::chrono::duration<double, std::milli>( end - start ).count();
  aggregator.reset();

  start                       = std::chrono::high_resolution_clock::now();
  double central_cgd_cost     = aggregator.solve_centralized( cgd_solver, params );
  end                         = std::chrono::high_resolution_clock::now();
  double centralized_cgd_time = std::chrono::duration<double, std::milli>( end - start ).count();
  aggregator.reset();

  // Solve in decentralized mode
  start                          = std::chrono::high_resolution_clock::now();
  double decentral_ilqr_cost     = aggregator.solve_decentralized_line_search( ilqr_solver, 1, params );
  end                            = std::chrono::high_resolution_clock::now();
  double decentralized_ilqr_time = std::chrono::duration<double, std::milli>( end - start ).count();
  aggregator.reset();

  start                          = std::chrono::high_resolution_clock::now();
  double decentral_osqp_cost     = aggregator.solve_decentralized_line_search( osqp_solver, 1, params );
  end                            = std::chrono::high_resolution_clock::now();
  double decentralized_osqp_time = std::chrono::duration<double, std::milli>( end - start ).count();
  aggregator.reset();

  start                         = std::chrono::high_resolution_clock::now();
  double decentral_cgd_cost     = aggregator.solve_decentralized_line_search( cgd_solver, 1, params );
  end                           = std::chrono::high_resolution_clock::now();
  double decentralized_cgd_time = std::chrono::duration<double, std::milli>( end - start ).count();
  aggregator.reset();

  // Print results
  std::cout << "Centralized iLQR time: " << centralized_ilqr_time << " ms  |   cost : " << central_ilqr_cost << std::endl;
  std::cout << "Centralized OSQP time: " << centralized_osqp_time << " ms  |   cost : " << central_osqp_cost << std::endl;
  std::cout << "Centralized CGD time: " << centralized_cgd_time << " ms  |   cost : " << central_cgd_cost << std::endl;

  std::cout << "Decentralized iLQR time: " << decentralized_ilqr_time << " ms  |   cost : " << decentral_ilqr_cost << std::endl;
  std::cout << "Decentralized OSQP time: " << decentralized_osqp_time << " ms  |   cost : " << decentral_osqp_cost << std::endl;
  std::cout << "Decentralized CGD time: " << decentralized_cgd_time << " ms  |   cost : " << decentral_cgd_cost << std::endl;
}
