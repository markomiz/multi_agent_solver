#pragma once
#include <chrono>
#include <iostream>
#include <map>
#include <memory>
#include <unordered_map>

#include "Eigen/Dense"

#include "multi_agent_aggregator.hpp"
#include "ocp.hpp"
#include "solver_output.hpp"
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
  const int    num_agents    = 4;
  const int    state_dim     = 4;
  const int    control_dim   = 4;
  const double dt            = 0.1;
  const int    horizon_steps = 10;

  // first test individual agent
  OCP          agent_ocp      = create_linear_lqr_ocp( state_dim, control_dim, dt, horizon_steps );
  SolverOutput agent_solution = ilqr_solver( agent_ocp, 100, 1e-5 );
  std::cout << "Single-agent LQR cost: " << agent_solution.cost << std::endl;

  // Create an aggregator for multi-agent problems.
  MultiAgentAggregator aggregator;

  // For each agent (here using agent IDs 0, 1, 2, 3), create an LQR OCP.
  for( int i = 0; i < num_agents; ++i )
  {
    std::shared_ptr<OCP> agent_ocp = std::make_shared<OCP>( create_linear_lqr_ocp( state_dim, control_dim, dt, horizon_steps ) );
    aggregator.agent_ocps[i]       = agent_ocp;
  }

  // Compute the offsets (which determine where each agent's state and control are located in the global vector).
  aggregator.compute_offsets();


  // Create the global OCP by aggregating all agent OCPs.
  OCP global_ocp = aggregator.create_global_ocp();

  std::cerr << "GLOBAL OCP SUCCESSFULLY CREATED" << std::endl;

  assert( global_ocp.objective_function && "❌ ERROR: Global OCP objective function was not set!" );


  // Solve the global OCP using the iLQR solver.
  // (You could also try OSQP or constrained GD if desired.)
  SolverOutput global_ilqr_solution = ilqr_solver( global_ocp, 100, 1e-5 );
  SolverOutput global_sqp_solution  = osqp_solver( global_ocp, 100, 1e-5 );


  // Extract individual agent solutions.
  std::unordered_map<size_t, SolverOutput> ilqr_agent_solutions = aggregator.extract_solutions( global_ilqr_solution );
  // std::unordered_map<size_t, SolverOutput> osqp_agent_solutions = aggregator.extract_solutions( global_sqp_solution );


  std::cout << "Multi-agent LQR results:" << std::endl;
  for( const auto& [agent_id, sol] : ilqr_agent_solutions )
  {
    std::cout << "Agent " << agent_id << " cost: " << sol.cost << std::endl;
  }
  // std::cout << "Multi-agent OSQP results:" << std::endl;
  // for( const auto& [agent_id, sol] : osqp_agent_solutions )
  // {
  //   std::cout << "Agent " << agent_id << " cost: " << sol.cost << std::endl;
  // }

  // Also, print the global cost.
  std::cout << "Global cost ilqr: " << global_ilqr_solution.cost << std::endl;
  // std::cout << "Global cost osqp: " << global_sqp_solution.cost << std::endl;
}
