#pragma once
#include <chrono>
#include <iostream>
#include <map>
#include <memory>
#include <vector>

#include "models/single_track_model.hpp"
#include "multi_agent_aggregator.hpp"
#include "ocp.hpp"
#include "solvers/constrained_gradient_descent.hpp"
#include "solvers/gradient_descent.hpp"
#include "solvers/ilqr.hpp"
#include "solvers/osqp_solver.hpp"
#include "types.hpp"

OCP
create_single_track_circular_ocp( double initial_theta, double track_radius, double target_velocity, int agent_id,
                                  const std::vector<std::shared_ptr<OCP>>& all_agents )
{
  OCP problem;

  // System dimensions
  problem.state_dim     = 5;
  problem.control_dim   = 2;
  problem.horizon_steps = 50;
  problem.dt            = 0.1;

  // Convert (theta, radius) to (x, y)
  double x0 = track_radius * cos( initial_theta );
  double y0 = track_radius * sin( initial_theta );

  // Initial state (X, Y, psi, vx)
  problem.initial_state = Eigen::VectorXd::Zero( problem.state_dim );
  problem.initial_state << x0, y0, initial_theta, target_velocity, 0.0;

  // Dynamics
  problem.dynamics = single_track_model;

  // Cost weights
  const double w_track      = 0.01; // Penalize deviation from circular track
  const double w_speed      = 0.1;  // Penalize speed error
  const double w_delta      = 0.1;  // Penalize steering
  const double w_acc        = 0.1;  // Penalize acceleration
  const double w_separation = 0.0;  // Penalize getting too close to the next agent

  // Stage cost function
  problem.stage_cost = [=]( const State& state, const Control& control ) -> double {
    // Extract states
    double x   = state( 0 );
    double y   = state( 1 );
    double psi = state( 2 );
    double vx  = state( 3 );

    // Extract controls
    double delta = control( 0 );
    double a_cmd = control( 1 );

    // Compute deviation from circular path
    double distance_from_track = std::abs( std::sqrt( x * x + y * y ) - track_radius );

    // Compute speed error
    double speed_error = vx - target_velocity;

    int time_index = state( 4 ) / problem.dt;

    // Find the closest vehicle ahead
    double min_distance = 100.0;
    for( const auto& other_agent : all_agents )
    {

      // Get other agent's state
      double dx = state.col( time_index )( 0 ) - other_agent->best_states.col( time_index )( 0 );
      double dy = state.col( time_index )( 1 ) - other_agent->best_states.col( time_index )( 1 );

      double distance2 = dy * dx + dy * dy;

      if( distance2 < min_distance )
        min_distance = distance2;
    }


    // Compute cost for maintaining safe separation
    double separation_cost = 0.0;


    double cost = w_track * ( distance_from_track * distance_from_track ) + w_speed * ( speed_error * speed_error )
                + w_delta * ( delta * delta ) + w_acc * ( a_cmd * a_cmd ) - min_distance * 0.01;
    return cost;
  };

  // Terminal cost
  problem.terminal_cost = [=]( const State& state ) -> double {
    return 0.0;
    // double x                   = state( 0 );
    // double y                   = state( 1 );
    // double distance_from_track = std::abs( std::sqrt( x * x + y * y ) - track_radius );
    // return 10.0 * ( distance_from_track * distance_from_track ); // Higher penalty for end deviation
  };

  // Control bounds( steering, acceleration );
  problem.input_lower_bounds = Eigen::VectorXd::Constant( problem.control_dim, -0.5 );
  problem.input_upper_bounds = Eigen::VectorXd::Constant( problem.control_dim, 0.5 );

  // Initialize and verify the problem
  problem.initialize_problem();
  problem.verify_problem();

  return problem;
}

void
multi_agent_circular_test( int num_agents = 6 )
{
  const double track_radius    = 20.0;
  const double target_velocity = 5.0;

  // Create aggregator for multi-agent problem
  MultiAgentAggregator aggregator;

  // Vector of agent OCPs (so cost function can access them)
  std::vector<std::shared_ptr<OCP>> agent_ocps;

  // Assign each agent an OCP with an initial position along the circle
  for( int i = 0; i < num_agents; ++i )
  {
    double initial_theta = 2.0 * M_PI * i / num_agents;
    auto   agent_ocp     = std::make_shared<OCP>(
      create_single_track_circular_ocp( initial_theta, track_radius, target_velocity, i, agent_ocps ) );
    agent_ocps.push_back( agent_ocp );
    aggregator.agent_ocps[i] = agent_ocp;
  }

  // Compute offsets for multi-agent system
  aggregator.compute_offsets();
  aggregator.reset();
  // Solve in centralized mode
  auto   start                 = std::chrono::high_resolution_clock::now();
  double central_ilqr_cost     = aggregator.solve_centralized( ilqr_solver, 100, 1e-5 );
  auto   end                   = std::chrono::high_resolution_clock::now();
  double centralized_ilqr_time = std::chrono::duration<double, std::milli>( end - start ).count();
  aggregator.reset();

  start                        = std::chrono::high_resolution_clock::now();
  double central_osqp_cost     = aggregator.solve_centralized( osqp_solver, 100, 1e-5 );
  end                          = std::chrono::high_resolution_clock::now();
  double centralized_osqp_time = std::chrono::duration<double, std::milli>( end - start ).count();
  aggregator.reset();

  start                       = std::chrono::high_resolution_clock::now();
  double central_cgd_cost     = aggregator.solve_centralized( constrained_gradient_descent_solver, 100, 1e-5 );
  end                         = std::chrono::high_resolution_clock::now();
  double centralized_cgd_time = std::chrono::duration<double, std::milli>( end - start ).count();
  aggregator.reset();

  // Solve in decentralized mode
  start                          = std::chrono::high_resolution_clock::now();
  double decentral_ilqr_cost     = aggregator.solve_decentralized( ilqr_solver, 10, 20, 1e-5 );
  end                            = std::chrono::high_resolution_clock::now();
  double decentralized_ilqr_time = std::chrono::duration<double, std::milli>( end - start ).count();
  aggregator.reset();

  start                          = std::chrono::high_resolution_clock::now();
  double decentral_osqp_cost     = aggregator.solve_decentralized( osqp_solver, 10, 20, 1e-5 );
  end                            = std::chrono::high_resolution_clock::now();
  double decentralized_osqp_time = std::chrono::duration<double, std::milli>( end - start ).count();
  aggregator.reset();

  start                         = std::chrono::high_resolution_clock::now();
  double decentral_cgd_cost     = aggregator.solve_decentralized( constrained_gradient_descent_solver, 10, 20, 1e-5 );
  end                           = std::chrono::high_resolution_clock::now();
  double decentralized_cgd_time = std::chrono::duration<double, std::milli>( end - start ).count();
  aggregator.reset();

  // Print results
  std::cerr << "\n\n\n++++++++++++++++++++++++++++++++" << std::endl;
  std::cout << "🚗 Multi-Agent Single-Track Lane Following Test 🚗\n";
  std::cout << "-----------------------------------\n";
  std::cout << "Centralized iLQR time: " << centralized_ilqr_time << " ms  | Cost: " << central_ilqr_cost << std::endl;
  std::cout << "Centralized OSQP time: " << centralized_osqp_time << " ms  | Cost: " << central_osqp_cost << std::endl;
  std::cout << "Centralized CGD time: " << centralized_cgd_time << "   ms  | Cost: " << central_cgd_cost << std::endl;

  std::cout << "Decentralized iLQR time: " << decentralized_ilqr_time << " ms  | Cost: " << decentral_ilqr_cost << std::endl;
  std::cout << "Decentralized OSQP time: " << decentralized_osqp_time << " ms  | Cost: " << decentral_osqp_cost << std::endl;
  std::cout << "Decentralized CGD time: " << decentralized_cgd_time << "   ms  | Cost: " << decentral_cgd_cost << std::endl;
}
