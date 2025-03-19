#pragma once
#include <chrono>
#include <iostream>
#include <map>
#include <memory>
#include <vector>

#include "models/single_track_model.hpp"
#include "multi_agent_aggregator.hpp"
#include "ocp.hpp"
#include "solvers/cgd.hpp"
#include "solvers/ilqr.hpp"
#include "solvers/osqp_solver.hpp"
#include "types.hpp"

OCP
create_single_track_circular_ocp( double initial_theta, double track_radius, double target_velocity, int agent_id,
                                  const std::vector<std::shared_ptr<OCP>> &all_agents, int time_steps )
{
  OCP problem;

  // System dimensions
  problem.state_dim     = 5;
  problem.control_dim   = 2;
  problem.horizon_steps = time_steps;
  problem.dt            = 0.1;

  // Convert (theta, radius) to (x, y)
  double x0 = track_radius * cos( initial_theta );
  double y0 = track_radius * sin( initial_theta );

  // Initial state (X, Y, psi, vx)
  problem.initial_state = Eigen::VectorXd::Zero( problem.state_dim );
  problem.initial_state << x0, y0, initial_theta, target_velocity, 0.0;

  // Dynamics
  problem.dynamics = single_track_model;

  problem.stage_cost = [&, id = agent_id]( const State &state, const Control &control ) -> double {
    // Cost weights
    const double w_track      = 0.1;
    const double w_speed      = 0.1;
    const double w_delta      = 0.1;
    const double w_acc        = 0.1;
    const double w_separation = 0.1;

    // Extract states
    double x    = state( 0 );
    double y    = state( 1 );
    double psi  = state( 2 );
    double vx   = state( 3 );
    double time = state( 4 );

    // Extract controls
    double delta = control( 0 );
    double a_cmd = control( 1 );

    // Compute deviation from circular path
    double distance_from_track = std::abs( std::sqrt( x * x + y * y ) - track_radius );

    // Compute speed error
    double speed_error = vx - target_velocity;

    // Compute time index
    int time_index = std::round( time / problem.dt );

    // Find the closest vehicle ahead and behind
    double min_distance_ahead  = std::numeric_limits<double>::max();
    double min_distance_behind = std::numeric_limits<double>::max();

    for( const auto &other_agent : all_agents )
    {
      if( other_agent == nullptr || other_agent->id == id )
        continue;

      int          other_time_index = std::min( time_index, static_cast<int>( other_agent->initial_states.cols() ) - 1 );
      const State &other_state      = other_agent->initial_states.col( other_time_index );

      double x_other = other_state( 0 );
      double y_other = other_state( 1 );

      double dx       = x_other - x;
      double dy       = y_other - y;
      double distance = std::sqrt( dx * dx + dy * dy );

      double theta_self       = std::atan2( y, x );
      double theta_other      = std::atan2( y_other, x_other );
      double angular_distance = theta_other - theta_self;

      if( angular_distance < 0 )
        angular_distance += 2.0 * M_PI;

      double arc_distance = track_radius * angular_distance;

      if( arc_distance > 0 && arc_distance < min_distance_ahead )
      {
        min_distance_ahead = arc_distance;
      }
      else if( arc_distance < 0 && std::abs( arc_distance ) < min_distance_behind )
      {
        min_distance_behind = std::abs( arc_distance );
      }
    }

    // Compute separation costs
    double min_safe_distance = 30.0;
    double separation_cost   = ( min_distance_ahead < min_safe_distance )
                               ? w_separation * std::exp( -( min_distance_ahead - min_safe_distance ) / min_safe_distance )
                               : 0.0;

    double behind_penalty = ( min_distance_behind < min_safe_distance )
                            ? w_separation * std::exp( -( min_distance_behind - min_safe_distance ) / min_safe_distance )
                            : 0.0;

    separation_cost += behind_penalty;

    // Final cost
    double cost = w_track * ( distance_from_track * distance_from_track ) + w_speed * ( speed_error * speed_error )
                + w_delta * ( delta * delta ) + w_acc * ( a_cmd * a_cmd ) + separation_cost;

    return cost / 100;
  };

  // Terminal cost
  problem.terminal_cost = [=]( const State &state ) -> double {
    return 0.0;
    // double x                   = state( 0 );
    // double y                   = state( 1 );
    // double distance_from_track = std::abs( std::sqrt( x * x + y * y ) - track_radius );
    // return 10.0 * ( distance_from_track * distance_from_track ); // Higher penalty for end deviation
  };

  // Control bounds( steering, acceleration );
  // problem.input_lower_bounds = Eigen::VectorXd::Constant( problem.control_dim, -0.5 );
  // problem.input_upper_bounds = Eigen::VectorXd::Constant( problem.control_dim, 0.5 );

  // Initialize and verify the problem
  problem.initialize_problem();
  problem.verify_problem();

  return problem;
}

void
multi_agent_circular_test( int num_agents = 10, int time_steps = 10 )
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
      create_single_track_circular_ocp( initial_theta, track_radius, target_velocity, i, agent_ocps, time_steps ) );
    agent_ocp->id = i;
    agent_ocps.push_back( agent_ocp );
    aggregator.agent_ocps[i] = agent_ocp;
  }

  const int    max_iter  = 2;
  const int    max_outer = 50;
  const double tolerance = 1e-8;

  // Compute offsets for multi-agent system
  aggregator.compute_offsets();
  aggregator.reset();

  // Define solvers and multi-agent solving methods
  std::vector<std::pair<std::string, Solver>> solvers = {
    { "iLQR", ilqr_solver },
    { "OSQP", osqp_solver },
    {  "CGD",  cgd_solver }
  };

  std::vector<std::pair<std::string, std::function<double( MultiAgentAggregator &, const Solver &, int, int, double )>>> solving_methods = {
    {                " Centralized", []( MultiAgentAggregator &agg, const Solver &solver, int outer, int inner,
     double tol ) { return agg.solve_centralized( solver, outer * inner, tol ); }              },
    {       " Decentralized_Simple", []( MultiAgentAggregator &agg, const Solver &solver, int outer, int inner,
     double tol ) { return agg.solve_decentralized_simple( solver, outer, inner, tol ); }      },
    {  " Decentralized_Line_Search", []( MultiAgentAggregator &agg, const Solver &solver, int outer, int inner,
     double tol ) { return agg.solve_decentralized_line_search( solver, outer, inner, tol ); } },
    { " Decentralized_Trust_Region", []( MultiAgentAggregator &agg, const Solver &solver, int outer, int inner,
     double tol ) { return agg.solve_decentralized_trust_region( solver, outer, inner, tol ); } }
  };

  // Storage for results
  std::map<std::string, std::map<std::string, std::pair<double, double>>> results; // [method][solver] -> (cost, time)

  // Run all solvers with all methods
  for( const auto &method : solving_methods )
  {
    for( const auto &solver : solvers )
    {
      aggregator.reset();
      auto   start_time = std::chrono::high_resolution_clock::now();
      double cost       = method.second( aggregator, solver.second, max_outer, max_iter, tolerance );
      auto   end_time   = std::chrono::high_resolution_clock::now();
      double time_ms    = std::chrono::duration<double, std::milli>( end_time - start_time ).count();

      results[method.first][solver.first] = { cost, time_ms };
    }
  }

  // **Find min/max for both cost and time**
  double min_cost = std::numeric_limits<double>::max();
  double max_cost = std::numeric_limits<double>::lowest();
  double min_time = std::numeric_limits<double>::max();
  double max_time = std::numeric_limits<double>::lowest();

  for( const auto &method : results )
  {
    for( const auto &solver : method.second )
    {
      double cost    = solver.second.first;
      double time_ms = solver.second.second;
      if( cost < min_cost )
        min_cost = cost;
      if( cost > max_cost )
        max_cost = cost;
      if( time_ms < min_time )
        min_time = time_ms;
      if( time_ms > max_time )
        max_time = time_ms;
    }
  }

  // **Formatted Output Table**
  std::cout << std::endl << "\n\n<<<<<<<  num agents  " << num_agents << "    horizon steps " << time_steps << std::endl;
  std::cout << "========================================================================================" << std::endl;
  std::cout << std::setw( 40 ) << "Outer Method" << std::setw( 20 ) << "Solver" << std::setw( 20 ) << "Time (ms)" << std::setw( 20 )
            << "Cost" << std::endl;
  std::cout << "========================================================================================\n";

  for( const auto &method : results )
  {
    for( const auto &solver : method.second )
    {
      double cost    = solver.second.first;
      double time_ms = solver.second.second;

      // **Apply color coding**
      std::string cost_color = RESET;
      std::string time_color = RESET;

      if( cost < 1.2 * min_cost )
        cost_color = GREEN; // Best cost 🟩
      else if( cost < 2 * min_cost )
        cost_color = YELLOW; // Mid-range 🟨
      else
        cost_color = RED; // Worst cost 🟥

      if( time_ms < 2 * min_time )
        time_color = GREEN; // Fastest 🟩
      else if( time_ms < 4 * min_time )
        time_color = YELLOW; // Mid-range 🟨
      else
        time_color = RED; // Slowest 🟥

      std::cout << std::setw( 40 ) << method.first << std::setw( 20 ) << solver.first << time_color // Apply time color
                << std::setw( 20 ) << std::fixed << std::setprecision( 2 ) << time_ms << RESET      // Reset color
                << cost_color                                                                       // Apply cost color
                << std::setw( 20 ) << std::fixed << std::setprecision( 4 ) << cost << RESET         // Reset color
                << "\n";
    }
  }

  std::cout << "=======================================================================================\n";
}
