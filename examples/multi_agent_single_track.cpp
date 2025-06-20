#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <tuple>
#include <vector>

#include "multi_agent_solver/models/single_track_model.hpp"
#include "multi_agent_solver/multi_agent_aggregator.hpp"
#include "multi_agent_solver/ocp.hpp"
#include "multi_agent_solver/solvers/solver.hpp"

mas::OCP
create_single_track_circular_ocp( double initial_theta, double track_radius, double target_velocity, int agent_id,
                                  const std::vector<std::shared_ptr<mas::OCP>> &all_agents, int time_steps )
{
  using namespace mas;
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

  problem.stage_cost = [&, id = agent_id]( const State &state, const Control &control, size_t idx ) -> double {
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

struct Result
{
  std::string name;
  double      cost;
  double      time_ms;
};

int
main()
{
  using namespace mas;
  constexpr int    num_agents      = 10;
  constexpr int    time_steps      = 10;
  constexpr double track_radius    = 20.0;
  constexpr double target_velocity = 5.0;

  MultiAgentAggregator              aggregator;
  std::vector<std::shared_ptr<OCP>> agent_ocps;

  for( int i = 0; i < num_agents; ++i )
  {
    double theta  = 2.0 * M_PI * i / num_agents;
    auto   ocp_sp = std::make_shared<OCP>(
      create_single_track_circular_ocp( theta, track_radius, target_velocity, i, agent_ocps, time_steps ) );
    ocp_sp->id = i;
    agent_ocps.push_back( ocp_sp );
    aggregator.agent_ocps[i] = ocp_sp;
  }

  aggregator.compute_offsets(); // once

  SolverParams params{
    { "max_iterations",    2 },
    {      "tolerance", 1e-5 },
    {         "max_ms", 1000 }
  };
  constexpr int max_outer = 50;

  std::vector<Result> results;

  auto time_solver = [&]( const std::string &name, auto &&solver_call ) {
    aggregator.reset();
    auto start = std::chrono::high_resolution_clock::now();
    solver_call();
    auto                                      end     = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    results.push_back( { name, aggregator.agent_cost_sum(), elapsed.count() } );
  };

  // Benchmark solvers ---------------------------------------------------------
  time_solver( "Centralized iLQR", [&] { aggregator.solve_centralized<iLQR>( params ); } );
  time_solver( "Centralized CGD", [&] { aggregator.solve_centralized<CGD>( params ); } );
  time_solver( "Centralized OSQP", [&] { aggregator.solve_centralized<OSQP>( params ); } );
  time_solver( "Centralized OSQPcol", [&] { aggregator.solve_centralized<OSQPCollocation>( params ); } );

  // decentralised variants ----------------------------------------------------
  time_solver( "Decentralized iLQR (Trust Region)", [&] { aggregator.solve_decentralized_trust_region<iLQR>( max_outer, params ); } );
  time_solver( "Decentralized iLQR (Line Search)", [&] { aggregator.solve_decentralized_line_search<iLQR>( max_outer, params ); } );
  time_solver( "Decentralized CGD (Trust Region)", [&] { aggregator.solve_decentralized_trust_region<CGD>( max_outer, params ); } );
  time_solver( "Decntralized CGD (Line Search)", [&] { aggregator.solve_decentralized_line_search<CGD>( max_outer, params ); } );
  time_solver( "Decentralized OSQP (Trust Region)", [&] { aggregator.solve_decentralized_trust_region<OSQP>( max_outer, params ); } );
  time_solver( "Decentralized OSQP (Line Search)", [&] { aggregator.solve_decentralized_line_search<OSQP>( max_outer, params ); } );
  time_solver( "Decentralized OSQPcol (Trust Region)",
               [&] { aggregator.solve_decentralized_trust_region<OSQPCollocation>( max_outer, params ); } );
  time_solver( "Decentralized OSQPcol (Line Search)",
               [&] { aggregator.solve_decentralized_line_search<OSQPCollocation>( max_outer, params ); } );


  /* ------------------------------------------------------------------ */
  /*  Pretty table with colours                                          */
  /* ------------------------------------------------------------------ */
  namespace pc = print_color; // alias

  auto best_cost = std::min_element( results.begin(), results.end(), []( auto &a, auto &b ) { return a.cost < b.cost; } )->cost;
  auto best_time = std::min_element( results.begin(), results.end(), []( auto &a, auto &b ) { return a.time_ms < b.time_ms; } )->time_ms;

  /* helper: choose colour -------------------------------------------------- */
  auto colour = []( double val, double best, bool lower_is_better = true ) {
    double ratio = lower_is_better ? val / best : best / val;
    if( ratio <= 1.1 )
      return pc::green; // ≤5 % from best
    else if( ratio <= 2.0 )
      return pc::yellow; // within 30 %
    else
      return pc::red; // worse
  };

  std::cout << "Multi-Agent Single Track OCP Benchmark\n";
  std::cout << "---------------------------------------\n";
  std::cout << "Number of agents: " << num_agents << ", Time steps: " << time_steps << '\n';

  std::cout << '\n' << std::fixed << std::setprecision( 6 );
  std::cout << std::setw( 40 ) << std::left << "Method" << std::setw( 15 ) << "Cost" << std::setw( 15 ) << "Time (ms)" << '\n'
            << std::string( 70, '-' ) << '\n';

  for( const auto &r : results )
  {
    std::cout << std::setw( 40 ) << std::left << r.name << colour( r.cost, best_cost ) << std::setw( 15 ) << r.cost << pc::reset
              << colour( r.time_ms, best_time ) << std::setw( 15 ) << r.time_ms << pc::reset << '\n';
  }
  return 0;
}