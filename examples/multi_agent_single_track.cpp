#include <cmath>

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "cli.hpp"
#include "example_utils.hpp"
#include "models/single_track_model.hpp"
#include "multi_agent_solver/agent.hpp"
#include "multi_agent_solver/multi_agent_problem.hpp"
#include "multi_agent_solver/solvers/solver.hpp"
#include "multi_agent_solver/strategies/strategy.hpp"

mas::OCP
create_single_track_circular_ocp( double initial_theta, double track_radius, double target_velocity, int time_steps )
{
  using namespace mas;
  OCP problem;
  problem.state_dim     = 4;
  problem.control_dim   = 2;
  problem.horizon_steps = time_steps;
  problem.dt            = 0.5;

  double x0             = track_radius * cos( initial_theta );
  double y0             = track_radius * sin( initial_theta );
  problem.initial_state = Eigen::VectorXd::Zero( problem.state_dim );
  problem.initial_state << x0, y0, 1.57 + initial_theta, 4.0;

  problem.dynamics = single_track_model;

  problem.stage_cost = [target_velocity, track_radius]( const State& state, const Control& control, size_t ) {
    const double w_track = 1.0, w_speed = 1.0, w_delta = 0.001, w_acc = 0.001;
    double       x = state( 0 ), y = state( 1 ), vx = state( 3 );
    double       delta = control( 0 ), a_cmd = control( 1 );
    double       distance_from_track = std::abs( std::sqrt( x * x + y * y ) - track_radius );
    double       speed_error         = vx - target_velocity;
    return w_track * distance_from_track * distance_from_track + w_speed * speed_error * speed_error + w_delta * delta * delta
         + w_acc * a_cmd * a_cmd;
  };
  problem.terminal_cost      = []( const State& ) { return 0.0; };
  problem.input_lower_bounds = Eigen::VectorXd::Constant( problem.control_dim, -0.5 );
  problem.input_upper_bounds = Eigen::VectorXd::Constant( problem.control_dim, 0.5 );

  problem.initialize_problem();
  problem.verify_problem();
  return problem;
}

struct Options
{
  bool        show_help = false;
  int         agents    = 10;
  int         max_outer = 10;
  std::string solver    = "ilqr";
  std::string strategy  = "centralized";
};

namespace
{

Options
parse_options( int argc, char** argv )
{
  Options                 options;
  examples::cli::ArgParser args( argc, argv );
  bool                    positional_agents = false;

  while( !args.empty() )
  {
    const std::string raw_arg = std::string( args.peek() );
    if( args.consume_flag( "--help", "-h" ) )
    {
      options.show_help = true;
      continue;
    }

    std::string value;
    if( args.consume_option( "--agents", value ) )
    {
      options.agents = examples::cli::parse_int( "--agents", value );
      continue;
    }
    if( args.consume_option( "--solver", value ) )
    {
      options.solver = value;
      continue;
    }
    if( args.consume_option( "--strategy", value ) )
    {
      options.strategy = value;
      continue;
    }
    if( args.consume_option( "--max-outer", value ) )
    {
      options.max_outer = examples::cli::parse_int( "--max-outer", value );
      continue;
    }

    if( examples::cli::is_positional( raw_arg ) && !positional_agents )
    {
      args.take();
      options.agents    = examples::cli::parse_int( "agents", raw_arg );
      positional_agents = true;
      continue;
    }

    throw std::invalid_argument( "Unknown argument '" + raw_arg + "'" );
  }

  return options;
}

void
print_usage()
{
  std::cout << "Usage: multi_agent_single_track [--agents N] [--solver NAME] [--strategy NAME] [--max-outer N]\n";
  std::cout << "       multi_agent_single_track N\n";
  std::cout << '\n';
  examples::print_available( std::cout );
}

} // namespace

int
main( int argc, char** argv )
{
  using namespace mas;
  try
  {
    const Options options = parse_options( argc, argv );
    if( options.show_help )
    {
      print_usage();
      return 0;
    }

    SolverParams params{
      { "max_iterations",  100 },
      {      "tolerance", 1e-5 },
      {         "max_ms", 1000 }
    };
    constexpr int    time_steps      = 10;
    constexpr double track_radius    = 20.0;
    constexpr double target_velocity = 5.0;

    MultiAgentProblem problem;
    for( int i = 0; i < options.agents; ++i )
    {
      double theta = 2.0 * M_PI * i / options.agents;
      auto   ocp   = std::make_shared<OCP>( create_single_track_circular_ocp( theta, track_radius, target_velocity, time_steps ) );
      problem.add_agent( std::make_shared<Agent>( i, ocp ) );
    }

    auto              solver        = examples::make_solver( options.solver );
    Strategy          strategy      = examples::make_strategy( options.strategy, std::move( solver ), params, options.max_outer );
    const auto        start         = std::chrono::steady_clock::now();
    const auto        solution      = mas::solve( strategy, problem );
    const auto        end           = std::chrono::steady_clock::now();
    const double      elapsed_ms    = std::chrono::duration<double, std::milli>( end - start ).count();
    const std::string solver_name   = examples::canonical_solver_name( options.solver );
    const std::string strategy_name = examples::canonical_strategy_name( options.strategy );

    std::cout << std::fixed << std::setprecision( 6 ) << "solver=" << solver_name << " strategy=" << strategy_name
              << " agents=" << options.agents << " cost=" << solution.total_cost << " time_ms=" << elapsed_ms << '\n';

    if( problem.blocks.empty() )
      problem.compute_offsets();

    for( std::size_t idx = 0; idx < solution.states.size() && idx < problem.blocks.size(); ++idx )
    {
      const auto& block      = problem.blocks[idx];
      const auto& ocp        = *block.agent->ocp;
      const std::string base = "agent_" + std::to_string( block.agent_id );
      examples::print_state_trajectory( std::cout, solution.states[idx], ocp.dt, base );
      examples::print_control_trajectory( std::cout, solution.controls[idx], ocp.dt, base );
    }
  }
  catch( const std::exception& e )
  {
    std::cerr << "Error: " << e.what() << "\n";
    std::cerr << "Use --help to see available options.\n";
    return 1;
  }
  return 0;
}
