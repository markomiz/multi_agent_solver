#include <cmath>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

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

int
parse_int( const std::string& label, const std::string& value )
{
  int         result   = 0;
  const char* begin    = value.data();
  const char* end      = begin + value.size();
  const auto [ptr, ec] = std::from_chars( begin, end, result );
  if( ec != std::errc() || ptr != end )
    throw std::invalid_argument( "Invalid value for " + label + ": '" + value + "'" );
  return result;
}

Options
parse_options( int argc, char** argv )
{
  Options options;
  bool    positional_agents = false;
  for( int i = 1; i < argc; ++i )
  {
    std::string arg = argv[i];
    if( arg.rfind( "--", 0 ) == 0 )
    {
      const auto eq_pos = arg.find( '=' );
      const auto end    = eq_pos == std::string::npos ? arg.size() : eq_pos;
      std::replace( arg.begin() + 2, arg.begin() + static_cast<std::ptrdiff_t>( end ), '_', '-' );
    }
    auto match_with_value = [&]( const std::string& name, std::string& out ) {
      const std::string prefix = name + "=";
      if( arg == name )
      {
        if( i + 1 >= argc )
          throw std::invalid_argument( "Missing value for option '" + name + "'" );
        out = argv[++i];
        return true;
      }
      if( arg.rfind( prefix, 0 ) == 0 )
      {
        out = arg.substr( prefix.size() );
        return true;
      }
      return false;
    };

    if( arg == "--help" || arg == "-h" )
    {
      options.show_help = true;
      continue;
    }

    std::string value;
    if( match_with_value( "--agents", value ) )
    {
      options.agents = parse_int( "--agents", value );
    }
    else if( match_with_value( "--solver", value ) )
    {
      options.solver = value;
    }
    else if( match_with_value( "--strategy", value ) )
    {
      options.strategy = value;
    }
    else if( match_with_value( "--max-outer", value ) )
    {
      options.max_outer = parse_int( "--max-outer", value );
    }
    else if( !arg.empty() && arg.front() != '-' && !positional_agents )
    {
      options.agents    = parse_int( "agents", arg );
      positional_agents = true;
    }
    else
    {
      throw std::invalid_argument( "Unknown argument '" + arg + "'" );
    }
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
  }
  catch( const std::exception& e )
  {
    std::cerr << "Error: " << e.what() << "\n";
    std::cerr << "Use --help to see available options.\n";
    return 1;
  }
  return 0;
}
