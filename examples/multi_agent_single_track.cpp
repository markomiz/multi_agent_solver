#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
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
#include "multi_agent_solver/types.hpp"

namespace
{

constexpr double kTwoPi = 6.28318530717958647692;

template<typename Scalar>
mas::OCP<Scalar>
create_single_track_circular_ocp( Scalar initial_theta, Scalar track_radius, Scalar target_velocity, int time_steps )
{
  using OCP      = mas::OCP<Scalar>;
  using State    = typename OCP::State;
  using Control  = typename OCP::Control;
  using StageCostFunction   = typename OCP::StageCostFunction;
  using TerminalCostFunction = typename OCP::TerminalCostFunction;

  OCP problem;
  problem.state_dim     = 4;
  problem.control_dim   = 2;
  problem.horizon_steps = time_steps;
  problem.dt            = static_cast<Scalar>( 0.5 );

  State initial_state = State::Zero( problem.state_dim );
  const Scalar x0     = track_radius * std::cos( initial_theta );
  const Scalar y0     = track_radius * std::sin( initial_theta );
  initial_state << x0, y0, static_cast<Scalar>( 1.57 ) + initial_theta, static_cast<Scalar>( 4.0 );
  problem.initial_state = initial_state;

  problem.dynamics = mas::single_track_model<Scalar>;

  const Scalar w_track = static_cast<Scalar>( 1.0 );
  const Scalar w_speed = static_cast<Scalar>( 1.0 );
  const Scalar w_delta = static_cast<Scalar>( 0.001 );
  const Scalar w_acc   = static_cast<Scalar>( 0.001 );

  problem.stage_cost = [=]( const State& state, const Control& control, std::size_t ) {
    const Scalar x     = state( 0 );
    const Scalar y     = state( 1 );
    const Scalar vx    = state( 3 );
    const Scalar delta = control( 0 );
    const Scalar a_cmd = control( 1 );
    const Scalar radius        = std::sqrt( x * x + y * y );
    const Scalar distance      = std::abs( radius - track_radius );
    const Scalar speed_error   = vx - target_velocity;
    return w_track * distance * distance + w_speed * speed_error * speed_error + w_delta * delta * delta
         + w_acc * a_cmd * a_cmd;
  };
  problem.terminal_cost = []( const State& ) { return static_cast<Scalar>( 0 ); };

  problem.input_lower_bounds = Control::Constant( problem.control_dim, static_cast<Scalar>( -0.5 ) );
  problem.input_upper_bounds = Control::Constant( problem.control_dim, static_cast<Scalar>( 0.5 ) );

  problem.initialize_problem();
  problem.verify_problem();
  return problem;
}

template<typename Scalar>
mas::MultiAgentProblemT<Scalar>
create_problem( int agents, Scalar track_radius, Scalar target_velocity, int time_steps )
{
  mas::MultiAgentProblemT<Scalar> problem;
  for( int i = 0; i < agents; ++i )
  {
    const Scalar theta = static_cast<Scalar>( kTwoPi * static_cast<double>( i ) / static_cast<double>( agents ) );
    auto         ocp   = std::make_shared<mas::OCP<Scalar>>( create_single_track_circular_ocp<Scalar>(
      theta, track_radius, target_velocity, time_steps ) );
    problem.add_agent( std::make_shared<mas::AgentBase<Scalar>>( static_cast<std::size_t>( i ), ocp ) );
  }
  return problem;
}

struct Options
{
  bool                     show_help         = false;
  int                      agents            = 10;
  int                      max_outer         = 10;
  bool                     dump_trajectories = false;
  std::vector<std::string> solvers;
  std::vector<std::string> strategies;
  std::vector<std::string> scalars;
};

void
trim_in_place( std::string& value )
{
  const auto first = value.find_first_not_of( " \t" );
  if( first == std::string::npos )
  {
    value.clear();
    return;
  }
  const auto last = value.find_last_not_of( " \t" );
  value            = value.substr( first, last - first + 1 );
}

void
append_list( std::vector<std::string>& target, const std::string& csv, const auto& canonicalizer )
{
  std::size_t start = 0;
  while( start <= csv.size() )
  {
    const auto comma = csv.find( ',', start );
    std::string token
      = csv.substr( start, comma == std::string::npos ? std::string::npos : comma - start );
    trim_in_place( token );
    if( !token.empty() )
      target.push_back( canonicalizer( token ) );
    if( comma == std::string::npos )
      break;
    start = comma + 1;
  }
}

int
parse_int( const std::string& label, const std::string& value )
{
  int result = 0;
  const char* begin = value.data();
  const char* end   = begin + value.size();
  const auto   [ptr, ec] = std::from_chars( begin, end, result );
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
    if( arg == "--dump-trajectories" )
    {
      options.dump_trajectories = true;
      continue;
    }

    std::string value;
    if( match_with_value( "--agents", value ) )
    {
      options.agents = parse_int( "--agents", value );
    }
    else if( match_with_value( "--solver", value ) )
    {
      append_list( options.solvers, value, examples::canonical_solver_name );
    }
    else if( match_with_value( "--solvers", value ) )
    {
      append_list( options.solvers, value, examples::canonical_solver_name );
    }
    else if( match_with_value( "--strategy", value ) )
    {
      append_list( options.strategies, value, examples::canonical_strategy_name );
    }
    else if( match_with_value( "--strategies", value ) )
    {
      append_list( options.strategies, value, examples::canonical_strategy_name );
    }
    else if( match_with_value( "--scalar", value ) )
    {
      append_list( options.scalars, value, examples::canonical_scalar_name );
    }
    else if( match_with_value( "--scalars", value ) )
    {
      append_list( options.scalars, value, examples::canonical_scalar_name );
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
  std::cout << "Usage: multi_agent_single_track [--agents N] [--solvers NAMES] [--strategies NAMES] [--max-outer N]"
               " [--scalars float,double] [--dump-trajectories]\n";
  std::cout << "       multi_agent_single_track N\n";
  std::cout << '\n';
  examples::print_available( std::cout );
}

template<typename Scalar>
void
run_for_scalar( const Options& options, const std::vector<std::string>& solver_names,
                const std::vector<std::string>& strategy_names )
{
  using Problem = mas::MultiAgentProblemT<Scalar>;

  const mas::SolverParamsT<Scalar> params{ { "max_iterations", static_cast<Scalar>( 100 ) },
                                           { "tolerance", static_cast<Scalar>( 1e-5 ) },
                                           { "max_ms", static_cast<Scalar>( 1000 ) } };

  const Scalar track_radius    = static_cast<Scalar>( 20.0 );
  const Scalar target_velocity = static_cast<Scalar>( 5.0 );
  constexpr int time_steps     = 10;
  const std::string scalar_label = examples::scalar_label<Scalar>();

  for( const auto& solver_name : solver_names )
  {
    if( !examples::solver_supported_for_scalar<Scalar>( solver_name ) )
    {
      std::cout << "scalar=" << scalar_label << " solver=" << solver_name
                << " unsupported (skipping)\n";
      continue;
    }

    for( const auto& strategy_name : strategy_names )
    {
      Problem problem = create_problem<Scalar>( options.agents, track_radius, target_velocity, time_steps );
      auto    solver  = examples::make_solver<Scalar>( solver_name );
      auto    strategy
        = examples::make_strategy<Scalar>( strategy_name, std::move( solver ), params, options.max_outer );

      const auto start    = std::chrono::steady_clock::now();
      auto       solution = mas::solve( strategy, problem );
      const auto end      = std::chrono::steady_clock::now();
      const double elapsed_ms = std::chrono::duration<double, std::milli>( end - start ).count();

      std::cout << std::fixed << std::setprecision( 6 )
                << "scalar=" << scalar_label
                << " solver=" << solver_name
                << " strategy=" << strategy_name
                << " agents=" << options.agents
                << " cost=" << static_cast<double>( solution.total_cost )
                << " time_ms=" << elapsed_ms
                << '\n';

      if( options.dump_trajectories )
      {
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
    }
  }
}

} // namespace

int
main( int argc, char** argv )
{
  try
  {
    const Options options = parse_options( argc, argv );
    if( options.show_help )
    {
      print_usage();
      return 0;
    }

    const std::vector<std::string> solver_names = options.solvers.empty()
                                                     ? examples::available_solver_names<double>()
                                                     : options.solvers;
    const std::vector<std::string> default_strategies{ "centralized", "sequential", "linesearch", "trustregion" };
    const std::vector<std::string> strategy_names
      = options.strategies.empty() ? default_strategies : options.strategies;

    const std::vector<std::string> scalar_names
      = options.scalars.empty() ? std::vector<std::string>{ "float", "double" } : options.scalars;

    for( const auto& scalar_name : scalar_names )
    {
      if( scalar_name == "float" )
        run_for_scalar<float>( options, solver_names, strategy_names );
      else if( scalar_name == "double" )
        run_for_scalar<double>( options, solver_names, strategy_names );
      else
        std::cout << "Unknown scalar '" << scalar_name << "' -- skipping\n";
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
