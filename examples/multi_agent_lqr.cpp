#include <charconv>
#include <chrono>
#include <system_error>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "Eigen/Dense"

#include "multi_agent_solver/agent.hpp"
#include "multi_agent_solver/multi_agent_problem.hpp"
#include "multi_agent_solver/solvers/solver.hpp"
#include "multi_agent_solver/strategies/strategy.hpp"
#include "multi_agent_solver/types.hpp"

#include "example_utils.hpp"

/*──────────────── create simple LQR OCP (unchanged) ───────────────*/
mas::OCP
create_linear_lqr_ocp( int n_x, int n_u, double dt, int T )
{
  using namespace mas;
  OCP ocp;
  ocp.state_dim     = n_x;
  ocp.control_dim   = n_u;
  ocp.dt            = dt;
  ocp.horizon_steps = T;
  ocp.initial_state = Eigen::VectorXd::Random( n_x );

  Eigen::MatrixXd A = Eigen::MatrixXd::Identity( n_x, n_x );
  Eigen::MatrixXd B = Eigen::MatrixXd::Identity( n_x, n_u );
  ocp.dynamics      = [A, B]( const State& x, const Control& u ) { return A * x + B * u; };

  Eigen::MatrixXd Q = Eigen::MatrixXd::Identity( n_x, n_x );
  Eigen::MatrixXd R = Eigen::MatrixXd::Identity( n_u, n_u );
  ocp.stage_cost    = [Q, R]( const State& x, const Control& u, std::size_t ) {
    return ( x.transpose() * Q * x ).value() + ( u.transpose() * R * u ).value();
  };
  ocp.terminal_cost = []( const State& ) { return 0.0; };

  ocp.initialize_problem();
  ocp.verify_problem();
  return ocp;
}

struct Options
{
  bool        show_help   = false;
  int         agents      = 10;
  int         max_outer   = 10;
  std::string solver      = "ilqr";
  std::string strategy    = "centralized";
};

namespace
{

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
    auto        match_with_value = [&]( const std::string& name, std::string& out ) {
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
      options.agents      = parse_int( "agents", arg );
      positional_agents   = true;
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
  std::cout << "Usage: multi_agent_lqr [--agents N] [--solver NAME] [--strategy NAME] [--max-outer N]\n";
  std::cout << "       multi_agent_lqr N\n";
  std::cout << '\n';
  examples::print_available( std::cout );
}

} // namespace

/*────────────────────────────  main  ──────────────────────────────*/
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

    constexpr int    n_x = 4, n_u = 4, T = 10;
    constexpr double dt  = 0.1;

    MultiAgentProblem problem;
    for( int i = 0; i < options.agents; ++i )
    {
      auto ocp = std::make_shared<OCP>( create_linear_lqr_ocp( n_x, n_u, dt, T ) );
      problem.add_agent( std::make_shared<Agent>( i, ocp ) );
    }

    SolverParams params{
      { "max_iterations",  100 },
      {      "tolerance", 1e-5 },
      {         "max_ms",  100 }
    };

    auto      solver          = examples::make_solver( options.solver );
    Strategy  strategy        = examples::make_strategy( options.strategy, std::move( solver ), params, options.max_outer );
    const auto start          = std::chrono::steady_clock::now();
    const auto solution       = mas::solve( strategy, problem );
    const auto end            = std::chrono::steady_clock::now();
    const double elapsed_ms   = std::chrono::duration<double, std::milli>( end - start ).count();
    const std::string solver_name   = examples::canonical_solver_name( options.solver );
    const std::string strategy_name = examples::canonical_strategy_name( options.strategy );

    std::cout << std::fixed << std::setprecision( 6 )
              << "solver=" << solver_name
              << " strategy=" << strategy_name
              << " agents=" << options.agents
              << " cost=" << solution.total_cost
              << " time_ms=" << elapsed_ms
              << '\n';
  }
  catch( const std::exception& e )
  {
    std::cerr << "Error: " << e.what() << "\n";
    std::cerr << "Use --help to see available options.\n";
    return 1;
  }
  return 0;
}
