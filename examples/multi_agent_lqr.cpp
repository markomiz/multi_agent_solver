#include <algorithm>
#include <chrono>
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

#include "cli.hpp"
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
  Eigen::VectorXd initial_state = Eigen::VectorXd::Zero( n_x );
  if( n_x > 0 )
    initial_state[0] = 1.0;
  ocp.initial_state = initial_state;

  Eigen::MatrixXd A = Eigen::MatrixXd::Identity( n_x, n_x );
  Eigen::MatrixXd B = Eigen::MatrixXd::Identity( n_x, n_u );
  ocp.dynamics      = [A, B]( const State& x, const Control& u ) { return A * x + B * u; };
  ocp.dynamics_state_jacobian
    = [A]( const MotionModel&, const State&, const Control& ) { return A; };
  ocp.dynamics_control_jacobian
    = [B]( const MotionModel&, const State&, const Control& ) { return B; };

  Eigen::MatrixXd Q = Eigen::MatrixXd::Identity( n_x, n_x );
  Eigen::MatrixXd R = Eigen::MatrixXd::Identity( n_u, n_u );
  Eigen::MatrixXd        Qf     = Q;
  const Eigen::MatrixXd  Qt     = Q + Q.transpose();
  const Eigen::MatrixXd  Rt     = R + R.transpose();
  const Eigen::MatrixXd  Qf_sym = Qf + Qf.transpose();
  ocp.stage_cost             = [Q, R]( const State& x, const Control& u, std::size_t ) {
    return ( x.transpose() * Q * x ).value() + ( u.transpose() * R * u ).value();
  };
  ocp.cost_state_gradient = [Qt]( const StageCostFunction&, const State& x, const Control&, std::size_t ) {
    return Qt * x;
  };
  ocp.cost_control_gradient = [Rt]( const StageCostFunction&, const State&, const Control& u, std::size_t ) {
    return Rt * u;
  };
  ocp.cost_state_hessian = [Qt]( const StageCostFunction&, const State&, const Control&, std::size_t ) {
    return Qt;
  };
  ocp.cost_control_hessian = [Rt]( const StageCostFunction&, const State&, const Control&, std::size_t ) {
    return Rt;
  };
  ocp.cost_cross_term = [n_x, n_u]( const StageCostFunction&, const State&, const Control&, std::size_t ) {
    return Eigen::MatrixXd::Zero( n_u, n_x );
  };
  ocp.terminal_cost = [Qf]( const State& x ) { return ( x.transpose() * Qf * x ).value(); };
  ocp.terminal_cost_gradient
    = [Qf_sym]( const TerminalCostFunction&, const State& x ) { return Qf_sym * x; };
  ocp.terminal_cost_hessian
    = [Qf_sym]( const TerminalCostFunction&, const State& ) { return Qf_sym; };

  ocp.initialize_problem();
  ocp.verify_problem();
  return ocp;
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
