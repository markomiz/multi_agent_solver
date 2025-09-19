#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include "example_utils.hpp"
#include "models/rocket_model.hpp"
#include "multi_agent_solver/ocp.hpp"
#include "multi_agent_solver/solvers/solver.hpp"
#include "multi_agent_solver/types.hpp"

namespace
{

mas::OCP
create_max_altitude_rocket_ocp()
{
  using namespace mas;

  examples::RocketParameters params;
  params.mass    = 1.0;
  params.gravity = 9.81;

  OCP problem;
  problem.state_dim     = 2;
  problem.control_dim   = 1;
  problem.horizon_steps = 40;
  problem.dt            = 0.1;

  problem.initial_state = State::Zero( problem.state_dim );

  problem.dynamics = examples::make_rocket_dynamics( params );

  const double max_thrust            = 20.0; // [N]
  const double w_thrust              = 5e-3;
  const double w_terminal_altitude   = 15.0;
  const double w_terminal_velocity   = 2.0;
  const double desired_terminal_vel  = 0.0;

  problem.stage_cost = [=]( const State&, const Control& control, size_t ) {
    const double thrust = control( 0 );
    return 0.5 * w_thrust * thrust * thrust;
  };

  problem.cost_state_gradient = []( const StageCostFunction&, const State& state, const Control&, size_t ) {
    return Eigen::VectorXd::Zero( state.size() );
  };

  problem.cost_state_hessian = []( const StageCostFunction&, const State& state, const Control&, size_t ) {
    return Eigen::MatrixXd::Zero( state.size(), state.size() );
  };

  problem.cost_control_gradient = [=]( const StageCostFunction&, const State&, const Control& control, size_t ) {
    Eigen::VectorXd grad = Eigen::VectorXd::Zero( control.size() );
    grad( 0 )            = w_thrust * control( 0 );
    return grad;
  };

  problem.cost_control_hessian = [=]( const StageCostFunction&, const State&, const Control&, size_t ) {
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero( 1, 1 );
    H( 0, 0 )         = w_thrust;
    return H;
  };

  problem.terminal_cost = [=]( const State& state ) {
    const double altitude        = state( 0 );
    const double velocity_error  = state( 1 ) - desired_terminal_vel;
    return -w_terminal_altitude * altitude + 0.5 * w_terminal_velocity * velocity_error * velocity_error;
  };

  problem.terminal_cost_gradient = [=]( const TerminalCostFunction&, const State& state ) {
    Eigen::VectorXd grad = Eigen::VectorXd::Zero( state.size() );
    grad( 0 )            = -w_terminal_altitude;
    grad( 1 )            = w_terminal_velocity * ( state( 1 ) - desired_terminal_vel );
    return grad;
  };

  problem.terminal_cost_hessian = [=]( const TerminalCostFunction&, const State& state ) {
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero( state.size(), state.size() );
    H( 1, 1 )         = w_terminal_velocity;
    return H;
  };

  problem.dynamics_state_jacobian = [=]( const MotionModel&, const State& state, const Control& control ) {
    return examples::rocket_state_jacobian( params, state, control );
  };

  problem.dynamics_control_jacobian = [=]( const MotionModel&, const State& state, const Control& control ) {
    return examples::rocket_control_jacobian( params, state, control );
  };

  Eigen::VectorXd u_lower( 1 ), u_upper( 1 );
  u_lower << 0.0;
  u_upper << max_thrust;
  problem.input_lower_bounds = u_lower;
  problem.input_upper_bounds = u_upper;

  problem.initialize_problem();
  problem.verify_problem();

  return problem;
}

struct Options
{
  bool        show_help   = false;
  bool        dump_traces = false;
  std::string solver      = "osqp";
};

Options
parse_options( int argc, char** argv )
{
  Options options;
  for( int i = 1; i < argc; ++i )
  {
    const std::string arg = argv[i];
    if( arg == "--help" || arg == "-h" )
    {
      options.show_help = true;
      continue;
    }
    if( arg == "--dump" )
    {
      options.dump_traces = true;
      continue;
    }

    const std::string solver_prefix = "--solver=";
    if( arg.rfind( solver_prefix, 0 ) == 0 )
    {
      options.solver = arg.substr( solver_prefix.size() );
      continue;
    }

    if( arg == "--solver" )
    {
      if( i + 1 >= argc )
        throw std::invalid_argument( "Missing value for --solver" );
      options.solver = argv[++i];
      continue;
    }

    throw std::invalid_argument( "Unknown argument '" + arg + "'" );
  }
  return options;
}

void
print_usage()
{
  std::cout << "Usage: rocket_max_altitude [--solver NAME] [--dump]\n";
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

    OCP problem = create_max_altitude_rocket_ocp();

    SolverParams params;
    params["max_iterations"] = 25;
    params["tolerance"]      = 1e-6;
    params["max_ms"]         = 200;

    Solver solver = examples::make_solver( options.solver );
    mas::set_params( solver, params );
    mas::solve( solver, problem );

    const auto& X = problem.best_states;
    const int   T = problem.horizon_steps;

    const double final_altitude = X( 0, T );
    const double final_velocity = X( 1, T );

    std::cout << std::fixed << std::setprecision( 3 );
    std::cout << "Best cost: " << problem.best_cost << '\n';
    std::cout << "Final altitude: " << final_altitude << " m\n";
    std::cout << "Final velocity: " << final_velocity << " m/s\n";

    if( options.dump_traces )
    {
      examples::print_state_trajectory( std::cout, X, problem.dt, "rocket" );
      examples::print_control_trajectory( std::cout, problem.best_controls, problem.dt, "rocket" );
    }
  }
  catch( const std::exception& ex )
  {
    std::cerr << "Error: " << ex.what() << '\n';
    return 1;
  }
  return 0;
}
