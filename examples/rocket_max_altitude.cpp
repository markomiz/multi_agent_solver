#include <iostream>
#include <stdexcept>
#include <string>

#include "example_utils.hpp"
#include "models/rocket_model.hpp"
#include "multi_agent_solver/ocp.hpp"
#include "multi_agent_solver/solvers/solver.hpp"
#include "multi_agent_solver/types.hpp"

namespace mas
{

mas::OCP
create_max_altitude_rocket_ocp()
{
  RocketParameters params;
  params.initial_mass     = 1.0;
  params.gravity          = 9.81;
  params.exhaust_velocity = 50.0;

  OCP problem;
  problem.state_dim     = 3;
  problem.control_dim   = 1;
  problem.horizon_steps = 50;
  problem.dt            = 0.1;

  problem.initial_state      = State::Zero( problem.state_dim );
  problem.initial_state( 2 ) = params.initial_mass;

  problem.dynamics = make_rocket_dynamics( params );

  const double max_thrust           = 20.0; // [N]
  const double w_thrust             = 5e-3;
  const double w_terminal_altitude  = 15.0;
  const double w_terminal_velocity  = 2.0;
  const double desired_terminal_vel = 0.0;

  problem.stage_cost = [=]( const State& state, const Control& control, size_t ) {
    const double thrust = control( 0 );
    return 0.5 * w_thrust * thrust * thrust;
  };


  problem.terminal_cost = [=]( const State& state ) {
    const double altitude       = state( 0 );
    const double velocity_error = state( 1 ) - desired_terminal_vel;
    const double alt_sign       = altitude >= 0.0 ? 1.0 : -1.0;
    return -w_terminal_altitude * altitude * altitude * alt_sign + 0.5 * w_terminal_velocity * velocity_error * velocity_error
         + state( 2 ) * state( 2 ) * 1e2;
  };


  problem.dynamics_state_jacobian = [=]( const MotionModel&, const State& state, const Control& control ) {
    return rocket_state_jacobian( params, state, control );
  };

  problem.dynamics_control_jacobian = [=]( const MotionModel&, const State& state, const Control& control ) {
    return rocket_control_jacobian( params, state, control );
  };

  Eigen::VectorXd u_lower( 1 ), u_upper( 1 );
  u_lower << 0.0;
  u_upper << max_thrust;
  problem.input_lower_bounds = u_lower;
  problem.input_upper_bounds = u_upper;

  Eigen::VectorXd x_lower    = Eigen::VectorXd::Constant( problem.state_dim, std::numeric_limits<double>::min() );
  x_lower( 2 )               = 0.0;
  problem.state_lower_bounds = x_lower;

  Eigen::VectorXd x_upper    = Eigen::VectorXd::Constant( problem.state_dim, std::numeric_limits<double>::max() );
  x_upper( 2 )               = params.initial_mass;
  problem.state_upper_bounds = x_upper;

  // initialize controls with just constant steady thrust
  problem.initial_controls = ControlTrajectory::Constant( problem.control_dim, problem.horizon_steps, max_thrust / 2.0 );

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

} // namespace mas

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
    set_params( solver, params );
    const auto start = std::chrono::steady_clock::now();
    mas::solve( solver, problem );
    const auto   end        = std::chrono::steady_clock::now();
    const double elapsed_ms = std::chrono::duration<double, std::milli>( end - start ).count();
    const auto&  X          = problem.best_states;
    const int    T          = problem.horizon_steps;

    const double final_altitude = X( 0, T );
    const double final_velocity = X( 1, T );
    const double final_mass     = X( 2, T );

    const std::string solver_name = examples::canonical_solver_name( options.solver );
    std::cout << std::fixed << std::setprecision( 6 ) << "solver=" << solver_name << " cost=" << problem.best_cost
              << " time_ms=" << elapsed_ms << '\n';


    examples::print_state_trajectory( std::cout, problem.best_states, problem.dt, "rocket" );
    examples::print_control_trajectory( std::cout, problem.best_controls, problem.dt, "rocket" );
  }
  catch( const std::exception& ex )
  {
    std::cerr << "Error: " << ex.what() << '\n';
    return 1;
  }
  return 0;
}