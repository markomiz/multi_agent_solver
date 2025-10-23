#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "example_utils.hpp"
#include "models/rocket_model.hpp"
#include "multi_agent_solver/ocp.hpp"
#include "multi_agent_solver/solvers/solver.hpp"
#include "multi_agent_solver/types.hpp"

namespace
{

template<typename Scalar>
mas::OCP<Scalar>
create_max_altitude_rocket_ocp()
{
  using OCP     = mas::OCP<Scalar>;
  using State   = typename OCP::State;
  using Control = typename OCP::Control;

  mas::RocketParametersT<Scalar> params;
  params.initial_mass     = static_cast<Scalar>( 1.0 );
  params.gravity          = static_cast<Scalar>( 9.81 );
  params.exhaust_velocity = static_cast<Scalar>( 50.0 );

  OCP problem;
  problem.state_dim     = 3;
  problem.control_dim   = 1;
  problem.horizon_steps = 50;
  problem.dt            = static_cast<Scalar>( 0.1 );

  problem.initial_state      = State::Zero( problem.state_dim );
  problem.initial_state( 2 ) = params.initial_mass;

  problem.dynamics = mas::make_rocket_dynamics<Scalar>( params );

  const Scalar max_thrust           = static_cast<Scalar>( 20.0 );
  const Scalar w_thrust             = static_cast<Scalar>( 5e-3 );
  const Scalar w_terminal_altitude  = static_cast<Scalar>( 15.0 );
  const Scalar w_terminal_velocity  = static_cast<Scalar>( 2.0 );
  const Scalar desired_terminal_vel = static_cast<Scalar>( 0.0 );

  problem.stage_cost = [=]( const State&, const Control& control, std::size_t ) {
    const Scalar thrust = control( 0 );
    return static_cast<Scalar>( 0.5 ) * w_thrust * thrust * thrust;
  };

  problem.cost_control_gradient = [=]( const typename OCP::StageCostFunction&, const State&, const Control& control,
                                       std::size_t ) {
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> gradient( 1 );
    gradient( 0 ) = w_thrust * control( 0 );
    return gradient;
  };

  problem.cost_control_hessian = [=]( const typename OCP::StageCostFunction&, const State&, const Control&, std::size_t ) {
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> hessian( 1, 1 );
    hessian( 0, 0 ) = w_thrust;
    return hessian;
  };

  problem.cost_state_gradient = []( const typename OCP::StageCostFunction&, const State& state, const Control&, std::size_t ) {
    return Eigen::Matrix<Scalar, Eigen::Dynamic, 1>::Zero( state.size() );
  };

  problem.cost_state_hessian = []( const typename OCP::StageCostFunction&, const State& state, const Control&, std::size_t ) {
    return Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Zero( state.size(), state.size() );
  };

  problem.terminal_cost = [=]( const State& state ) {
    const Scalar altitude       = state( 0 );
    const Scalar velocity_error = state( 1 ) - desired_terminal_vel;
    return -w_terminal_altitude * altitude + static_cast<Scalar>( 0.5 ) * w_terminal_velocity * velocity_error * velocity_error;
  };

  problem.terminal_cost_gradient = [=]( const typename OCP::TerminalCostFunction&, const State& state ) {
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> gradient = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>::Zero( state.size() );
    gradient( 0 )                                     = -w_terminal_altitude;
    gradient( 1 )                                     = w_terminal_velocity * ( state( 1 ) - desired_terminal_vel );
    return gradient;
  };

  problem.terminal_cost_hessian = [=]( const typename OCP::TerminalCostFunction&, const State& state ) {
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> hessian
      = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Zero( state.size(), state.size() );
    hessian( 1, 1 ) = w_terminal_velocity;
    return hessian;
  };

  problem.dynamics_state_jacobian = [=]( const typename OCP::MotionModel&, const State& state, const Control& control ) {
    return mas::rocket_state_jacobian<Scalar>( params, state, control );
  };

  problem.dynamics_control_jacobian = [=]( const typename OCP::MotionModel&, const State& state, const Control& control ) {
    return mas::rocket_control_jacobian<Scalar>( params, state, control );
  };

  Control u_lower( 1 ), u_upper( 1 );
  u_lower << static_cast<Scalar>( 0.0 );
  u_upper << max_thrust;
  problem.input_lower_bounds = u_lower;
  problem.input_upper_bounds = u_upper;

  State x_lower = State::Constant( problem.state_dim, std::numeric_limits<Scalar>::lowest() );
  x_lower( 2 )  = static_cast<Scalar>( 0.0 );
  problem.state_lower_bounds = x_lower;

  State x_upper = State::Constant( problem.state_dim, std::numeric_limits<Scalar>::max() );
  x_upper( 2 )  = params.initial_mass;
  problem.state_upper_bounds = x_upper;

  problem.initial_controls
    = OCP::ControlTrajectory::Constant( problem.control_dim, problem.horizon_steps,
                                        max_thrust / static_cast<Scalar>( 2.0 ) );

  problem.initialize_problem();
  problem.verify_problem();

  return problem;
}

struct Options
{
  bool                     show_help         = false;
  bool                     dump_trajectories = false;
  std::vector<std::string> solvers;
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

Options
parse_options( int argc, char** argv )
{
  Options options;
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
    if( arg == "--dump" || arg == "--dump-trajectories" )
    {
      options.dump_trajectories = true;
      continue;
    }

    std::string value;
    if( match_with_value( "--solver", value ) )
    {
      append_list( options.solvers, value, examples::canonical_solver_name );
    }
    else if( match_with_value( "--solvers", value ) )
    {
      append_list( options.solvers, value, examples::canonical_solver_name );
    }
    else if( match_with_value( "--scalar", value ) )
    {
      append_list( options.scalars, value, examples::canonical_scalar_name );
    }
    else if( match_with_value( "--scalars", value ) )
    {
      append_list( options.scalars, value, examples::canonical_scalar_name );
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
  std::cout << "Usage: rocket_max_altitude [--solvers NAMES] [--scalars float,double] [--dump]\n";
  std::cout << '\n';
  examples::print_available( std::cout );
}

template<typename Scalar>
void
run_for_scalar( const Options& options, const std::vector<std::string>& solver_names )
{
  const mas::SolverParamsT<Scalar> params{ { "max_iterations", static_cast<Scalar>( 25 ) },
                                           { "tolerance", static_cast<Scalar>( 1e-6 ) },
                                           { "max_ms", static_cast<Scalar>( 200 ) } };

  const std::string scalar_label = examples::scalar_label<Scalar>();

  for( const auto& solver_name : solver_names )
  {
    if( !examples::solver_supported_for_scalar<Scalar>( solver_name ) )
    {
      std::cout << "scalar=" << scalar_label << " solver=" << solver_name << " unsupported (skipping)\n";
      continue;
    }

    auto problem = create_max_altitude_rocket_ocp<Scalar>();
    auto solver  = examples::make_solver<Scalar>( solver_name );
    mas::set_params( solver, params );

    const auto start    = std::chrono::steady_clock::now();
    mas::solve( solver, problem );
    const auto end      = std::chrono::steady_clock::now();
    const double elapsed_ms = std::chrono::duration<double, std::milli>( end - start ).count();

    std::cout << std::fixed << std::setprecision( 6 )
              << "scalar=" << scalar_label
              << " solver=" << solver_name
              << " cost=" << static_cast<double>( problem.best_cost )
              << " time_ms=" << elapsed_ms
              << '\n';

    if( options.dump_trajectories )
    {
      examples::print_state_trajectory( std::cout, problem.best_states, problem.dt, "rocket" );
      examples::print_control_trajectory( std::cout, problem.best_controls, problem.dt, "rocket" );
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
    const std::vector<std::string> scalar_names
      = options.scalars.empty() ? std::vector<std::string>{ "float", "double" } : options.scalars;

    for( const auto& scalar_name : scalar_names )
    {
      if( scalar_name == "float" )
        run_for_scalar<float>( options, solver_names );
      else if( scalar_name == "double" )
        run_for_scalar<double>( options, solver_names );
      else
        std::cout << "Unknown scalar '" << scalar_name << "' -- skipping\n";
    }
  }
  catch( const std::exception& ex )
  {
    std::cerr << "Error: " << ex.what() << '\n';
    return 1;
  }
  return 0;
}
