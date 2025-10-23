#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "example_utils.hpp"
#include "models/single_track_model.hpp"
#include "multi_agent_solver/ocp.hpp"
#include "multi_agent_solver/solvers/solver.hpp"
#include "multi_agent_solver/types.hpp"

namespace
{

template<typename Scalar>
mas::OCP<Scalar>
create_single_track_lane_following_ocp()
{
  using OCP     = mas::OCP<Scalar>;
  using State   = typename OCP::State;
  using Control = typename OCP::Control;

  OCP problem;
  problem.state_dim     = 4;
  problem.control_dim   = 2;
  problem.horizon_steps = 80;
  problem.dt            = static_cast<Scalar>( 0.1 );

  State initial_state = State::Zero( problem.state_dim );
  initial_state << static_cast<Scalar>( 0.0 ), static_cast<Scalar>( 1.0 ), static_cast<Scalar>( 0.0 ), static_cast<Scalar>( 0.0 );
  problem.initial_state = initial_state;

  problem.dynamics = mas::single_track_model<Scalar>;

  const Scalar desired_velocity = static_cast<Scalar>( 1.0 );
  const Scalar w_lane           = static_cast<Scalar>( 10.0 );
  const Scalar w_speed          = static_cast<Scalar>( 1.0 );
  const Scalar w_delta          = static_cast<Scalar>( 0.1 );
  const Scalar w_acc            = static_cast<Scalar>( 0.1 );

  problem.stage_cost = [=]( const State& state, const Control& control, std::size_t ) {
    const Scalar y         = state( 1 );
    const Scalar vx        = state( 3 );
    const Scalar delta     = control( 0 );
    const Scalar a_command = control( 1 );

    const Scalar lane_error  = y;
    const Scalar speed_error = vx - desired_velocity;

    return w_lane * lane_error * lane_error + w_speed * speed_error * speed_error + w_delta * delta * delta
         + w_acc * a_command * a_command;
  };

  problem.terminal_cost = []( const State& ) { return static_cast<Scalar>( 0 ); };

  problem.cost_state_gradient
    = [=]( const typename OCP::StageCostFunction&, const State& state, const Control&, std::size_t ) {
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> grad = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>::Zero( state.size() );
        grad( 1 )                                     = static_cast<Scalar>( 2.0 ) * w_lane * state( 1 );
        grad( 3 )                                     = static_cast<Scalar>( 2.0 ) * w_speed * ( state( 3 ) - desired_velocity );
        return grad;
      };

  problem.cost_control_gradient
    = [=]( const typename OCP::StageCostFunction&, const State&, const Control& control, std::size_t ) {
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> grad = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>::Zero( control.size() );
        grad( 0 )                                     = static_cast<Scalar>( 2.0 ) * w_delta * control( 0 );
        grad( 1 )                                     = static_cast<Scalar>( 2.0 ) * w_acc * control( 1 );
        return grad;
      };

  problem.cost_state_hessian
    = [=]( const typename OCP::StageCostFunction&, const State& state, const Control&, std::size_t ) {
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> H
          = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Zero( state.size(), state.size() );
        H( 1, 1 ) = static_cast<Scalar>( 2.0 ) * w_lane;
        H( 3, 3 ) = static_cast<Scalar>( 2.0 ) * w_speed;
        return H;
      };

  problem.cost_control_hessian
    = [=]( const typename OCP::StageCostFunction&, const State&, const Control&, std::size_t ) {
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> H
          = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Zero( 2, 2 );
        H( 0, 0 ) = static_cast<Scalar>( 2.0 ) * w_delta;
        H( 1, 1 ) = static_cast<Scalar>( 2.0 ) * w_acc;
        return H;
      };

  problem.dynamics_state_jacobian = []( const typename OCP::MotionModel&, const State& x, const Control& u ) {
    return mas::single_track_state_jacobian<Scalar>( x, u );
  };

  problem.dynamics_control_jacobian = []( const typename OCP::MotionModel&, const State& x, const Control& u ) {
    return mas::single_track_control_jacobian<Scalar>( x, u );
  };

  Control lower_bounds( problem.control_dim ), upper_bounds( problem.control_dim );
  lower_bounds << static_cast<Scalar>( -0.7 ), static_cast<Scalar>( -1.0 );
  upper_bounds << static_cast<Scalar>( 0.7 ), static_cast<Scalar>( 1.0 );
  problem.input_lower_bounds = lower_bounds;
  problem.input_upper_bounds = upper_bounds;

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
    if( arg == "--dump-trajectories" )
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
  std::cout << "Usage: single_track_ocp [--solvers NAMES] [--scalars float,double] [--dump-trajectories]\n";
  std::cout << '\n';
  examples::print_available( std::cout );
}

template<typename Scalar>
void
run_for_scalar( const Options& options, const std::vector<std::string>& solver_names )
{
  const mas::SolverParamsT<Scalar> params{ { "max_iterations", static_cast<Scalar>( 10 ) },
                                           { "tolerance", static_cast<Scalar>( 1e-5 ) },
                                           { "max_ms", static_cast<Scalar>( 100 ) } };

  const std::string scalar_label = examples::scalar_label<Scalar>();

  for( const auto& solver_name : solver_names )
  {
    if( !examples::solver_supported_for_scalar<Scalar>( solver_name ) )
    {
      std::cout << "scalar=" << scalar_label << " solver=" << solver_name << " unsupported (skipping)\n";
      continue;
    }

    auto problem = create_single_track_lane_following_ocp<Scalar>();
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
      examples::print_state_trajectory( std::cout, problem.best_states, problem.dt, "single_track" );
      examples::print_control_trajectory( std::cout, problem.best_controls, problem.dt, "single_track" );
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
  catch( const std::exception& e )
  {
    std::cerr << "Error: " << e.what() << "\n";
    std::cerr << "Use --help to see available options.\n";
    return 1;
  }
  return 0;
}
