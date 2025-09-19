#include <chrono>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include "models/single_track_model.hpp"
#include "multi_agent_solver/ocp.hpp"
#include "multi_agent_solver/solvers/solver.hpp"
#include "multi_agent_solver/types.hpp"

#include "example_utils.hpp"

mas::OCP
create_single_track_lane_following_ocp()
{
  using namespace mas;
  OCP problem;

  // Dimensions
  problem.state_dim     = 4;
  problem.control_dim   = 2;
  problem.horizon_steps = 30;  // Example: 5 steps for testing.
  problem.dt            = 0.1; // 0.1 s per step.

  // Initial state: for example, X=1, Y=1, psi=1, vx=1
  problem.initial_state = Eigen::VectorXd::Zero( problem.state_dim );
  problem.initial_state << 0.0, 1.0, 0.0, 0.0;

  // Dynamics: use the dynamic_bicycle_model defined in your code.
  problem.dynamics = single_track_model;

  // Desired velocity.
  const double desired_velocity = 1.0; // [m/s]

  // Cost weights.
  const double w_lane  = 1.0; // Penalize lateral deviation.
  const double w_speed = 1.0; // Penalize speed error.
  const double w_delta = 0.1; // Penalize steering.
  const double w_acc   = 0.1; // Penalize acceleration.

  // Stage cost function.
  problem.stage_cost = [=]( const State& state, const Control& control, size_t idx ) -> double {
    // Unpack state: we only use Y (index 1) and vx (index 3).
    double y  = state( 1 );
    double vx = state( 3 );

    // Unpack control: steering delta and acceleration.
    double delta = control( 0 );
    double a_cmd = control( 1 );

    // Compute errors.
    double lane_error  = y;
    double speed_error = ( vx - desired_velocity );

    double cost = w_lane * ( lane_error * lane_error ) + w_speed * ( speed_error * speed_error ) + w_delta * ( delta * delta )
                + w_acc * ( a_cmd * a_cmd );
    return cost;
  };

  // Terminal cost (set to zero here, can be modified if needed).
  problem.terminal_cost       = [=]( const State& state ) -> double { return 0.0; };
  problem.cost_state_gradient = [=]( const StageCostFunction&, const State& state, const Control&, size_t time_idx ) -> Eigen::VectorXd {
    Eigen::VectorXd grad = Eigen::VectorXd::Zero( state.size() );
    // Only Y (index 1) and vx (index 3) appear in the cost.
    grad( 1 ) = 2.0 * w_lane * state( 1 );
    grad( 3 ) = 2.0 * w_speed * ( state( 3 ) - desired_velocity );
    return grad;
  };

  // Gradient with respect to control.
  problem.cost_control_gradient = [=]( const StageCostFunction&, const State&, const Control& control,
                                       size_t time_idx ) -> Eigen::VectorXd {
    Eigen::VectorXd grad = Eigen::VectorXd::Zero( control.size() );
    grad( 0 )            = 2.0 * w_delta * control( 0 );
    grad( 1 )            = 2.0 * w_acc * control( 1 );
    return grad;
  };

  // Hessian with respect to state.
  problem.cost_state_hessian = [=]( const StageCostFunction&, const State& state, const Control&, size_t ) -> Eigen::MatrixXd {
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero( state.size(), state.size() );
    H( 1, 1 )         = 2.0 * w_lane;
    H( 3, 3 )         = 2.0 * w_speed;
    return H;
  };

  // Hessian with respect to control.
  problem.cost_control_hessian = [=]( const StageCostFunction&, const State&, const Control&, size_t time_idx ) -> Eigen::MatrixXd {
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero( 2, 2 );
    H( 0, 0 )         = 2.0 * w_delta;
    H( 1, 1 )         = 2.0 * w_acc;
    return H;
  };


  problem.dynamics_state_jacobian = []( const MotionModel& /*dyn*/, const State& x, const Control& u ) -> Eigen::MatrixXd {
    return single_track_state_jacobian( x, u );
  };
  problem.dynamics_control_jacobian = []( const MotionModel& /*dyn*/, const State& x, const Control& u ) -> Eigen::MatrixXd {
    return single_track_control_jacobian( x, u );
  };


  Eigen::VectorXd lower_bounds( 2 ), upper_bounds( 2 );
  lower_bounds << -0.7, -1.0;
  upper_bounds << 0.7, 1.0;
  problem.input_lower_bounds = lower_bounds;
  problem.input_upper_bounds = upper_bounds;

  // Initialize and verify the problem.
  problem.initialize_problem();
  problem.verify_problem();

  return problem;
}

struct Options
{
  bool        show_help = false;
  std::string solver    = "ilqr";
};

namespace
{

Options
parse_options( int argc, char** argv )
{
  Options options;
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
    if( match_with_value( "--solver", value ) )
    {
      options.solver = value;
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
  std::cout << "Usage: single_track_ocp [--solver NAME]\n";
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

    OCP problem = create_single_track_lane_following_ocp();

    SolverParams params;
    params["max_iterations"] = 10;
    params["tolerance"]      = 1e-5;
    params["max_ms"]         = 100;

    auto solver = examples::make_solver( options.solver );
    mas::set_params( solver, params );

    const auto start        = std::chrono::steady_clock::now();
    mas::solve( solver, problem );
    const auto end          = std::chrono::steady_clock::now();
    const double elapsed_ms = std::chrono::duration<double, std::milli>( end - start ).count();

    const std::string solver_name = examples::canonical_solver_name( options.solver );
    std::cout << std::fixed << std::setprecision( 6 )
              << "solver=" << solver_name
              << " cost=" << problem.best_cost
              << " time_ms=" << elapsed_ms
              << '\n';

    examples::print_state_trajectory( std::cout, problem.best_states, problem.dt, "single_track" );
    examples::print_control_trajectory( std::cout, problem.best_controls, problem.dt, "single_track" );
  }
  catch( const std::exception& e )
  {
    std::cerr << "Error: " << e.what() << "\n";
    std::cerr << "Use --help to see available options.\n";
    return 1;
  }
  return 0;
}