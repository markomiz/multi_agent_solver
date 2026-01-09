#include <iostream>
#include <stdexcept>
#include <string>

#include "cli.hpp"
#include "example_utils.hpp"
#include "models/rocket_model.hpp"
#include "multi_agent_solver/ocp.hpp"
#include "multi_agent_solver/solvers/solver.hpp"
#include "multi_agent_solver/types.hpp"

// The Rocket Max Altitude problem optimizes the thrust profile of a vertical rocket
// to reach the maximum possible altitude given a limited fuel mass.
//
// State: [altitude, velocity, mass]
// Control: [thrust]
//
// Dynamics are governed by:
//   dh/dt = v
//   dv/dt = (thrust / mass) - g
//   dm/dt = -thrust / exhaust_velocity
//
// This is a constrained optimization problem:
// - Thrust is bounded [0, max_thrust]
// - Mass decreases over time
// - Altitude and velocity start at 0

namespace mas
{

mas::OCP
create_max_altitude_rocket_ocp()
{
  RocketParameters params;
  params.initial_mass     = 1.0;  // [kg]
  params.gravity          = 9.81; // [m/s^2]
  params.exhaust_velocity = 50.0; // [m/s] specific impulse related

  OCP problem;
  problem.state_dim     = 3;
  problem.control_dim   = 1;
  problem.horizon_steps = 50;
  problem.dt            = 0.1;

  problem.initial_state      = State::Zero( problem.state_dim );
  problem.initial_state( 2 ) = params.initial_mass;

  problem.dynamics = make_rocket_dynamics( params );

  const double max_thrust           = 20.0; // [N]

  // Weights for the cost function
  const double w_thrust             = 5e-3; // Small penalty on usage to smooth control
  const double w_terminal_altitude  = 15.0; // Large reward (negative cost) for final altitude
  const double w_terminal_velocity  = 2.0;  // Penalty for residual velocity at the peak
  const double desired_terminal_vel = 0.0;

  // Stage cost: Minimize control effort slightly (regularization)
  problem.stage_cost = [=]( const State& state, const Control& control, size_t ) {
    const double thrust = control( 0 );
    return 0.5 * w_thrust * thrust * thrust;
  };


  problem.cost_control_gradient = [=]( const StageCostFunction&, const State&, const Control& control, size_t ) {
    Eigen::VectorXd gradient( 1 );
    gradient( 0 ) = w_thrust * control( 0 );
    return gradient;
  };

  problem.cost_control_hessian = [=]( const StageCostFunction&, const State&, const Control&, size_t ) {
    Eigen::MatrixXd hessian( 1, 1 );
    hessian( 0, 0 ) = w_thrust;
    return hessian;
  };

  problem.cost_state_gradient = []( const StageCostFunction&, const State& state, const Control&, size_t ) {
    return Eigen::VectorXd::Zero( state.size() );
  };

  problem.cost_state_hessian = []( const StageCostFunction&, const State& state, const Control&, size_t ) {
    return Eigen::MatrixXd::Zero( state.size(), state.size() );
  };

  // Terminal cost: Maximize altitude (minimize -altitude)
  // Also try to reach zero velocity at the peak (apogee)
  problem.terminal_cost = [=]( const State& state ) {
    const double altitude       = state( 0 );
    const double velocity_error = state( 1 ) - desired_terminal_vel;
    return -w_terminal_altitude * altitude + 0.5 * w_terminal_velocity * velocity_error * velocity_error;
  };

  problem.terminal_cost_gradient = [=]( const TerminalCostFunction&, const State& state ) {
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero( state.size() );
    gradient( 0 )            = -w_terminal_altitude;
    gradient( 1 )            = w_terminal_velocity * ( state( 1 ) - desired_terminal_vel );
    return gradient;
  };

  problem.terminal_cost_hessian = [=]( const TerminalCostFunction&, const State& state ) {
    Eigen::MatrixXd hessian = Eigen::MatrixXd::Zero( state.size(), state.size() );
    hessian( 1, 1 )         = w_terminal_velocity;
    return hessian;
  };


  problem.dynamics_state_jacobian = [=]( const MotionModel&, const State& state, const Control& control ) {
    return rocket_state_jacobian( params, state, control );
  };

  problem.dynamics_control_jacobian = [=]( const MotionModel&, const State& state, const Control& control ) {
    return rocket_control_jacobian( params, state, control );
  };

  // Constraints
  Eigen::VectorXd u_lower( 1 ), u_upper( 1 );
  u_lower << 0.0;
  u_upper << max_thrust;
  problem.input_lower_bounds = u_lower;
  problem.input_upper_bounds = u_upper;

  Eigen::VectorXd x_lower    = Eigen::VectorXd::Constant( problem.state_dim, std::numeric_limits<double>::min() );
  x_lower( 2 )               = 0.0; // Mass cannot be negative
  problem.state_lower_bounds = x_lower;

  Eigen::VectorXd x_upper    = Eigen::VectorXd::Constant( problem.state_dim, std::numeric_limits<double>::max() );
  x_upper( 2 )               = params.initial_mass; // Mass cannot exceed initial mass
  problem.state_upper_bounds = x_upper;

  // initialize controls with just constant steady thrust
  problem.initial_controls = ControlTrajectory::Constant( problem.control_dim, problem.horizon_steps, max_thrust / 2.0 );

  problem.initialize_problem();
  problem.verify_problem();

  return problem;
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
    const examples::cli::RocketOptions options = examples::cli::parse_rocket_options( argc, argv );
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
