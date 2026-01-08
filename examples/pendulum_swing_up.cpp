#include <cmath>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include "cli.hpp"
#include "example_utils.hpp"
#include "models/pendulum_model.hpp"
#include "multi_agent_solver/ocp.hpp"
#include "multi_agent_solver/solvers/solver.hpp"
#include "multi_agent_solver/types.hpp"

// The Pendulum Swing-Up problem is a classic control challenge.
// The goal is to swing a pendulum from a resting position (hanging down)
// to an inverted position (pointing up) and balance it there.
//
// State: [theta, omega]
//   theta: angle (0 = hanging down, pi = pointing up)
//   omega: angular velocity
// Control: [torque]
//   torque: applied at the pivot
//
// The cost function penalizes:
// 1. Deviation from the goal angle (pi)
// 2. High angular velocity (we want to stop at the top)
// 3. Excessive control effort (energy efficiency)

mas::OCP
create_pendulum_swingup_ocp()
{
  using namespace mas;
  OCP problem;

  problem.state_dim     = 2;
  problem.control_dim   = 1;
  problem.horizon_steps = 200;
  problem.dt            = 0.05;

  // Start hanging down (0 angle, 0 velocity)
  problem.initial_state = Eigen::Vector2d::Zero();

  problem.dynamics = pendulum_dynamics;

  const double theta_goal = M_PI;
  const double w_theta    = 1.0;
  const double w_omega    = 0.1;
  const double w_torque   = 0.01;
  const double torque_max = 10.0;

  // Stage cost: integrated over the trajectory
  // J = sum( w_theta * (theta - goal)^2 + w_omega * omega^2 + w_torque * torque^2 )
  problem.stage_cost = [=]( const State& x, const Control& u, size_t t_idx ) {
    double theta = x( 0 );
    double omega = x( 1 );
    double tau   = u( 0 );
    // Scaled down to allow more flexibility during the swing-up
    return ( w_theta * std::pow( theta - theta_goal, 2 ) + w_omega * std::pow( omega, 2 ) + w_torque * std::pow( tau, 2 ) ) / 100.0;
  };

  // Terminal cost: heavily penalize not being at the goal at the end
  // Weights are significantly higher to "lock" the pendulum at the top
  problem.terminal_cost = [=]( const State& x ) {
    const double term_w_theta = 10000.0;
    const double term_w_omega = 100.0;
    double theta = x( 0 );
    double omega = x( 1 );
    return term_w_theta * std::pow( theta - theta_goal, 2 ) + term_w_omega * std::pow( omega, 2 );
  };

  // Note: Gradients and Hessians can be provided manually for better performance,
  // but the solver can also use automatic differentiation or finite differences if they are not provided.
  // In this example, we rely on the solver's internal differentiation for simplicity.

  Eigen::VectorXd lower( 1 ), upper( 1 );
  lower << -torque_max;
  upper << torque_max;
  problem.input_lower_bounds = lower;
  problem.input_upper_bounds = upper;

  problem.initialize_problem();
  problem.verify_problem();
  return problem;
}

using Options = examples::cli::SolverOptions;

namespace
{

void
print_usage()
{
  std::cout << "Usage: pendulum_swing_up [--solver NAME]\n";
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
    const Options options = examples::cli::parse_solver_options( argc, argv );
    if( options.show_help )
    {
      print_usage();
      return 0;
    }

    OCP problem = create_pendulum_swingup_ocp();

    SolverParams params;
    params["max_iterations"] = 500;
    params["tolerance"]      = 1e-5;
    params["max_ms"]         = 1000;

    auto solver = examples::make_solver( options.solver );
    mas::set_params( solver, params );

    const auto start = std::chrono::steady_clock::now();
    mas::solve( solver, problem );
    const auto   end        = std::chrono::steady_clock::now();
    const double elapsed_ms = std::chrono::duration<double, std::milli>( end - start ).count();

    const std::string solver_name = examples::canonical_solver_name( options.solver );
    std::cout << std::fixed << std::setprecision( 6 ) << "solver=" << solver_name << " cost=" << problem.best_cost
              << " time_ms=" << elapsed_ms << '\n';

    examples::print_state_trajectory( std::cout, problem.best_states, problem.dt, "pendulum" );
    examples::print_control_trajectory( std::cout, problem.best_controls, problem.dt, "pendulum" );
  }
  catch( const std::exception& e )
  {
    std::cerr << "Error: " << e.what() << "\n";
    std::cerr << "Use --help to see available options.\n";
    return 1;
  }
  return 0;
}
