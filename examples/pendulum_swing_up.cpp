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

// The Pendulum Swing-Up problem is reformulated here using Energy Shaping.
//
// Goal: Swing the pendulum from the stable downward position (theta=pi)
//       to the unstable upward position (theta=0).
//
// Dynamics (0=Up, +sin):
//   theta_ddot = (g/l) * sin(theta) + u / (m*l^2) - (b/ml^2)*omega
//
// Energy:
//   E = Kinetic + Potential
//   T = 0.5 * m * l^2 * omega^2
//   V = m * g * l * cos(theta)  (Max at 0, Min at pi)
//   E_des = m * g * l (at theta=0)

mas::OCP
create_pendulum_swingup_ocp()
{
  using namespace mas;
  OCP problem;

  problem.state_dim     = 2;
  problem.control_dim   = 1;
  problem.horizon_steps = 100; // 5 seconds
  problem.dt            = 0.05;

  // Start hanging down (stable equilibrium)
  // Perturb slightly to ensure non-zero gradients for the energy cost
  // theta = pi is down.
  problem.initial_state = Eigen::Vector2d( M_PI - 0.05, 0.0 );

  problem.dynamics = pendulum_dynamics;

  const double g = 9.81;
  const double l = 1.0;
  const double m = 1.0;
  const double E_des = m * g * l; // Potential energy at upright (theta=0)

  // Weights
  const double w_energy = 10.0;
  const double w_ctrl   = 0.1;
  const double w_omega  = 0.0;

  const double term_w_pos = 500.0;
  const double term_w_vel = 100.0;

  problem.stage_cost = [=]( const State& x, const Control& u, size_t ) {
    double theta = x( 0 );
    double omega = x( 1 );
    double torque = u( 0 );

    // Current Energy
    double T = 0.5 * m * l * l * omega * omega;
    double V = m * g * l * std::cos( theta );
    double E = T + V;

    // Normalized Energy Error: (E - E_des) / mgl
    // Denominator = m*g*l = 9.81
    double energy_error = (E - E_des) / (m * g * l);

    return w_energy * energy_error * energy_error
         + w_ctrl * torque * torque
         + w_omega * omega * omega;
  };

  // Terminal cost: Removed as requested to rely on energy shaping stage cost
  problem.terminal_cost = [=]( const State& ) {
    return 0.0;
  };

  // Input constraints (Voltage/Torque limit)
  // Let's use a limit that requires swinging (cannot lift statically).
  // Static torque required is m*g*l = 9.81.
  // Limit to 5.0 to force energy pumping.
  const double torque_max = 5.0;
  Eigen::VectorXd lower( 1 ), upper( 1 );
  lower << -torque_max;
  upper << torque_max;
  problem.input_lower_bounds = lower;
  problem.input_upper_bounds = upper;

  // No initial control guess (zero) to test robust formulation
  problem.initial_controls = ControlTrajectory::Zero( problem.control_dim, problem.horizon_steps );

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
    params["max_iterations"] = 1000;
    params["tolerance"]      = 1e-4;
    params["max_ms"]         = 5000;

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
