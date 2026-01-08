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
// Dynamics (0=Up):
//   theta_ddot = (g/l) * sin(theta) + u / (m*l^2) - b*omega
//
// Energy:
//   T = 0.5 * m * l^2 * omega^2
//   V = m * g * l * cos(theta)
//   E_des = m * g * l (at theta=0)

mas::OCP create_pendulum_swingup_ocp()
{
  using namespace mas;

  OCP problem;
  problem.state_dim     = 2;
  problem.control_dim   = 1;
  problem.horizon_steps = 100;   // 5 seconds
  problem.dt            = 0.05;

  // Downward is theta=pi in your convention; add small perturbation.
  problem.initial_state = Eigen::Vector2d(M_PI - 0.05, 0.0);

  problem.dynamics = pendulum_dynamics;

  const double g = 9.81;
  const double l = 1.0;
  const double m = 1.0;

  const double mgl   = m * g * l;
  const double E_des = mgl;  // energy at upright with omega=0

  // ---- Weights (tune these first) ----
  const double w_energy = 5.0;     // keep but reduce vs. before
  const double w_u      = 0.05;
  const double w_shape  = 2.0;     // stage "point toward upright"
  const double w_omega  = 0.05;    // stage damping (small)

  const double wT_pos   = 500.0;   // terminal upright
  const double wT_vel   = 100.0;   // terminal zero omega

  problem.stage_cost = [=](const State& x, const Control& u, size_t /*k*/) {
    const double theta  = x(0);
    const double omega  = x(1);
    const double torque = u(0);

    // Energy
    const double T = 0.5 * m * l * l * omega * omega;
    const double V = mgl * std::cos(theta);
    const double E = T + V;

    const double energy_error = (E - E_des) / mgl;

    // Upright shaping: theta=0 is upright => 1 - cos(theta) is 0 at upright, smooth & periodic
    const double upright_error = 1.0 - std::cos(theta);

    return w_energy * energy_error * energy_error
         + w_shape  * upright_error
         + w_omega  * omega * omega
         + w_u      * torque * torque;
  };

  problem.terminal_cost = [=](const State& x) {
    const double theta = x(0);
    const double omega = x(1);

    const double upright_error = 1.0 - std::cos(theta); // 0 at theta=0
    return wT_pos * upright_error + wT_vel * omega * omega;
  };

  // Input constraints
  const double torque_max = 5.0;
  problem.input_lower_bounds = Eigen::VectorXd::Constant(1, -torque_max);
  problem.input_upper_bounds = Eigen::VectorXd::Constant(1,  torque_max);

  // Initial guess (keep zero if you want, but a tiny sinusoid often helps)
  problem.initial_controls =
      ControlTrajectory::Zero(problem.control_dim, problem.horizon_steps);

  // If you’re willing to seed slightly:
  // for (int k = 0; k < problem.horizon_steps; ++k) {
  //   const double t = k * problem.dt;
  //   problem.initial_controls(0, k) = 0.2 * torque_max * std::sin(2.0 * M_PI * t);
  // }

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
