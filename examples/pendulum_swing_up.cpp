#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include "models/pendulum_model.hpp"
#include "multi_agent_solver/ocp.hpp"
#include "multi_agent_solver/solvers/solver.hpp"
#include "multi_agent_solver/types.hpp"

#include "example_utils.hpp"

mas::OCP
create_pendulum_swingup_ocp()
{
  using namespace mas;
  OCP problem;

  problem.state_dim     = 2;
  problem.control_dim   = 1;
  problem.horizon_steps = 100;
  problem.dt            = 0.05;

  problem.initial_state = Eigen::Vector2d::Zero();

  problem.dynamics = pendulum_dynamics;

  const double theta_goal = M_PI;
  const double w_theta    = 10.0;
  const double w_omega    = 1.0;
  const double w_torque   = 0.01;
  const double torque_max = 2.0;

  problem.stage_cost = [=]( const State& x, const Control& u, size_t ) {
    double theta = x( 0 );
    double omega = x( 1 );
    double tau   = u( 0 );
    return w_theta * std::pow( theta - theta_goal, 2 ) + w_omega * std::pow( omega, 2 ) + w_torque * std::pow( tau, 2 );
  };

  problem.terminal_cost = [=]( const State& x ) {
    double theta = x( 0 );
    double omega = x( 1 );
    return w_theta * std::pow( theta - theta_goal, 2 ) + w_omega * std::pow( omega, 2 );
  };

  problem.cost_state_gradient = [=]( const StageCostFunction&, const State& x, const Control&, size_t ) {
    Eigen::Vector2d grad;
    grad( 0 ) = 2.0 * w_theta * ( x( 0 ) - theta_goal );
    grad( 1 ) = 2.0 * w_omega * x( 1 );
    return grad;
  };

  problem.cost_control_gradient = [=]( const StageCostFunction&, const State&, const Control& u, size_t ) {
    Eigen::VectorXd grad = Eigen::VectorXd::Zero( 1 );
    grad( 0 )            = 2.0 * w_torque * u( 0 );
    return grad;
  };

  problem.cost_state_hessian = [=]( const StageCostFunction&, const State&, const Control&, size_t ) {
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero( 2, 2 );
    H( 0, 0 )         = 2.0 * w_theta;
    H( 1, 1 )         = 2.0 * w_omega;
    return H;
  };

  problem.cost_control_hessian = [=]( const StageCostFunction&, const State&, const Control&, size_t ) {
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero( 1, 1 );
    H( 0, 0 )         = 2.0 * w_torque;
    return H;
  };

  problem.dynamics_state_jacobian = []( const MotionModel&, const State& x, const Control& u ) { return pendulum_state_jacobian( x, u ); };
  problem.dynamics_control_jacobian = []( const MotionModel&, const State& x, const Control& u ) {
    return pendulum_control_jacobian( x, u );
  };

  Eigen::VectorXd lower( 1 ), upper( 1 );
  lower << -torque_max;
  upper << torque_max;
  problem.input_lower_bounds = lower;
  problem.input_upper_bounds = upper;

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
    const Options options = parse_options( argc, argv );
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
  }
  catch( const std::exception& e )
  {
    std::cerr << "Error: " << e.what() << "\n";
    std::cerr << "Use --help to see available options.\n";
    return 1;
  }
  return 0;
}