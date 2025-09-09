#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>

#include "models/pendulum_model.hpp"
#include "multi_agent_solver/ocp.hpp"
#include "multi_agent_solver/solvers/solver.hpp"
#include "multi_agent_solver/types.hpp"

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

int
main( int, char** )
{
  using namespace mas;
  OCP problem = create_pendulum_swingup_ocp();

  SolverParams params;
  params["max_iterations"] = 500;
  params["tolerance"]      = 1e-5;
  params["max_ms"]         = 1000;

  std::map<std::string, Solver> solvers;
  solvers.emplace( "iLQR", iLQR() );
  solvers.emplace( "CGD", CGD() );
#ifdef MAS_HAVE_OSQP
  solvers.emplace( "OSQP", OSQP() );
  solvers.emplace( "OSQP Collocation", OSQPCollocation() );
#endif

  struct SolverResult
  {
    double cost;
    double time_ms;
  };

  std::map<std::string, SolverResult> results;

  for( auto& [name, solver] : solvers )
  {
    auto problem_copy = problem;
    auto start        = std::chrono::high_resolution_clock::now();
    mas::set_params( solver, params );
    mas::solve( solver, problem_copy );
    auto end      = std::chrono::high_resolution_clock::now();
    results[name] = { problem_copy.best_cost, std::chrono::duration<double, std::milli>( end - start ).count() };
  }

  std::cout << "\nPendulum Swing-Up Test\n"
            << "---------------------------------------------\n"
            << std::left << std::setw( 20 ) << "Solver" << std::setw( 15 ) << "Cost" << std::setw( 15 ) << "Time (ms)\n"
            << "---------------------------------------------\n";
  for( const auto& [name, res] : results )
    std::cout << std::left << std::setw( 20 ) << name << std::setw( 15 ) << res.cost << std::setw( 15 ) << res.time_ms << '\n';
}