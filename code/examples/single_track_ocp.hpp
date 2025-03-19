#pragma once
#include <chrono>
#include <iostream>
#include <map>
#include <string>

#include "models/single_track_model.hpp"
#include "ocp.hpp"
#include "solvers/cgd.hpp"
#include "solvers/ilqr.hpp"
#include "solvers/osqp_solver.hpp"
#include "types.hpp"

OCP
create_single_track_lane_following_ocp()
{
  OCP problem;

  // Dimensions
  problem.state_dim     = 4;
  problem.control_dim   = 2;
  problem.horizon_steps = 30;  // Example: 5 steps for testing.
  problem.dt            = 0.1; // 0.1 s per step.

  // Initial state: for example, X=1, Y=1, psi=1, vx=1
  problem.initial_state = Eigen::VectorXd::Zero( problem.state_dim );
  problem.initial_state << 0.0, 5.0, 0.0, 0.0;

  // Dynamics: use the dynamic_bicycle_model defined in your code.
  problem.dynamics = single_track_model;

  // Desired velocity.
  const double desired_velocity = 5.0; // [m/s]

  // Cost weights.
  const double w_lane  = 1.0; // Penalize lateral deviation.
  const double w_speed = 1.0; // Penalize speed error.
  const double w_delta = 0.1; // Penalize steering.
  const double w_acc   = 0.1; // Penalize acceleration.

  // Stage cost function.
  problem.stage_cost = [=]( const State& state, const Control& control ) -> double {
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
  problem.terminal_cost = [=]( const State& state ) -> double { return 0.0; };
  // --- Add analytic derivatives for the cost function. ---
  // Gradient with respect to state.
  problem.cost_state_gradient = [=]( const StageCostFunction&, const State& state, const Control& ) -> Eigen::VectorXd {
    Eigen::VectorXd grad = Eigen::VectorXd::Zero( state.size() );
    // Only Y (index 1) and vx (index 3) appear in the cost.
    grad( 1 ) = 2.0 * w_lane * state( 1 );
    grad( 3 ) = 2.0 * w_speed * ( state( 3 ) - desired_velocity );
    return grad;
  };

  // Gradient with respect to control.
  problem.cost_control_gradient = [=]( const StageCostFunction&, const State&, const Control& control ) -> Eigen::VectorXd {
    Eigen::VectorXd grad = Eigen::VectorXd::Zero( control.size() );
    grad( 0 )            = 2.0 * w_delta * control( 0 );
    grad( 1 )            = 2.0 * w_acc * control( 1 );
    return grad;
  };

  // Hessian with respect to state.
  problem.cost_state_hessian = [=]( const StageCostFunction&, const State&, const Control& ) -> Eigen::MatrixXd {
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero( 6, 6 );
    H( 1, 1 )         = 2.0 * w_lane;
    H( 3, 3 )         = 2.0 * w_speed;
    return H;
  };

  // Hessian with respect to control.
  problem.cost_control_hessian = [=]( const StageCostFunction&, const State&, const Control& ) -> Eigen::MatrixXd {
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

#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>

void
single_track_test()
{
  // Build the lane-following OCP.
  OCP problem = create_single_track_lane_following_ocp();

  // Set solver parameters.
  int    max_iterations = 100;
  double tolerance      = 1e-7;

  // Define solvers in a map
  std::map<std::string, Solver> solvers = {
    { "iLQR", ilqr_solver },
    {  "CGD",  cgd_solver },
    { "OSQP", osqp_solver }
  };

  struct SolverResult
  {
    double cost;
    double time;
  };

  std::map<std::string, SolverResult> results;

  // Run solvers
  for( const auto& [name, solver] : solvers )
  {
    auto start = std::chrono::high_resolution_clock::now();
    solver( problem, max_iterations, tolerance );
    auto end = std::chrono::high_resolution_clock::now();

    results[name] = { problem.best_cost, std::chrono::duration<double, std::milli>( end - start ).count() };

    problem.reset();
  }

  // Find best and worst values
  auto [min_cost, max_cost] = std::minmax_element( results.begin(), results.end(),
                                                   []( const auto& a, const auto& b ) { return a.second.cost < b.second.cost; } );

  auto [min_time, max_time] = std::minmax_element( results.begin(), results.end(),
                                                   []( const auto& a, const auto& b ) { return a.second.time < b.second.time; } );

  // Print structured results
  std::cerr << "\n\n\n++++++++++++++++++++++++++++++++" << std::endl;
  std::cout << "🚗 Single-Track Lane Following Test 🚗\n";
  std::cout << "-----------------------------------\n";
  std::cout << std::left << std::setw( 25 ) << "Solver" << std::setw( 15 ) << "Cost" << std::setw( 15 ) << "Time (ms)\n";
  std::cout << "---------------------------------------------\n";

  for( const auto& [name, result] : results )
  {
    std::cout << std::left << std::setw( 25 ) << name << std::setw( 15 ) << result.cost << std::setw( 15 ) << result.time << "\n";
  }

  std::cout << "\n\n\n++++++++++++++++++++++++++++++++" << std::endl;
}
