#pragma once
#include <chrono>
#include <iostream>
#include <map>
#include <string>

#include "models/dynmic_bicycle_model.hpp"
#include "models/single_track_model.hpp"
#include "ocp.hpp"
#include "solvers/constrained_gradient_descent.hpp"
#include "solvers/gradient_descent.hpp"
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
  problem.horizon_steps = 50;  // Example: 5 steps for testing.
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
  const double w_speed = 0.1; // Penalize speed error.
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

  // // --- Add analytic derivatives for the cost function. ---
  // // Gradient with respect to state.
  // problem.cost_state_gradient = [=]( const StageCostFunction&, const State& state, const Control& ) -> Eigen::VectorXd {
  //   Eigen::VectorXd grad = Eigen::VectorXd::Zero( state.size() );
  //   // Only Y (index 1) and vx (index 3) appear in the cost.
  //   grad( 1 ) = 2.0 * w_lane * state( 1 );
  //   grad( 3 ) = 2.0 * w_speed * ( state( 3 ) - desired_velocity );
  //   return grad;
  // };

  // // Gradient with respect to control.
  // problem.cost_control_gradient = [=]( const StageCostFunction&, const State&, const Control& control ) -> Eigen::VectorXd {
  //   Eigen::VectorXd grad = Eigen::VectorXd::Zero( control.size() );
  //   grad( 0 )            = 2.0 * w_delta * control( 0 );
  //   grad( 1 )            = 2.0 * w_acc * control( 1 );
  //   return grad;
  // };

  // // Hessian with respect to state.
  // problem.cost_state_hessian = [=]( const StageCostFunction&, const State&, const Control& ) -> Eigen::MatrixXd {
  //   Eigen::MatrixXd H = Eigen::MatrixXd::Zero( 6, 6 );
  //   H( 1, 1 )         = 2.0 * w_lane;
  //   H( 3, 3 )         = 2.0 * w_speed;
  //   return H;
  // };

  // // Hessian with respect to control.
  // problem.cost_control_hessian = [=]( const StageCostFunction&, const State&, const Control& ) -> Eigen::MatrixXd {
  //   Eigen::MatrixXd H = Eigen::MatrixXd::Zero( 2, 2 );
  //   H( 0, 0 )         = 2.0 * w_delta;
  //   H( 1, 1 )         = 2.0 * w_acc;
  //   return H;
  // };

  // // Cross-term Hessian (∂²l/∂u∂x). Since cost is separable, this is zero.
  // problem.cost_cross_term = [=]( const StageCostFunction&, const State&, const Control& ) -> Eigen::MatrixXd {
  //   return Eigen::MatrixXd::Zero( 2, 6 );
  // };

  // // (Optionally, you can also add analytic derivatives for the dynamics, but if your
  // // dynamic_bicycle_model is complex, you might continue to use finite differences.)
  // problem.dynamics_state_jacobian = []( const MotionModel& /*dyn*/, const State& x, const Control& u ) -> Eigen::MatrixXd {
  //   return dynamic_bicycle_state_jacobian( x, u );
  // };
  // problem.dynamics_control_jacobian = []( const MotionModel& /*dyn*/, const State& x, const Control& u ) -> Eigen::MatrixXd {
  //   return dynamic_bicycle_control_jacobian( x, u );
  // };


  // Eigen::VectorXd lower_bounds( 2 ), upper_bounds( 2 );
  // lower_bounds << -100.0, -100.0;
  // upper_bounds << 100.0, 100.0;
  // problem.input_lower_bounds = lower_bounds;
  // problem.input_upper_bounds = upper_bounds;

  // Initialize and verify the problem.
  problem.initialize_problem();
  problem.verify_problem();

  return problem;
}

void
single_track_test()
{
  // Build the lane-following OCP.
  OCP problem = create_single_track_lane_following_ocp();

  // Set solver parameters.
  int    max_iterations = 100;
  double tolerance      = 1e-7;

  // Solve with iLQR
  auto start_ilqr = std::chrono::high_resolution_clock::now();
  ilqr_solver( problem, max_iterations, tolerance );
  auto   end_ilqr     = std::chrono::high_resolution_clock::now();
  double elapsed_ilqr = std::chrono::duration<double, std::milli>( end_ilqr - start_ilqr ).count();
  double ilqr_cost    = problem.best_cost;
  problem.reset();

  // Solve with Gradient Descent (GD)
  auto start_gd = std::chrono::high_resolution_clock::now();
  gradient_descent_solver( problem, max_iterations, tolerance );
  auto   end_gd     = std::chrono::high_resolution_clock::now();
  double elapsed_gd = std::chrono::duration<double, std::milli>( end_gd - start_gd ).count();
  double gd_cost    = problem.best_cost;
  problem.reset();

  // Solve with Constrained Gradient Descent (CGD)
  auto start_cgd = std::chrono::high_resolution_clock::now();
  constrained_gradient_descent_solver( problem, max_iterations, tolerance );
  auto   end_cgd     = std::chrono::high_resolution_clock::now();
  double elapsed_cgd = std::chrono::duration<double, std::milli>( end_cgd - start_cgd ).count();
  double cgd_cost    = problem.best_cost;
  problem.reset();

  // Solve with OSQP
  auto start_osqp = std::chrono::high_resolution_clock::now();
  osqp_solver( problem, max_iterations, tolerance );
  auto   end_osqp     = std::chrono::high_resolution_clock::now();
  double elapsed_osqp = std::chrono::duration<double, std::milli>( end_osqp - start_osqp ).count();
  double osqp_cost    = problem.best_cost;
  problem.reset();

  // Print structured results
  std::cerr << "\n\n\n++++++++++++++++++++++++++++++++" << std::endl;
  std::cout << "🚗 Single-Track Lane Following Test 🚗\n";
  std::cout << "-----------------------------------\n";

  std::cout << "🔹 Single-track iLQR cost: " << ilqr_cost << "  | Time: " << elapsed_ilqr << " ms\n";
  std::cout << "🔹 Single-track Gradient Descent cost: " << gd_cost << "  | Time: " << elapsed_gd << " ms\n";
  std::cout << "🔹 Single-track Constrained GD cost: " << cgd_cost << "  | Time: " << elapsed_cgd << " ms\n";
  std::cout << "🔹 Single-track OSQP cost: " << osqp_cost << "  | Time: " << elapsed_osqp << " ms\n";

  std::cout << "\n\n\n++++++++++++++++++++++++++++++++" << std::endl;
}
