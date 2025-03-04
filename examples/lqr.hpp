#pragma once
#include <chrono>
#include <iostream>

#include <Eigen/Dense>

#include "finite_differences.hpp"
#include "integrator.hpp"
#include "line_search.hpp"
#include "ocp.hpp"
#include "single_track_ocp.hpp"
#include "solver_output.hpp"
#include "solvers/constrained_gradient_descent.hpp"
#include "solvers/gradient_descent.hpp"
#include "solvers/ilqr.hpp"
#include "solvers/osqp_solver.hpp"
#include "types.hpp"

void
lqr_test( bool with_derivatives )
{
  // Linear dynamics
  Eigen::MatrixXd A = Eigen::MatrixXd::Identity( 4, 4 ); // State transition matrix
  Eigen::MatrixXd B = Eigen::MatrixXd::Identity( 4, 4 ); // Control input matrix

  // Define a classical LQR OCP
  OCP problem;

  // Initial state
  problem.initial_state = Eigen::VectorXd::Random( 4 ); // Random initial state

  // Linear dynamics (capture A and B in the lambda)
  problem.dynamics = [A, B]( const State& x, const Control& u ) {
    return A * x + B * u; // Discrete linear dynamics
  };

  // Quadratic cost function
  Eigen::MatrixXd Q = Eigen::MatrixXd::Identity( 4, 4 ); // State cost matrix
  Eigen::MatrixXd R = Eigen::MatrixXd::Identity( 4, 4 ); // Control cost matrix

  problem.stage_cost = [Q, R]( const State& state, const Control& control ) {
    double cost = 0.0;

    cost += ( state.transpose() * Q * state ).value() + ( control.transpose() * R * control ).value();

    return cost;
  };

  problem.terminal_cost = []( const State& ) { return 0; };

  if( with_derivatives ) // without these finite differences is used
  {

    // Analytical derivatives for dynamics
    problem.dynamics_state_jacobian = [A]( const MotionModel&, const State&, const Control& ) {
      return A; // ∂f/∂x
    };

    problem.dynamics_control_jacobian = [B]( const MotionModel&, const State&, const Control& ) {
      return B; // ∂f/∂u
    };

    // Analytical derivatives for cost
    problem.cost_state_gradient = [Q]( const StageCostFunction&, const State& x, const Control& ) {
      return 2 * Q * x; // Gradient ∂l/∂x
    };

    problem.cost_control_gradient = [R]( const StageCostFunction&, const State&, const Control& u ) {
      return 2 * R * u; // Gradient ∂l/∂u
    };

    problem.cost_state_hessian = [Q]( const StageCostFunction&, const State&, const Control& ) {
      return 2 * Q; // Hessian ∂²l/∂x²
    };

    problem.cost_control_hessian = [R]( const StageCostFunction&, const State&, const Control& ) {
      return 2 * R; // Hessian ∂²l/∂u²
    };

    problem.cost_cross_term = []( const StageCostFunction&, const State&, const Control& ) {
      return Eigen::MatrixXd::Zero( 4, 4 ); // Cross term ∂²l/∂x∂u
    };
  }


  // Problem dimensions and time settings
  problem.dt            = 0.1; // Time step
  problem.horizon_steps = 50;  // Horizon length
  problem.control_dim   = 4;
  problem.state_dim     = 4;

  // Control bounds
  problem.input_lower_bounds = Eigen::VectorXd::Constant( 1, -1.0 );
  problem.input_upper_bounds = Eigen::VectorXd::Constant( 1, 1.0 );

  // Verify problem setup
  problem.initialize_problem();

  problem.verify_problem();

  // Run gradient descent solver with timing
  int    max_iterations = 100;
  double tolerance      = 1e-5; // Convergence tolerance

  auto                          start_gd    = std::chrono::high_resolution_clock::now();
  auto                          solution_gd = gradient_descent_solver( problem, max_iterations, tolerance );
  auto                          end_gd      = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_gd  = end_gd - start_gd;

  // Run iLQR solver with timing
  auto                          start_ilqr    = std::chrono::high_resolution_clock::now();
  auto                          solution_ilqr = ilqr_solver( problem, max_iterations, tolerance );
  auto                          end_ilqr      = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_ilqr  = end_ilqr - start_ilqr;

  auto                          start_cgd    = std::chrono::high_resolution_clock::now();
  auto                          solution_cgd = constrained_gradient_descent_solver( problem, max_iterations, tolerance );
  auto                          end_cgd      = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_cgd  = end_cgd - start_cgd;

  // Run iLQR solver with timing
  auto                          start_osqp    = std::chrono::high_resolution_clock::now();
  auto                          solution_osqp = osqp_solver( problem, max_iterations, tolerance );
  auto                          end_osqp      = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_osqp  = end_osqp - start_osqp;

  // Output results
  std::cout << "iLQR cost: " << solution_ilqr.cost << std::endl;
  std::cout << "Gradient Descent Cost: " << solution_gd.cost << std::endl;
  std::cout << "Constrained Gradient Descent Cost: " << solution_cgd.cost << std::endl;
  std::cout << "OSQP Cost " << solution_osqp.cost << std::endl;

  std::cout << "---------------" << std::endl;
  std::cout << "iLQR runtime: " << elapsed_ilqr.count() << " seconds" << std::endl;
  std::cout << "Gradient Descent runtime: " << elapsed_gd.count() << " seconds" << std::endl;
  std::cout << "CGD runtime: " << elapsed_cgd.count() << " seconds" << std::endl;
  std::cout << "OSQP runtime: " << elapsed_osqp.count() << " seconds" << std::endl;

  std::cout << "\n---------------------------------------" << std::endl;
}