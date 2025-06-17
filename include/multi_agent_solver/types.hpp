#pragma once
#include <functional>
#include <iomanip> // For std::setw
#include <iostream>
#include <map>
#include <unordered_map>

#include <Eigen/Dense>
// Types defined for clarity in other functions
using State             = Eigen::VectorXd;
using StateDerivative   = Eigen::VectorXd;
using Control           = Eigen::VectorXd;
using ControlTrajectory = Eigen::MatrixXd;
using StateTrajectory   = Eigen::MatrixXd;

// Dynamics
using MotionModel = std::function<StateDerivative( const State&, const Control& )>;

// Cost Function
using ObjectiveFunction = std::function<double( const StateTrajectory&, const ControlTrajectory& )>;

// A stage cost is evaluated on a single (state, control) pair.
using StageCostFunction = std::function<double( const State&, const Control&, size_t time_idx )>;
// A terminal cost is evaluated on the final state.
using TerminalCostFunction = std::function<double( const State& )>;


// Constraints
using ConstraintViolations        = Eigen::VectorXd;
using ConstraintsFunction         = std::function<ConstraintViolations( const State&, const Control& )>;
using ConstraintsJacobian         = Eigen::MatrixXd;
using ConstraintsJacobianFunction = std::function<ConstraintsJacobian( const State&, const Control& )>;


// Derivative interfaces
using DynamicsStateJacobian   = std::function<Eigen::MatrixXd( const MotionModel& dynamics, const State&, const Control& )>;
using DynamicsControlJacobian = std::function<Eigen::MatrixXd( const MotionModel& dynamics, const State&, const Control& )>;
using CostStateGradient       = std::function<Eigen::VectorXd( const StageCostFunction&, const State&, const Control&, size_t )>;
using CostControlGradient     = std::function<Eigen::VectorXd( const StageCostFunction&, const State&, const Control&, size_t )>;
using CostStateHessian        = std::function<Eigen::MatrixXd( const StageCostFunction&, const State&, const Control&, size_t )>;
using CostControlHessian      = std::function<Eigen::MatrixXd( const StageCostFunction&, const State&, const Control&, size_t )>;
using CostCrossTerm           = std::function<Eigen::MatrixXd( const StageCostFunction&, const State&, const Control&, size_t )>;
using ControlGradient         = Eigen::MatrixXd;

// GradientComputer interface
using GradientComputer
  = std::function<ControlGradient( const State& initial_state, const ControlTrajectory& controls, const MotionModel& dynamics,
                                   const ObjectiveFunction& objective_function, double timestep )>;


// ANSI Escape Codes for Colors
#define RESET  "\033[0m"
#define GREEN  "\033[1;32m"
#define YELLOW "\033[1;33m"
#define RED    "\033[1;31m"