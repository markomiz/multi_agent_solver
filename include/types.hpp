#pragma once
#include <functional>

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

// Constraints
using ConstraintViolations        = Eigen::VectorXd;
using ConstraintsFunction         = std::function<ConstraintViolations( const State&, const Control& )>;
using ConstraintsJacobian         = Eigen::MatrixXd;
using ConstraintsJacobianFunction = std::function<ConstraintsJacobian( const State&, const Control& )>;


// Derivative interfaces
using DynamicsStateJacobian   = std::function<Eigen::MatrixXd( const MotionModel& dynamics, const State&, const Control& )>;
using DynamicsControlJacobian = std::function<Eigen::MatrixXd( const MotionModel& dynamics, const State&, const Control& )>;
using CostStateGradient       = std::function<Eigen::VectorXd( const ObjectiveFunction& objective, const State&, const Control& )>;
using CostControlGradient     = std::function<Eigen::VectorXd( const ObjectiveFunction& objective, const State&, const Control& )>;
using CostStateHessian        = std::function<Eigen::MatrixXd( const ObjectiveFunction& objective, const State&, const Control& )>;
using CostControlHessian      = std::function<Eigen::MatrixXd( const ObjectiveFunction& objective, const State&, const Control& )>;
using CostCrossTerm           = std::function<Eigen::MatrixXd( const ObjectiveFunction& objective, const State&, const Control& )>;
using ControlGradient         = Eigen::MatrixXd;

// GradientComputer interface
using GradientComputer
  = std::function<ControlGradient( const State& initial_state, const ControlTrajectory& controls, const MotionModel& dynamics,
                                   const ObjectiveFunction& objective_function, double timestep )>;