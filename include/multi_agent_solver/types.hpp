#pragma once
#include <functional>
#include <iomanip> // For std::setw
#include <iostream>
#include <map>
#include <unordered_map>

#include <Eigen/Dense>

#ifdef MAS_USE_AUTODIFF
  #include <autodiff/forward/dual.hpp>
  #include <autodiff/forward/dual/eigen.hpp>
#endif

namespace mas
{

// Select scalar type: plain double by default or autodiff dual numbers.
#ifdef MAS_USE_AUTODIFF
using Scalar = autodiff::dual;
#else
using Scalar = double;
#endif

// Types defined for clarity in other functions
using State             = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
using StateDerivative   = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
using Control           = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
using ControlTrajectory = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
using StateTrajectory   = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

// Dynamics
using MotionModel = std::function<StateDerivative( const State&, const Control& )>;

// Cost Function
using ObjectiveFunction = std::function<Scalar( const StateTrajectory&, const ControlTrajectory& )>;

// A stage cost is evaluated on a single (state, control) pair.
using StageCostFunction = std::function<Scalar( const State&, const Control&, size_t time_idx )>;
// A terminal cost is evaluated on the final state.
using TerminalCostFunction = std::function<Scalar( const State& )>;


// Constraints
using ConstraintViolations        = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
using ConstraintsFunction         = std::function<ConstraintViolations( const State&, const Control& )>;
using ConstraintsJacobian         = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
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
using SolverParams = std::unordered_map<std::string, double>;

// Utility to extract a double from Scalar (works for both double and autodiff types)
inline double
to_double( const Scalar& x )
{
#ifdef MAS_USE_AUTODIFF
  return autodiff::val( x );
#else
  return x;
#endif
}

// ANSI Escape Codes for Colors
namespace print_color
{
inline constexpr const char* reset  = "\033[0m";
inline constexpr const char* green  = "\033[1;32m";
inline constexpr const char* yellow = "\033[1;33m";
inline constexpr const char* red    = "\033[1;31m";
} // namespace print_color

} // namespace mas
