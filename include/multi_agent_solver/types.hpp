#pragma once

#include <functional>
#include <iomanip> // For std::setw
#include <iostream>
#include <map>
#include <unordered_map>

#include <Eigen/Dense>

namespace mas
{

// Types defined for clarity in other functions
template<typename Scalar>
using StateT = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
template<typename Scalar>
using StateDerivativeT = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
template<typename Scalar>
using ControlT = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
template<typename Scalar>
using ControlTrajectoryT = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
template<typename Scalar>
using StateTrajectoryT = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

using State             = StateT<double>;
using StateDerivative   = StateDerivativeT<double>;
using Control           = ControlT<double>;
using ControlTrajectory = ControlTrajectoryT<double>;
using StateTrajectory   = StateTrajectoryT<double>;

// Dynamics
template<typename Scalar>
using MotionModelT = std::function<StateDerivativeT<Scalar>( const StateT<Scalar>&, const ControlT<Scalar>& )>;
using MotionModel  = MotionModelT<double>;

// Cost Function
template<typename Scalar>
using ObjectiveFunctionT = std::function<Scalar( const StateTrajectoryT<Scalar>&, const ControlTrajectoryT<Scalar>& )>;
using ObjectiveFunction  = ObjectiveFunctionT<double>;

// A stage cost is evaluated on a single (state, control) pair.
template<typename Scalar>
using StageCostFunctionT = std::function<Scalar( const StateT<Scalar>&, const ControlT<Scalar>&, size_t time_idx )>;
using StageCostFunction  = StageCostFunctionT<double>;
// A terminal cost is evaluated on the final state.
template<typename Scalar>
using TerminalCostFunctionT = std::function<Scalar( const StateT<Scalar>& )>;
using TerminalCostFunction  = TerminalCostFunctionT<double>;


// Constraints
template<typename Scalar>
using ConstraintViolationsT = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
template<typename Scalar>
using ConstraintsFunctionT = std::function<ConstraintViolationsT<Scalar>( const StateT<Scalar>&, const ControlT<Scalar>& )>;
template<typename Scalar>
using ConstraintsJacobianT = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
template<typename Scalar>
using ConstraintsJacobianFunctionT = std::function<ConstraintsJacobianT<Scalar>( const StateT<Scalar>&, const ControlT<Scalar>& )>;

using ConstraintViolations        = ConstraintViolationsT<double>;
using ConstraintsFunction         = ConstraintsFunctionT<double>;
using ConstraintsJacobian         = ConstraintsJacobianT<double>;
using ConstraintsJacobianFunction = ConstraintsJacobianFunctionT<double>;


// Derivative interfaces
template<typename Scalar>
using DynamicsStateJacobianT = std::function<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>(
  const MotionModelT<Scalar>& dynamics, const StateT<Scalar>&, const ControlT<Scalar>& )>;
template<typename Scalar>
using DynamicsControlJacobianT = std::function<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>(
  const MotionModelT<Scalar>& dynamics, const StateT<Scalar>&, const ControlT<Scalar>& )>;
template<typename Scalar>
using CostStateGradientT = std::function<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>( const StageCostFunctionT<Scalar>&, const StateT<Scalar>&,
                                                                                   const ControlT<Scalar>&, size_t )>;
template<typename Scalar>
using CostControlGradientT = std::function<
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1>( const StageCostFunctionT<Scalar>&, const StateT<Scalar>&, const ControlT<Scalar>&, size_t )>;
template<typename Scalar>
using CostStateHessianT = std::function<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>(
  const StageCostFunctionT<Scalar>&, const StateT<Scalar>&, const ControlT<Scalar>&, size_t )>;
template<typename Scalar>
using CostControlHessianT = std::function<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>(
  const StageCostFunctionT<Scalar>&, const StateT<Scalar>&, const ControlT<Scalar>&, size_t )>;
template<typename Scalar>
using CostCrossTermT = std::function<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>(
  const StageCostFunctionT<Scalar>&, const StateT<Scalar>&, const ControlT<Scalar>&, size_t )>;
template<typename Scalar>
using TerminalCostGradientT
  = std::function<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>( const TerminalCostFunctionT<Scalar>&, const StateT<Scalar>& )>;
template<typename Scalar>
using TerminalCostHessianT
  = std::function<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>( const TerminalCostFunctionT<Scalar>&, const StateT<Scalar>& )>;
template<typename Scalar>
using ControlGradientT = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

using DynamicsStateJacobian   = DynamicsStateJacobianT<double>;
using DynamicsControlJacobian = DynamicsControlJacobianT<double>;
using CostStateGradient       = CostStateGradientT<double>;
using CostControlGradient     = CostControlGradientT<double>;
using CostStateHessian        = CostStateHessianT<double>;
using CostControlHessian      = CostControlHessianT<double>;
using CostCrossTerm           = CostCrossTermT<double>;
using TerminalCostGradient    = TerminalCostGradientT<double>;
using TerminalCostHessian     = TerminalCostHessianT<double>;
using ControlGradient         = ControlGradientT<double>;

// GradientComputer interface
template<typename Scalar>
using GradientComputerT = std::function<
  ControlGradientT<Scalar>( const StateT<Scalar>& initial_state, const ControlTrajectoryT<Scalar>& controls,
                            const MotionModelT<Scalar>& dynamics, const ObjectiveFunctionT<Scalar>& objective_function, Scalar timestep )>;
using GradientComputer = GradientComputerT<double>;

template<typename Scalar>
using SolverParamsT = std::unordered_map<std::string, Scalar>;
using SolverParams  = SolverParamsT<double>;
using SolverParamsf = SolverParamsT<float>;

// ANSI Escape Codes for Colors
namespace print_color
{
inline constexpr const char* reset  = "\033[0m";
inline constexpr const char* green  = "\033[1;32m";
inline constexpr const char* yellow = "\033[1;33m";
inline constexpr const char* red    = "\033[1;31m";
} // namespace print_color

} // namespace mas
