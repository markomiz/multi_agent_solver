
#pragma once
#include <functional>
#include <optional>

#include <Eigen/Dense>

#include "types.hpp"

struct OCP
{

  // Dynamics and Objective
  State             initial_state;
  MotionModel       dynamics;
  ObjectiveFunction objective_function;

  // Static bounds
  std::optional<Eigen::VectorXd> state_lower_bounds;
  std::optional<Eigen::VectorXd> state_upper_bounds;
  std::optional<Eigen::VectorXd> input_lower_bounds;
  std::optional<Eigen::VectorXd> input_upper_bounds;

  // General inequality constraints: g(x,u) <= 0
  std::optional<ConstraintsFunction> inequality_constraints;

  // equality constraints: h(x,u) = 0
  std::optional<ConstraintsFunction> equality_constraints;
};
