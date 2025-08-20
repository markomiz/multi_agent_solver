#pragma once
#include <vector>
#include "multi_agent_solver/ocp.hpp"

namespace mas {

struct Solution {
  std::vector<StateTrajectory> states;
  std::vector<ControlTrajectory> controls;
  std::vector<double> costs;
  double total_cost = 0.0;
};

} // namespace mas
