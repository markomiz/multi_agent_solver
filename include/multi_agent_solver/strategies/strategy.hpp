#pragma once

#include <variant>

#include "multi_agent_solver/multi_agent_problem.hpp"
#include "multi_agent_solver/solution.hpp"
#include "multi_agent_solver/strategies/nash.hpp"
#include "multi_agent_solver/strategies/centralized.hpp"

namespace mas {

using Strategy = std::variant<CentralizedStrategy, SequentialNashStrategy, LineSearchNashStrategy, TrustRegionNashStrategy>;

inline Solution solve(Strategy& strategy, MultiAgentProblem& problem) {
  return std::visit([&](auto& s) { return s(problem); }, strategy);
}

} // namespace mas

