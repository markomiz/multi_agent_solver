#pragma once

#include <vector>

#include "multi_agent_solver/multi_agent_problem.hpp"
#include "multi_agent_solver/solution.hpp"
#include "multi_agent_solver/solvers/solver.hpp"

namespace mas {

struct BestResponseStrategy {
  int max_outer;
  Solver solver_proto;

  BestResponseStrategy(int outer, Solver s)
      : max_outer(outer), solver_proto(std::move(s)) {}

  Solution operator()(MultiAgentProblem& problem) {
    problem.compute_offsets();
    std::vector<Solver> solvers(problem.blocks.size(), solver_proto);

    for (int outer = 0; outer < max_outer; ++outer) {
#pragma omp parallel for
      for (std::size_t i = 0; i < problem.blocks.size(); ++i) {
        mas::solve(solvers[i], *problem.blocks[i].agent->ocp);
        problem.blocks[i].agent->update_initial_with_best();
      }
    }

    Solution sol;
    sol.total_cost = 0.0;
    for (auto& blk : problem.blocks) {
      auto& ocp = *blk.agent->ocp;
      sol.states.push_back(ocp.best_states);
      sol.controls.push_back(ocp.best_controls);
      sol.costs.push_back(ocp.best_cost);
      sol.total_cost += ocp.best_cost;
    }
    return sol;
  }
};

} // namespace mas

