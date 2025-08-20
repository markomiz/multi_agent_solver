#pragma once
#include <algorithm>
#include <memory>
#include <optional>
#include <vector>

#include "multi_agent_solver/agent.hpp"
#include "multi_agent_solver/coupling.hpp"
#include "multi_agent_solver/integrator.hpp"
#include "multi_agent_solver/ocp.hpp"

namespace mas {

struct AgentBlockInfo {
  std::size_t agent_id;
  int state_offset;
  int control_offset;
  int state_dim;
  int control_dim;
  AgentPtr agent;
};

class MultiAgentProblem {
public:
  std::vector<AgentPtr> agents;
  InteractionGraph graph;
  std::vector<AgentBlockInfo> blocks;

  void add_agent(const AgentPtr& a) { agents.push_back(a); }
  void add_coupling(const CouplingPtr& c) { graph.add_coupling(c); }

  void compute_offsets() {
    blocks.clear();
    std::vector<AgentPtr> sorted = agents;
    std::sort(sorted.begin(), sorted.end(), [](const AgentPtr& a, const AgentPtr& b){return a->id < b->id;});
    int s_off = 0, u_off = 0;
    for (auto& a : sorted) {
      blocks.push_back({a->id, s_off, u_off, a->state_dim(), a->control_dim(), a});
      s_off += a->state_dim();
      u_off += a->control_dim();
    }
  }

  OCP build_global_ocp() const {
    OCP g;
    // assume offsets computed
    int total_x = 0, total_u = 0;
    for (auto& b : blocks) { total_x += b.state_dim; total_u += b.control_dim; }
    g.state_dim = total_x;
    g.control_dim = total_u;
    if (!blocks.empty()) {
      g.horizon_steps = blocks.front().agent->ocp->horizon_steps;
      g.dt = blocks.front().agent->ocp->dt;
    }
    g.initial_state = Eigen::VectorXd(total_x);
    for (auto& b : blocks) {
      g.initial_state.segment(b.state_offset, b.state_dim) = b.agent->ocp->initial_state;
    }

    g.dynamics = [bs=blocks](const State& X, const Control& U){
      StateDerivative out = StateDerivative::Zero(X.size());
      for (auto& b : bs) {
        State sx = X.segment(b.state_offset,b.state_dim);
        Control su = U.segment(b.control_offset,b.control_dim);
        out.segment(b.state_offset,b.state_dim) = b.agent->ocp->dynamics(sx,su);
      }
      return out;
    };
    g.stage_cost = [bs=blocks](const State& full_x, const Control& full_u, size_t t){
      double cost=0.0;
      for (auto& b : bs){
        State x = full_x.segment(b.state_offset,b.state_dim);
        Control u = full_u.segment(b.control_offset,b.control_dim);
        cost += b.agent->ocp->stage_cost(x,u,t);
      }
      return cost;
    };
    g.terminal_cost = [bs=blocks](const State& full_x){
      double cost=0.0;
      for(auto& b: bs){
        State x = full_x.segment(b.state_offset,b.state_dim);
        cost += b.agent->ocp->terminal_cost(x);
      }
      return cost;
    };

    g.initialize_problem();
    g.verify_problem();
    return g;
  }

  struct LocalView {
    AgentPtr agent;
    std::vector<CouplingPtr> couplings;
  };

  LocalView get_local_view(std::size_t id) const {
    LocalView view;
    for (auto& a : agents) if (a->id==id) view.agent=a;
    view.couplings = graph.for_agent(id);
    return view;
  }
};

} // namespace mas
