#pragma once
#include <memory>
#include "multi_agent_solver/ocp.hpp"

namespace mas {

struct Agent {
  std::size_t id;
  std::shared_ptr<OCP> ocp;

  Agent(std::size_t id_, std::shared_ptr<OCP> ocp_)
      : id(id_), ocp(std::move(ocp_)) {}

  int state_dim() const { return ocp->state_dim; }
  int control_dim() const { return ocp->control_dim; }

  void reset() { ocp->reset(); }
  void update_initial_with_best() { ocp->update_initial_with_best(); }
};
using AgentPtr = std::shared_ptr<Agent>;

} // namespace mas
