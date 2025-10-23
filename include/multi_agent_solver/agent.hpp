#pragma once
#include <memory>

#include "multi_agent_solver/ocp.hpp"

namespace mas
{

template<typename Scalar = double>
struct AgentBase
{
  using ScalarType = Scalar;
  using OCPType    = OCP<Scalar>;
  using OCPPtr     = std::shared_ptr<OCPType>;

  std::size_t id;
  OCPPtr      ocp;

  AgentBase( std::size_t id_, OCPPtr ocp_ ) :
    id( id_ ),
    ocp( std::move( ocp_ ) )
  {}

  int
  state_dim() const
  {
    return ocp->state_dim;
  }

  int
  control_dim() const
  {
    return ocp->control_dim;
  }

  void
  reset()
  {
    ocp->reset();
  }

  void
  update_initial_with_best()
  {
    ocp->update_initial_with_best();
  }
};

template<typename Scalar>
using AgentPtrT = std::shared_ptr<AgentBase<Scalar>>;

template<typename Scalar = double>
using AgentT = AgentBase<Scalar>;

using Agentd = AgentBase<double>;
using Agentf = AgentBase<float>;
using Agent  = Agentd;

using AgentPtr  = AgentPtrT<double>;
using AgentPtrd = AgentPtrT<double>;
using AgentPtrf = AgentPtrT<float>;

} // namespace mas
