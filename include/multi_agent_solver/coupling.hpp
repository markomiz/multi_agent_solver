#pragma once
#include <functional>
#include <memory>
#include <vector>

#include "multi_agent_solver/types.hpp"

namespace mas
{

struct Coupling
{
  std::vector<std::size_t>                                            agent_ids;
  std::function<double( const State&, const Control& )>               cost_fn;
  std::function<ConstraintViolations( const State&, const Control& )> eq_constraint_fn;
  std::function<ConstraintViolations( const State&, const Control& )> ineq_constraint_fn;
};

using CouplingPtr = std::shared_ptr<Coupling>;

struct InteractionGraph
{
  std::vector<CouplingPtr> couplings;

  void
  add_coupling( const CouplingPtr& c )
  {
    couplings.push_back( c );
  }

  std::vector<CouplingPtr>
  for_agent( std::size_t id ) const
  {
    std::vector<CouplingPtr> out;
    for( const auto& c : couplings )
    {
      if( std::find( c->agent_ids.begin(), c->agent_ids.end(), id ) != c->agent_ids.end() )
        out.push_back( c );
    }
    return out;
  }
};

} // namespace mas
