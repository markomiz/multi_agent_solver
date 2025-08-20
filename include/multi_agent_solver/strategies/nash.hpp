#pragma once

#include <vector>
#include <type_traits>

#include "multi_agent_solver/multi_agent_problem.hpp"
#include "multi_agent_solver/solution.hpp"
#include "multi_agent_solver/solvers/solver.hpp"

namespace mas {

namespace detail {

inline Solver
make_solver_like( const Solver& proto )
{
  return std::visit( []( const auto& s ) -> Solver { return std::decay_t<decltype( s )>{}; }, proto );
}

inline Solution
run_best_response( int max_outer, const Solver& solver_proto, MultiAgentProblem& problem )
{
  problem.compute_offsets();

  std::vector<Solver> solvers;
  solvers.reserve( problem.blocks.size() );
  for( std::size_t i = 0; i < problem.blocks.size(); ++i )
    solvers.emplace_back( make_solver_like( solver_proto ) );

  for( int outer = 0; outer < max_outer; ++outer )
  {
#pragma omp parallel for
    for( std::size_t i = 0; i < problem.blocks.size(); ++i )
    {
      mas::solve( solvers[i], *problem.blocks[i].agent->ocp );
      problem.blocks[i].agent->update_initial_with_best();
    }
  }

  Solution sol;
  sol.total_cost = 0.0;
  for( auto& blk : problem.blocks )
  {
    auto& ocp = *blk.agent->ocp;
    sol.states.push_back( ocp.best_states );
    sol.controls.push_back( ocp.best_controls );
    sol.costs.push_back( ocp.best_cost );
    sol.total_cost += ocp.best_cost;
  }
  return sol;
}

} // namespace detail

struct SequentialNashStrategy
{
  int    max_outer;
  Solver solver_proto;

  SequentialNashStrategy( int outer, Solver s )
    : max_outer( outer )
    , solver_proto( std::move( s ) )
  {}

  Solution
  operator()( MultiAgentProblem& problem )
  {
    return detail::run_best_response( max_outer, solver_proto, problem );
  }
};

struct LineSearchNashStrategy
{
  int    max_outer;
  Solver solver_proto;

  LineSearchNashStrategy( int outer, Solver s )
    : max_outer( outer )
    , solver_proto( std::move( s ) )
  {}

  Solution
  operator()( MultiAgentProblem& problem )
  {
    return detail::run_best_response( max_outer, solver_proto, problem );
  }
};

struct TrustRegionNashStrategy
{
  int    max_outer;
  Solver solver_proto;

  TrustRegionNashStrategy( int outer, Solver s )
    : max_outer( outer )
    , solver_proto( std::move( s ) )
  {}

  Solution
  operator()( MultiAgentProblem& problem )
  {
    return detail::run_best_response( max_outer, solver_proto, problem );
  }
};

} // namespace mas

