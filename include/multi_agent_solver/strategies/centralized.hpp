#pragma once

#include "multi_agent_solver/multi_agent_problem.hpp"
#include "multi_agent_solver/solution.hpp"
#include "multi_agent_solver/solvers/solver.hpp"

namespace mas
{

struct CentralizedStrategy
{
  Solver solver;

  explicit CentralizedStrategy( Solver s ) :
    solver( std::move( s ) )
  {}

  Solution
  operator()( MultiAgentProblem& problem )
  {
    problem.compute_offsets();
    OCP global = problem.build_global_ocp();
    mas::solve( solver, global );

    Solution sol;
    sol.total_cost = global.best_cost;
    for( const auto& blk : problem.blocks )
    {
      auto& ocp         = *blk.agent->ocp;
      ocp.best_states   = global.best_states.block( blk.state_offset, 0, blk.state_dim, global.best_states.cols() );
      ocp.best_controls = global.best_controls.block( blk.control_offset, 0, blk.control_dim, global.best_controls.cols() );
      ocp.best_cost     = ocp.objective_function( ocp.best_states, ocp.best_controls );
      sol.states.push_back( ocp.best_states );
      sol.controls.push_back( ocp.best_controls );
      sol.costs.push_back( ocp.best_cost );
    }
    return sol;
  }
};

} // namespace mas
