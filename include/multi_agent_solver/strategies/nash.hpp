#pragma once

#include <type_traits>
#include <vector>

#include "multi_agent_solver/integrator.hpp"
#include "multi_agent_solver/multi_agent_problem.hpp"
#include "multi_agent_solver/solution.hpp"
#include "multi_agent_solver/solvers/solver.hpp"

namespace mas
{

namespace detail
{

inline Solver
make_solver_like( const Solver& proto )
{
  return std::visit( []( const auto& s ) -> Solver { return std::decay_t<decltype( s )>{}; }, proto );
}

inline Solution
collect_solution( MultiAgentProblem& problem )
{
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

inline double
total_cost( MultiAgentProblem& problem )
{
  double    c = 0.0;
  const int n = static_cast<int>( problem.blocks.size() );

#pragma omp parallel for reduction( + : c ) schedule( static )
  for( int i = 0; i < n; ++i )
  {
    c += problem.blocks[static_cast<std::size_t>( i )].agent->ocp->best_cost;
  }
  return c;
}

inline void
sequential_solve( std::vector<Solver>& solvers, MultiAgentProblem& problem )
{
  // Parallel Jacobi step: solve all -> update all
  const int n = static_cast<int>( problem.blocks.size() );

#pragma omp parallel for schedule( static )
  for( int i = 0; i < n; ++i )
  {
    auto idx = static_cast<std::size_t>( i );
    mas::solve( solvers[idx], *problem.blocks[idx].agent->ocp );
  }

#pragma omp parallel for schedule( static )
  for( int i = 0; i < n; ++i )
  {
    auto idx = static_cast<std::size_t>( i );
    problem.blocks[idx].agent->update_initial_with_best();
  }
}

inline Solution
run_sequential( int max_outer, const Solver& solver_proto, const SolverParams& params, MultiAgentProblem& problem )
{
  problem.compute_offsets();
  std::vector<Solver> solvers;
  solvers.reserve( problem.blocks.size() );
  for( std::size_t i = 0; i < problem.blocks.size(); ++i )
  {
    solvers.emplace_back( make_solver_like( solver_proto ) );
    set_params( solvers.back(), params );
  }

  for( int outer = 0; outer < max_outer; ++outer )
    sequential_solve( solvers, problem );

  return collect_solution( problem );
}

inline Solution
run_line_search( int max_outer, const Solver& solver_proto, const SolverParams& params, MultiAgentProblem& problem )
{
  problem.compute_offsets();
  std::vector<Solver> solvers;
  solvers.reserve( problem.blocks.size() );
  for( std::size_t i = 0; i < problem.blocks.size(); ++i )
  {
    solvers.emplace_back( make_solver_like( solver_proto ) );
    set_params( solvers.back(), params );
  }

  double base_cost = total_cost( problem );
  for( int outer = 0; outer < max_outer; ++outer )
  {
    const int n = static_cast<int>( problem.blocks.size() );

    std::vector<ControlTrajectory> old_controls( n );
    std::vector<StateTrajectory>   old_states( n );
    for( std::size_t i = 0; i < n; ++i )
    {
      auto& ocp       = *problem.blocks[i].agent->ocp;
      old_controls[i] = ocp.best_controls;
      old_states[i]   = ocp.best_states;
    }

    sequential_solve( solvers, problem );
    double new_cost = total_cost( problem );

    if( new_cost >= base_cost )
    {
      std::vector<ControlTrajectory> cand_controls( n );
      for( std::size_t i = 0; i < n; ++i )
        cand_controls[i] = problem.blocks[i].agent->ocp->best_controls;

      double alpha    = 0.5;
      bool   accepted = false;
      while( alpha > 1e-3 && !accepted )
      {
        std::vector<ControlTrajectory> trial_controls( n );
        std::vector<StateTrajectory>   trial_states( n );
        double                         trial_cost = 0.0;
#pragma omp parallel for reduction( + : trial_cost ) schedule( static )
        for( int i = 0; i < n; ++i )
        {
          auto  idx            = static_cast<std::size_t>( i );
          auto& ocp            = *problem.blocks[idx].agent->ocp;
          trial_controls[idx]  = old_controls[idx] + alpha * ( cand_controls[idx] - old_controls[idx] );
          trial_states[idx]    = integrate_horizon( ocp.initial_state, trial_controls[idx], ocp.dt, ocp.dynamics, integrate_rk4 );
          trial_cost          += ocp.objective_function( trial_states[idx], trial_controls[idx] );
        }
        if( trial_cost < base_cost )
        {
          for( std::size_t i = 0; i < n; ++i )
          {
            auto& ocp         = *problem.blocks[i].agent->ocp;
            ocp.best_controls = trial_controls[i];
            ocp.best_states   = trial_states[i];
            ocp.best_cost     = ocp.objective_function( trial_states[i], trial_controls[i] );
            ocp.update_initial_with_best();
          }
          base_cost = trial_cost;
          accepted  = true;
        }
        else
        {
          alpha *= 0.5;
        }
      }
      if( !accepted )
      {
        for( std::size_t i = 0; i < n; ++i )
        {
          auto& ocp         = *problem.blocks[i].agent->ocp;
          ocp.best_controls = old_controls[i];
          ocp.best_states   = old_states[i];
          ocp.best_cost     = ocp.objective_function( old_states[i], old_controls[i] );
          ocp.update_initial_with_best();
        }
      }
    }
    else
    {
      base_cost = new_cost;
    }
  }

  return collect_solution( problem );
}

inline Solution
run_trust_region( int max_outer, const Solver& solver_proto, const SolverParams& params, MultiAgentProblem& problem )
{
  problem.compute_offsets();
  std::vector<Solver> solvers;
  solvers.reserve( problem.blocks.size() );
  for( std::size_t i = 0; i < problem.blocks.size(); ++i )
  {
    solvers.emplace_back( make_solver_like( solver_proto ) );
    set_params( solvers.back(), params );
  }

  std::vector<double> radii( problem.blocks.size(), 1.0 );

  for( int outer = 0; outer < max_outer; ++outer )
  {
    const int n = static_cast<int>( problem.blocks.size() );
#ifdef _OPENMP
  #pragma omp parallel for schedule( static )
#endif
    for( int i = 0; i < n; ++i )
    {
      auto  idx = static_cast<std::size_t>( i );
      auto& blk = problem.blocks[idx];
      auto& ocp = *blk.agent->ocp;

      ControlTrajectory old_u    = ocp.best_controls;
      StateTrajectory   old_x    = ocp.best_states;
      double            old_cost = ocp.best_cost;

      mas::solve( solvers[idx], ocp );

      ControlTrajectory cand_u    = ocp.best_controls;
      StateTrajectory   cand_x    = ocp.best_states;
      double            cand_cost = ocp.best_cost;

      ControlTrajectory delta = cand_u - old_u;
      double            norm  = delta.norm();
      if( norm > radii[idx] )
      {
        double scale = radii[idx] / norm;
        cand_u       = old_u + scale * delta;
        cand_x       = integrate_horizon( ocp.initial_state, cand_u, ocp.dt, ocp.dynamics, integrate_rk4 );
        cand_cost    = ocp.objective_function( cand_x, cand_u );
      }

      if( cand_cost < old_cost )
      {
        ocp.best_controls = cand_u;
        ocp.best_states   = cand_x;
        ocp.best_cost     = cand_cost;
        ocp.update_initial_with_best();
        radii[idx] *= 1.5;
      }
      else
      {
        ocp.best_controls = old_u;
        ocp.best_states   = old_x;
        ocp.best_cost     = old_cost;
        ocp.update_initial_with_best();
        radii[idx] *= 0.5;
      }
    }
  }

  return collect_solution( problem );
}

} // namespace detail

struct SequentialNashStrategy
{
  int          max_outer;
  Solver       solver_proto;
  SolverParams params;

  SequentialNashStrategy( int outer, Solver s, SolverParams p ) :
    max_outer( outer ),
    solver_proto( std::move( s ) ),
    params( std::move( p ) )
  {}

  Solution
  operator()( MultiAgentProblem& problem )
  {
    return detail::run_sequential( max_outer, solver_proto, params, problem );
  }
};

struct LineSearchNashStrategy
{
  int          max_outer;
  Solver       solver_proto;
  SolverParams params;

  LineSearchNashStrategy( int outer, Solver s, SolverParams p ) :
    max_outer( outer ),
    solver_proto( std::move( s ) ),
    params( std::move( p ) )
  {}

  Solution
  operator()( MultiAgentProblem& problem )
  {
    return detail::run_line_search( max_outer, solver_proto, params, problem );
  }
};

struct TrustRegionNashStrategy
{
  int          max_outer;
  Solver       solver_proto;
  SolverParams params;

  TrustRegionNashStrategy( int outer, Solver s, SolverParams p ) :
    max_outer( outer ),
    solver_proto( std::move( s ) ),
    params( std::move( p ) )
  {}

  Solution
  operator()( MultiAgentProblem& problem )
  {
    return detail::run_trust_region( max_outer, solver_proto, params, problem );
  }
};

} // namespace mas
