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

template<typename Variant>
inline Variant
make_solver_like( const Variant& proto )
{
  return std::visit( []( const auto& s ) -> Variant { return std::decay_t<decltype( s )>{}; }, proto );
}

template<typename Scalar>
inline SolutionT<Scalar>
collect_solution( MultiAgentProblemT<Scalar>& problem )
{
  SolutionT<Scalar> sol;
  sol.total_cost = static_cast<Scalar>( 0 );
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

template<typename Scalar>
inline Scalar
total_cost( MultiAgentProblemT<Scalar>& problem )
{
  Scalar    c = static_cast<Scalar>( 0 );
  const int n = static_cast<int>( problem.blocks.size() );

#pragma omp parallel for reduction( + : c ) schedule( static )
  for( int i = 0; i < n; ++i )
  {
    c += problem.blocks[static_cast<std::size_t>( i )].agent->ocp->best_cost;
  }
  return c;
}

template<typename Scalar>
inline void
sequential_solve( std::vector<SolverVariant<Scalar>>& solvers, MultiAgentProblemT<Scalar>& problem )
{
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

template<typename Scalar>
inline SolutionT<Scalar>
run_sequential( int max_outer, const SolverVariant<Scalar>& solver_proto, const SolverParamsT<Scalar>& params,
                MultiAgentProblemT<Scalar>& problem )
{
  problem.compute_offsets();
  std::vector<SolverVariant<Scalar>> solvers;
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

template<typename Scalar>
inline SolutionT<Scalar>
run_line_search( int max_outer, const SolverVariant<Scalar>& solver_proto, const SolverParamsT<Scalar>& params,
                 MultiAgentProblemT<Scalar>& problem )
{
  using ControlTrajectory = ControlTrajectoryT<Scalar>;
  using StateTrajectory   = StateTrajectoryT<Scalar>;

  problem.compute_offsets();
  std::vector<SolverVariant<Scalar>> solvers;
  solvers.reserve( problem.blocks.size() );
  for( std::size_t i = 0; i < problem.blocks.size(); ++i )
  {
    solvers.emplace_back( make_solver_like( solver_proto ) );
    set_params( solvers.back(), params );
  }

  Scalar base_cost = total_cost( problem );
  for( int outer = 0; outer < max_outer; ++outer )
  {
    const int n = static_cast<int>( problem.blocks.size() );

    std::vector<ControlTrajectory> old_controls( static_cast<std::size_t>( n ) );
    std::vector<StateTrajectory>   old_states( static_cast<std::size_t>( n ) );
    for( std::size_t i = 0; i < static_cast<std::size_t>( n ); ++i )
    {
      auto& ocp       = *problem.blocks[i].agent->ocp;
      old_controls[i] = ocp.best_controls;
      old_states[i]   = ocp.best_states;
    }

    sequential_solve( solvers, problem );
    Scalar new_cost = total_cost( problem );

    if( new_cost >= base_cost )
    {
      std::vector<ControlTrajectory> cand_controls( static_cast<std::size_t>( n ) );
      for( std::size_t i = 0; i < static_cast<std::size_t>( n ); ++i )
        cand_controls[i] = problem.blocks[i].agent->ocp->best_controls;

      Scalar alpha    = static_cast<Scalar>( 0.5 );
      bool   accepted = false;
      while( alpha > static_cast<Scalar>( 1e-3 ) && !accepted )
      {
        std::vector<ControlTrajectory> trial_controls( static_cast<std::size_t>( n ) );
        std::vector<StateTrajectory>   trial_states( static_cast<std::size_t>( n ) );
        Scalar                         trial_cost = static_cast<Scalar>( 0 );
#pragma omp parallel for reduction( + : trial_cost ) schedule( static )
        for( int i = 0; i < n; ++i )
        {
          auto  idx            = static_cast<std::size_t>( i );
          auto& ocp            = *problem.blocks[idx].agent->ocp;
          trial_controls[idx]  = old_controls[idx] + alpha * ( cand_controls[idx] - old_controls[idx] );
          trial_states[idx]    = integrate_horizon<Scalar>( ocp.initial_state, trial_controls[idx], ocp.dt, ocp.dynamics,
                                                            integrate_rk4<Scalar> );
          trial_cost          += ocp.objective_function( trial_states[idx], trial_controls[idx] );
        }
        if( trial_cost < base_cost )
        {
          for( std::size_t i = 0; i < static_cast<std::size_t>( n ); ++i )
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
          alpha *= static_cast<Scalar>( 0.5 );
        }
      }
      if( !accepted )
      {
        for( std::size_t i = 0; i < static_cast<std::size_t>( n ); ++i )
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

template<typename Scalar>
inline SolutionT<Scalar>
run_trust_region( int max_outer, const SolverVariant<Scalar>& solver_proto, const SolverParamsT<Scalar>& params,
                  MultiAgentProblemT<Scalar>& problem )
{
  using ControlTrajectory = ControlTrajectoryT<Scalar>;
  using StateTrajectory   = StateTrajectoryT<Scalar>;

  problem.compute_offsets();
  std::vector<SolverVariant<Scalar>> solvers;
  solvers.reserve( problem.blocks.size() );
  for( std::size_t i = 0; i < problem.blocks.size(); ++i )
  {
    solvers.emplace_back( make_solver_like( solver_proto ) );
    set_params( solvers.back(), params );
  }

  std::vector<Scalar> radii( problem.blocks.size(), static_cast<Scalar>( 1 ) );

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
      Scalar            old_cost = ocp.best_cost;

      mas::solve( solvers[idx], ocp );

      ControlTrajectory cand_u    = ocp.best_controls;
      StateTrajectory   cand_x    = ocp.best_states;
      Scalar            cand_cost = ocp.best_cost;

      ControlTrajectory delta = cand_u - old_u;
      Scalar            norm  = delta.norm();
      if( norm > radii[idx] )
      {
        Scalar scale = radii[idx] / norm;
        cand_u       = old_u + scale * delta;
        cand_x       = integrate_horizon<Scalar>( ocp.initial_state, cand_u, ocp.dt, ocp.dynamics, integrate_rk4<Scalar> );
        cand_cost    = ocp.objective_function( cand_x, cand_u );
      }

      if( cand_cost < old_cost )
      {
        ocp.best_controls = cand_u;
        ocp.best_states   = cand_x;
        ocp.best_cost     = cand_cost;
        ocp.update_initial_with_best();
        radii[idx] *= static_cast<Scalar>( 1.5 );
      }
      else
      {
        ocp.best_controls = old_u;
        ocp.best_states   = old_x;
        ocp.best_cost     = old_cost;
        ocp.update_initial_with_best();
        radii[idx] *= static_cast<Scalar>( 0.5 );
      }
    }
  }

  return collect_solution( problem );
}

} // namespace detail

template<typename Scalar = double>
struct SequentialNashStrategy
{
  using SolverType  = SolverVariant<Scalar>;
  using ParamsType  = SolverParamsT<Scalar>;
  using ProblemType = MultiAgentProblemT<Scalar>;
  using Solution    = SolutionT<Scalar>;

  int         max_outer;
  SolverType  solver_proto;
  ParamsType  params;

  SequentialNashStrategy( int outer, SolverType s, ParamsType p ) :
    max_outer( outer ),
    solver_proto( std::move( s ) ),
    params( std::move( p ) )
  {}

  Solution
  operator()( ProblemType& problem )
  {
    return detail::run_sequential<Scalar>( max_outer, solver_proto, params, problem );
  }
};

template<typename Scalar = double>
struct LineSearchNashStrategy
{
  using SolverType  = SolverVariant<Scalar>;
  using ParamsType  = SolverParamsT<Scalar>;
  using ProblemType = MultiAgentProblemT<Scalar>;
  using Solution    = SolutionT<Scalar>;

  int         max_outer;
  SolverType  solver_proto;
  ParamsType  params;

  LineSearchNashStrategy( int outer, SolverType s, ParamsType p ) :
    max_outer( outer ),
    solver_proto( std::move( s ) ),
    params( std::move( p ) )
  {}

  Solution
  operator()( ProblemType& problem )
  {
    return detail::run_line_search<Scalar>( max_outer, solver_proto, params, problem );
  }
};

template<typename Scalar = double>
struct TrustRegionNashStrategy
{
  using SolverType  = SolverVariant<Scalar>;
  using ParamsType  = SolverParamsT<Scalar>;
  using ProblemType = MultiAgentProblemT<Scalar>;
  using Solution    = SolutionT<Scalar>;

  int         max_outer;
  SolverType  solver_proto;
  ParamsType  params;

  TrustRegionNashStrategy( int outer, SolverType s, ParamsType p ) :
    max_outer( outer ),
    solver_proto( std::move( s ) ),
    params( std::move( p ) )
  {}

  Solution
  operator()( ProblemType& problem )
  {
    return detail::run_trust_region<Scalar>( max_outer, solver_proto, params, problem );
  }
};

using SequentialNashStrategyd    = SequentialNashStrategy<double>;
using SequentialNashStrategyf    = SequentialNashStrategy<float>;
using LineSearchNashStrategyd    = LineSearchNashStrategy<double>;
using LineSearchNashStrategyf    = LineSearchNashStrategy<float>;
using TrustRegionNashStrategyd   = TrustRegionNashStrategy<double>;
using TrustRegionNashStrategyf   = TrustRegionNashStrategy<float>;

} // namespace mas
