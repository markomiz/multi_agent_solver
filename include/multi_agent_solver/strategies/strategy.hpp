#pragma once

#include <variant>

#include "multi_agent_solver/multi_agent_problem.hpp"
#include "multi_agent_solver/solution.hpp"
#include "multi_agent_solver/strategies/centralized.hpp"
#include "multi_agent_solver/strategies/nash.hpp"

namespace mas
{

template<typename Scalar = double>
using StrategyT
  = std::variant<CentralizedStrategy<Scalar>, SequentialNashStrategy<Scalar>, LineSearchNashStrategy<Scalar>,
                 TrustRegionNashStrategy<Scalar>>;

using Strategy  = StrategyT<double>;
using Strategyd = StrategyT<double>;
using Strategyf = StrategyT<float>;

template<typename Scalar>
inline SolutionT<Scalar>
solve( StrategyT<Scalar>& strategy, MultiAgentProblemT<Scalar>& problem )
{
  return std::visit( [&]( auto& s ) { return s( problem ); }, strategy );
}

inline Solution
solve( Strategy& strategy, MultiAgentProblem& problem )
{
  return solve<double>( strategy, problem );
}

} // namespace mas
