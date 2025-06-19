
#pragma once

#include <variant>

#include "multi_agent_solver/solvers/cgd.hpp"
#include "multi_agent_solver/solvers/ilqr.hpp"
#include "multi_agent_solver/solvers/osqp.hpp"
#include "multi_agent_solver/solvers/osqp_collocation.hpp"

namespace mas
{

// Holds any of the concrete solver objects.
using Solver = std::variant<iLQR, CGD, OSQP, OSQPCollocation>;

/**
 * @brief Convenience visitor to call solve() on the variant without
 *        repeating std::visit everywhere.
 */
inline void
solve( Solver& solver, OCP& problem )
{
  std::visit( [&]( auto& s ) { s.solve( problem ); }, solver );
}

inline void
set_params( Solver& solver, const SolverParams& params )
{
  std::visit( [&]( auto& s ) { s.set_params( params ); }, solver );
}

} // namespace mas
