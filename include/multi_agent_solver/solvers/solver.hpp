
#pragma once

#include <variant>

#include "multi_agent_solver/solvers/cgd.hpp"
#include "multi_agent_solver/solvers/ilqr.hpp"
#ifdef MAS_HAVE_OSQP
#  include "multi_agent_solver/solvers/osqp.hpp"
#  include "multi_agent_solver/solvers/osqp_collocation.hpp"
#endif

namespace mas
{

// Holds any of the concrete solver objects.
using Solver = std::variant<iLQR, CGD
#ifdef MAS_HAVE_OSQP
                             , OSQP, OSQPCollocation
#endif
                             >;

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

template<typename SolverT>
std::shared_ptr<Solver>
create()
{
  return std::make_shared<Solver>( std::in_place_type<SolverT> );
}

} // namespace mas
