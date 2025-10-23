
#pragma once

#include <memory>
#include <type_traits>
#include <variant>

#include "multi_agent_solver/solvers/cgd.hpp"
#include "multi_agent_solver/solvers/ilqr.hpp"
#ifdef MAS_HAVE_OSQP
  #include "multi_agent_solver/solvers/osqp.hpp"
  #include "multi_agent_solver/solvers/osqp_collocation.hpp"
#endif

namespace mas
{

namespace detail
{

template<typename Scalar, bool EnableOSQP>
struct SolverVariantSelector;

template<typename Scalar>
struct SolverVariantSelector<Scalar, false>
{
  using Type = std::variant<iLQR<Scalar>, CGD<Scalar>>;
};

#ifdef MAS_HAVE_OSQP
template<typename Scalar>
struct SolverVariantSelector<Scalar, true>
{
  using Type = std::variant<iLQR<Scalar>, CGD<Scalar>, OSQP<Scalar>, OSQPCollocation<Scalar>>;
};
#endif

template<typename Scalar>
struct SolverVariantFor
{
#ifdef MAS_HAVE_OSQP
  static constexpr bool kEnableOSQP = std::is_same_v<Scalar, double>;
#else
  static constexpr bool kEnableOSQP = false;
#endif
  using Type = typename SolverVariantSelector<Scalar, kEnableOSQP>::Type;
};

} // namespace detail

// Holds any of the concrete solver objects.
template<typename Scalar>
using SolverVariant = typename detail::SolverVariantFor<Scalar>::Type;

using Solverd = SolverVariant<double>;
using Solverf = SolverVariant<float>;
using Solver  = Solverd;

/**
 * @brief Convenience visitor to call solve() on the variant without
 *        repeating std::visit everywhere.
 */
template<typename Scalar>
inline void
solve( SolverVariant<Scalar>& solver, OCP<Scalar>& problem )
{
  std::visit( [&]( auto& s ) { s.solve( problem ); }, solver );
}

inline void
solve( Solver& solver, OCP<>& problem )
{
  solve<double>( solver, problem );
}

template<typename Scalar>
inline void
set_params( SolverVariant<Scalar>& solver, const SolverParamsT<Scalar>& params )
{
  std::visit( [&]( auto& s ) { s.set_params( params ); }, solver );
}

inline void
set_params( Solver& solver, const SolverParams& params )
{
  set_params<double>( solver, params );
}

template<typename SolverT>
std::shared_ptr<Solver>
create()
{
  return std::make_shared<Solver>( std::in_place_type<SolverT> );
}

} // namespace mas
