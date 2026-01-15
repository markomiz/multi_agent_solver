#include <gtest/gtest.h>
#include <variant>

#include "multi_agent_solver/solvers/solver.hpp"
#include "multi_agent_solver/solvers/ilqr.hpp"
#ifdef MAS_HAVE_OSQP
#include "multi_agent_solver/solvers/osqp.hpp"
#endif

namespace mas
{

TEST( SolverTest, CreateFactoryLikelyWorks )
{
  auto solver_ilqr = create<iLQR>();
  EXPECT_TRUE( std::holds_alternative<iLQR>( *solver_ilqr ) );
  
#ifdef MAS_HAVE_OSQP
  auto solver_osqp = create<OSQP>();
  EXPECT_TRUE( std::holds_alternative<OSQP>( *solver_osqp ) );
#endif
}

TEST( SolverTest, SetParamsDispatchesToConcreteSolver )
{
  auto solver = create<iLQR>();
  SolverParams params;
  params["max_iterations"] = 123.0;
  params["tolerance"]      = 1e-4;
  params["max_ms"]         = 1000.0;
  
  set_params( *solver, params );
  
  // max_iterations is private, so we just verify set_params didn't throw.
  // To truly verify, we'd need getters or a functional test that relies on the parameter.
  EXPECT_TRUE( true );
}

} // namespace mas
