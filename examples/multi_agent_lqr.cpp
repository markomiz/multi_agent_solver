#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <tuple>
#include <vector>

#include "Eigen/Dense"

#include "multi_agent_solver/models/single_track_model.hpp"
#include "multi_agent_solver/multi_agent_aggregator.hpp"
#include "multi_agent_solver/ocp.hpp"
#include "multi_agent_solver/solvers/solver.hpp"
#include "multi_agent_solver/types.hpp"

/*──────────────── create simple LQR OCP (unchanged) ───────────────*/
mas::OCP
create_linear_lqr_ocp( int n_x, int n_u, double dt, int T )
{
  using namespace mas;
  OCP ocp;
  ocp.state_dim     = n_x;
  ocp.control_dim   = n_u;
  ocp.dt            = dt;
  ocp.horizon_steps = T;
  ocp.initial_state = Eigen::VectorXd::Random( n_x );

  Eigen::MatrixXd A = Eigen::MatrixXd::Identity( n_x, n_x );
  Eigen::MatrixXd B = Eigen::MatrixXd::Identity( n_x, n_u );
  ocp.dynamics      = [A, B]( const State& x, const Control& u ) { return A * x + B * u; };

  Eigen::MatrixXd Q = Eigen::MatrixXd::Identity( n_x, n_x );
  Eigen::MatrixXd R = Eigen::MatrixXd::Identity( n_u, n_u );
  ocp.stage_cost    = [Q, R]( const State& x, const Control& u, std::size_t ) {
    return ( x.transpose() * Q * x ).value() + ( u.transpose() * R * u ).value();
  };
  ocp.terminal_cost = []( const State& ) { return 0.0; };

  ocp.initialize_problem();
  ocp.verify_problem();
  return ocp;
}

struct Result
{
  std::string name;
  double      cost;
  double      time_ms;
};

/*────────────────────────────  main  ──────────────────────────────*/
int
main()
{
  using namespace mas;
  constexpr int    N   = 10;
  constexpr int    n_x = 4, n_u = 4, T = 10;
  constexpr double dt = 0.1;

  MultiAgentAggregator              agg;
  std::vector<std::shared_ptr<OCP>> ocps;
  ocps.reserve( N );
  for( int i = 0; i < N; ++i )
  {
    auto ocp = std::make_shared<OCP>( create_linear_lqr_ocp( n_x, n_u, dt, T ) );
    ocps.push_back( ocp );
    agg.agent_ocps[i] = ocp;
  }
  agg.compute_offsets();

  // example

  SolverParams p{
    { "max_iterations",  100 },
    {      "tolerance", 1e-5 },
    {         "max_ms",  100 }
  };
  constexpr int max_outer = 10;

  std::vector<Result> results;

  auto time_solver = [&]( const std::string& name, auto&& solver_call ) {
    agg.reset();
    auto start = std::chrono::high_resolution_clock::now();
    solver_call();
    auto                                      end     = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    results.push_back( { name, agg.agent_cost_sum(), elapsed.count() } );
  };

  time_solver( "Centralized iLQR", [&]() { agg.solve_centralized<iLQR>( p ); } );
  time_solver( "Centralized CGD", [&]() { agg.solve_centralized<CGD>( p ); } );
  time_solver( "Centralized OSQP", [&]() { agg.solve_centralized<OSQP>( p ); } );

  time_solver( "Decentralized iLQR (Trust Region)", [&]() { agg.solve_decentralized_trust_region<iLQR>( max_outer, p ); } );
  time_solver( "Decentralized iLQR (Line Search)", [&]() { agg.solve_decentralized_line_search<iLQR>( max_outer, p ); } );
  time_solver( "Decentralized CGD (Trust Region)", [&]() { agg.solve_decentralized_trust_region<CGD>( max_outer, p ); } );
  time_solver( "Decentralized CGD (Line Search)", [&]() { agg.solve_decentralized_line_search<CGD>( max_outer, p ); } );
  time_solver( "Decentralized OSQP (Trust Region)", [&]() { agg.solve_decentralized_trust_region<OSQP>( max_outer, p ); } );
  time_solver( "Decentralized OSQP (Line Search)", [&]() { agg.solve_decentralized_line_search<OSQP>( max_outer, p ); } );


  std::cout << std::fixed << std::setprecision( 6 );
  std::cout << "\n";
  std::cout << std::setw( 40 ) << std::left << "Method" << std::setw( 15 ) << "Cost" << std::setw( 15 ) << "Time (ms)" << "\n";
  std::cout << std::string( 70, '-' ) << "\n";

  for( const auto& r : results )
  {
    std::cout << std::setw( 40 ) << std::left << r.name << std::setw( 15 ) << r.cost << std::setw( 15 ) << r.time_ms << "\n";
  }
}
