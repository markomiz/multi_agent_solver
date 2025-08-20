#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <tuple>
#include <vector>

#include "Eigen/Dense"

#include "multi_agent_solver/agent.hpp"
#include "multi_agent_solver/multi_agent_problem.hpp"
#include "multi_agent_solver/solvers/solver.hpp"
#include "multi_agent_solver/strategies/strategy.hpp"
#include "multi_agent_solver/types.hpp"

/*──────────────── create simple LQR OCP (unchanged) ───────────────*/
mas::OCP create_linear_lqr_ocp(int n_x, int n_u, double dt, int T) {
  using namespace mas;
  OCP ocp;
  ocp.state_dim = n_x;
  ocp.control_dim = n_u;
  ocp.dt = dt;
  ocp.horizon_steps = T;
  ocp.initial_state = Eigen::VectorXd::Random(n_x);

  Eigen::MatrixXd A = Eigen::MatrixXd::Identity(n_x, n_x);
  Eigen::MatrixXd B = Eigen::MatrixXd::Identity(n_x, n_u);
  ocp.dynamics = [A,B](const State& x, const Control& u){ return A*x + B*u; };

  Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(n_x, n_x);
  Eigen::MatrixXd R = Eigen::MatrixXd::Identity(n_u, n_u);
  ocp.stage_cost = [Q,R](const State& x, const Control& u, std::size_t){
    return (x.transpose()*Q*x).value() + (u.transpose()*R*u).value(); };
  ocp.terminal_cost = [](const State&){ return 0.0; };

  ocp.initialize_problem();
  ocp.verify_problem();
  return ocp;
}

struct Result { std::string name; double cost; double time_ms; };

/*────────────────────────────  main  ──────────────────────────────*/
int main() {
  using namespace mas;
  constexpr int N = 10;  constexpr int n_x=4, n_u=4, T=10;  constexpr double dt=0.1;

  MultiAgentProblem problem;
  for (int i=0;i<N;++i) {
    auto ocp = std::make_shared<OCP>(create_linear_lqr_ocp(n_x,n_u,dt,T));
    problem.add_agent(std::make_shared<Agent>(i, ocp));
  }

  SolverParams p{{"max_iterations",100},{"tolerance",1e-5},{"max_ms",100}};
  constexpr int max_outer=10;

  std::vector<Result> results;
  auto time_solver = [&](const std::string& name, auto&& call) {
    for (auto& a : problem.agents) a->reset();
    auto start = std::chrono::high_resolution_clock::now();
    auto sol = call();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    results.push_back({name, sol.total_cost, elapsed.count()});
  };

  time_solver("Centralized iLQR", [&](){
    Solver solver{std::in_place_type<iLQR>};
    set_params(solver, p);
    Strategy strat = CentralizedStrategy{solver};
    return mas::solve(strat, problem);
  });
  time_solver("Centralized CGD", [&](){
    Solver solver{std::in_place_type<CGD>};
    set_params(solver, p);
    Strategy strat = CentralizedStrategy{solver};
    return mas::solve(strat, problem);
  });
#ifdef MAS_HAVE_OSQP
  time_solver("Centralized OSQP", [&](){
    Solver solver{std::in_place_type<OSQP>};
    set_params(solver, p);
    Strategy strat = CentralizedStrategy{solver};
    return mas::solve(strat, problem);
  });
  time_solver("Centralized OSQP-collocation", [&](){
    Solver solver{std::in_place_type<OSQPCollocation>};
    set_params(solver, p);
    Strategy strat = CentralizedStrategy{solver};
    return mas::solve(strat, problem);
  });
#endif

  time_solver("Best-Response iLQR", [&](){
    Solver solver{std::in_place_type<iLQR>};
    set_params(solver, p);
    Strategy strat = BestResponseStrategy{max_outer, solver};
    return mas::solve(strat, problem);
  });
  time_solver("Best-Response CGD", [&](){
    Solver solver{std::in_place_type<CGD>};
    set_params(solver, p);
    Strategy strat = BestResponseStrategy{max_outer, solver};
    return mas::solve(strat, problem);
  });
#ifdef MAS_HAVE_OSQP
  time_solver("Best-Response OSQP", [&](){
    Solver solver{std::in_place_type<OSQP>};
    set_params(solver, p);
    Strategy strat = BestResponseStrategy{max_outer, solver};
    return mas::solve(strat, problem);
  });
  time_solver("Best-Response OSQP-collocation", [&](){
    Solver solver{std::in_place_type<OSQPCollocation>};
    set_params(solver, p);
    Strategy strat = BestResponseStrategy{max_outer, solver};
    return mas::solve(strat, problem);
  });
#endif

  std::cout<<std::fixed<<std::setprecision(6)<<"\n";
  std::cout<<std::setw(40)<<std::left<<"Method"<<std::setw(15)<<"Cost"<<std::setw(15)<<"Time (ms)"<<"\n";
  std::cout<<std::string(70,'-')<<"\n";
  for(const auto& r: results)
    std::cout<<std::setw(40)<<std::left<<r.name<<std::setw(15)<<r.cost<<std::setw(15)<<r.time_ms<<"\n";
}
