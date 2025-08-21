#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>
#include <string>

#include "multi_agent_solver/agent.hpp"
#include "multi_agent_solver/models/single_track_model.hpp"
#include "multi_agent_solver/multi_agent_problem.hpp"
#include "multi_agent_solver/solvers/solver.hpp"
#include "multi_agent_solver/strategies/strategy.hpp"

mas::OCP create_single_track_circular_ocp(double initial_theta, double track_radius,
                                          double target_velocity, int time_steps) {
  using namespace mas;
  OCP problem;
  problem.state_dim = 4;
  problem.control_dim = 2;
  problem.horizon_steps = time_steps;
  problem.dt = 0.5;

  double x0 = track_radius * cos(initial_theta);
  double y0 = track_radius * sin(initial_theta);
  problem.initial_state = Eigen::VectorXd::Zero(problem.state_dim);
  problem.initial_state << x0, y0, 1.57 + initial_theta, 0.0;

  problem.dynamics = single_track_model;

  problem.stage_cost = [target_velocity, track_radius](const State& state, const Control& control, size_t) {
    const double w_track = 1.0, w_speed = 1.0, w_delta = 0.001, w_acc = 0.001;
    double x = state(0), y = state(1), vx = state(3);
    double delta = control(0), a_cmd = control(1);
    double distance_from_track = std::abs(std::sqrt(x*x + y*y) - track_radius);
    double speed_error = vx - target_velocity;
    return w_track*distance_from_track*distance_from_track +
           w_speed*speed_error*speed_error +
           w_delta*delta*delta + w_acc*a_cmd*a_cmd;
  };
  problem.terminal_cost = [](const State&) { return 0.0; };
  problem.input_lower_bounds = Eigen::VectorXd::Constant(problem.control_dim,-0.5);
  problem.input_upper_bounds = Eigen::VectorXd::Constant(problem.control_dim,0.5);

  problem.initialize_problem();
  problem.verify_problem();
  return problem;
}

struct Result { std::string name; double cost; double time_ms; };

int main(int argc, char** argv) {
  using namespace mas;
  SolverParams params{{"max_iterations",10},{"tolerance",1e-5},{"max_ms",1000}};
  constexpr int max_outer = 1;
  const int num_agents = (argc > 1) ? std::stoi(argv[1]) : 2;
  constexpr int time_steps = 10;
  constexpr double track_radius = 20.0;
  constexpr double target_velocity = 10.0;

  MultiAgentProblem problem;
  for(int i=0;i<num_agents;++i){
    double theta = 2.0*M_PI*i/num_agents;
    auto ocp = std::make_shared<OCP>(create_single_track_circular_ocp(theta,track_radius,target_velocity,time_steps));
    problem.add_agent(std::make_shared<Agent>(i, ocp));
  }

  std::vector<Result> results;
  auto time_solver = [&](const std::string& name, auto&& call) {
    for (auto& a : problem.agents) a->reset();
    auto start = std::chrono::high_resolution_clock::now();
    auto sol = call();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    results.push_back({name, sol.total_cost, elapsed.count()});
  };

  time_solver("Centralized CGD", [&](){
    Solver solver{std::in_place_type<CGD>};
    set_params(solver, params);
    Strategy strat = CentralizedStrategy{std::move(solver)};
    return mas::solve(strat, problem);
  });
#ifdef MAS_HAVE_OSQP
  time_solver("Centralized OSQP", [&](){
    Solver solver{std::in_place_type<OSQP>};
    set_params(solver, params);
    Strategy strat = CentralizedStrategy{std::move(solver)};
    return mas::solve(strat, problem);
  });
  time_solver("Centralized OSQP-collocation", [&](){
    Solver solver{std::in_place_type<OSQPCollocation>};
    set_params(solver, params);
    Strategy strat = CentralizedStrategy{std::move(solver)};
    return mas::solve(strat, problem);
  });
#endif
  time_solver("Nash Sequential CGD", [&](){
    Solver solver{std::in_place_type<CGD>};
    Strategy strat = SequentialNashStrategy{max_outer, std::move(solver), params};
    return mas::solve(strat, problem);
  });
#ifdef MAS_HAVE_OSQP
  time_solver("Nash Sequential OSQP", [&](){
    Solver solver{std::in_place_type<OSQP>};
    Strategy strat = SequentialNashStrategy{max_outer, std::move(solver), params};
    return mas::solve(strat, problem);
  });
  time_solver("Nash Sequential OSQP-collocation", [&](){
    Solver solver{std::in_place_type<OSQPCollocation>};
    Strategy strat = SequentialNashStrategy{max_outer, std::move(solver), params};
    return mas::solve(strat, problem);
  });
#endif

  time_solver("Nash LineSearch CGD", [&](){
    Solver solver{std::in_place_type<CGD>};
    Strategy strat = LineSearchNashStrategy{max_outer, std::move(solver), params};
    return mas::solve(strat, problem);
  });
#ifdef MAS_HAVE_OSQP
  time_solver("Nash LineSearch OSQP", [&](){
    Solver solver{std::in_place_type<OSQP>};
    Strategy strat = LineSearchNashStrategy{max_outer, std::move(solver), params};
    return mas::solve(strat, problem);
  });
  time_solver("Nash LineSearch OSQP-collocation", [&](){
    Solver solver{std::in_place_type<OSQPCollocation>};
    Strategy strat = LineSearchNashStrategy{max_outer, std::move(solver), params};
    return mas::solve(strat, problem);
  });
#endif

  std::cout<<std::fixed<<std::setprecision(3)<<"\n";
  std::cout<<std::setw(40)<<std::left<<"Method"<<std::setw(15)<<"Cost"<<std::setw(15)<<"Time (ms)"<<"\n";
  std::cout<<std::string(70,'-')<<"\n";
  for(const auto& r: results)
    std::cout<<std::setw(40)<<std::left<<r.name<<std::setw(15)<<r.cost<<std::setw(15)<<r.time_ms<<"\n";
}
