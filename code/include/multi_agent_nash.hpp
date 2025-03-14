// #pragma once

// #include <memory>
// #include <vector>

// #include "multi_agent_aggregator.hpp"
// #include "ocp.hpp"

// class MultiAgentNashSolver
// {
// public:

// using OCPPtr         = std::shared_ptr<OCP>;
// using SolverFunction = std::function<SolverOutput( const OCP&, int, double )>;

// MultiAgentNashSolver( std::vector<OCPPtr> agents, SolverFunction solver, int max_outer_iterations, double tolerance ) :
//   agents_( std::move( agents ) ),
//   solver_( std::move( solver ) ),
//   max_outer_iterations_( max_outer_iterations ),
//   tolerance_( tolerance )
// {
//   for( const auto& agent : agents_ )
//   {
//     agent_predictions_.emplace_back( agent->initial_state, agent->horizon_steps );
//   }
// }

// std::vector<SolverOutput>
// solve()
// {
//   std::vector<SolverOutput> solutions( agents_.size() );

// for( int outer_iter = 0; outer_iter < max_outer_iterations_; ++outer_iter )
// {
//   for( size_t i = 0; i < agents_.size(); ++i )
//   {
//     // Set the agent's perceived environment as the latest predictions
//     agents_[i]->set_other_agents_predictions( agent_predictions_ );

// // Solve for the current agent
// solutions[i] = solver_( *agents_[i], max_inner_iterations_, tolerance_ );

// // Update predictions
// agent_predictions_[i] = solutions[i].trajectory;
// }
// }
// return solutions;
// }

// private:

// std::vector<OCPPtr> agents;
// SolverFunction      solver;
// int                 max_outer_iterations = 10;
// int                 max_inner_iterations = 10; // Steps per agent per outer iteration
// double              tolerance            = 1e-5;
// };
