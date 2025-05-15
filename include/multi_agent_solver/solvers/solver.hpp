#include "multi_agent_solver/types.hpp"

using SolverParams = std::unordered_map<std::string, double>;
using Solver       = std::function<void( OCP&, const SolverParams& )>;