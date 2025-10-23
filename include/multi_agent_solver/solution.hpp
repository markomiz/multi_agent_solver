#pragma once
#include <vector>

#include "multi_agent_solver/ocp.hpp"

namespace mas
{

template<typename Scalar = double>
struct SolutionT
{
  using StateTrajectory   = StateTrajectoryT<Scalar>;
  using ControlTrajectory = ControlTrajectoryT<Scalar>;

  std::vector<StateTrajectory>   states;
  std::vector<ControlTrajectory> controls;
  std::vector<Scalar>            costs;
  Scalar                         total_cost = static_cast<Scalar>( 0 );
};

using Solution   = SolutionT<double>;
using Solutiond  = SolutionT<double>;
using Solutionf  = SolutionT<float>;

} // namespace mas
