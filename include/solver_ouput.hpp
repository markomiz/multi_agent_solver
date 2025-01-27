#pragma once
#include "types.hpp"

struct SolverOutput
{
  double            cost = 0.0;
  ControlTrajectory controls;
  StateTrajectory   trajectory;
};