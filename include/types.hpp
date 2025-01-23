#pragma once
#include <functional>

#include <Eigen/Dense>

using State             = Eigen::VectorXd;
using StateDerivative   = Eigen::VectorXd;
using Control           = Eigen::VectorXd;
using ControlTrajectory = Eigen::MatrixXd;
using StateTrajectory   = Eigen::MatrixXd;

using MotionModel       = std::function<StateDerivative( const State&, const Control& )>;
using ObjectiveFunction = std::function<double( const StateTrajectory&, const ControlTrajectory& )>;

using ConstraintViolations = Eigen::VectorXd;
using ConstraintsFunction  = std::function<ConstraintViolations( const State&, const Control& )>;
