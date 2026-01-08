#pragma once
#include <functional>
#include <iostream>

#include <Eigen/Dense>

#include "multi_agent_solver/finite_differences.hpp"
#include "multi_agent_solver/integrator.hpp"
#include "multi_agent_solver/line_search.hpp"
#include "multi_agent_solver/ocp.hpp"
#include "multi_agent_solver/types.hpp"

namespace mas
{

// Helper function to compute the augmented cost
inline double
compute_augmented_cost( const OCP& problem, const ConstraintViolationsTrajectory& equality_multipliers,
                        const ConstraintViolationsTrajectory& inequality_multipliers, double penalty_parameter, const StateTrajectory& states,
                        const ControlTrajectory& controls )
{
  double cost = problem.objective_function( states, controls );

  for( int t = 0; t < controls.cols(); ++t )
  {
    if( problem.equality_constraints )
    {
      ConstraintViolations eq_residuals  = problem.equality_constraints( states.col( t ), controls.col( t ) );
      if (equality_multipliers.cols() > t) {
        cost += equality_multipliers.col(t).dot( eq_residuals ) + 0.5 * penalty_parameter * eq_residuals.squaredNorm();
      }
    }

    if( problem.inequality_constraints )
    {
      ConstraintViolations ineq_residuals  = problem.inequality_constraints( states.col( t ), controls.col( t ) );
      if (inequality_multipliers.cols() > t) {
        // PHR augmented Lagrangian term for inequalities:
        // (1 / 2rho) * ( max(0, lambda + rho * g)^2 - lambda^2 )
        const auto& lambda = inequality_multipliers.col(t);
        Eigen::VectorXd combined = lambda + penalty_parameter * ineq_residuals;
        Eigen::VectorXd combined_plus = combined.cwiseMax(0.0);
        cost += (0.5 / penalty_parameter) * (combined_plus.squaredNorm() - lambda.squaredNorm());
      }
    }
  }

  return cost;
}

// Helper function to update Lagrange multipliers
inline void
update_lagrange_multipliers( const OCP& problem, const StateTrajectory& states, const ControlTrajectory& controls,
                             ConstraintViolationsTrajectory& equality_multipliers, ConstraintViolationsTrajectory& inequality_multipliers,
                             double penalty_parameter )
{
  for( int t = 0; t < controls.cols(); ++t )
  {
    if( problem.equality_constraints )
    {
      ConstraintViolations eq_residuals  = problem.equality_constraints( states.col( t ), controls.col( t ) );
      if (equality_multipliers.cols() > t) {
          equality_multipliers.col(t) += penalty_parameter * eq_residuals;
      }
    }

    if( problem.inequality_constraints )
    {
      ConstraintViolations ineq_residuals  = problem.inequality_constraints( states.col( t ), controls.col( t ) );
      if (inequality_multipliers.cols() > t) {
          // Update rule: lambda_next = max(0, lambda + rho * g)
          inequality_multipliers.col(t) = (inequality_multipliers.col(t) + penalty_parameter * ineq_residuals).cwiseMax(0.0);
      }
    }
  }
}

// Helper function to increase penalty parameter
inline void
increase_penalty_parameter( double& penalty_parameter, const OCP& problem, const StateTrajectory& states, const ControlTrajectory& controls,
                            double tolerance )
{
  double eq_violation_norm   = 0.0;
  double ineq_violation_norm = 0.0;

  for( int t = 0; t < controls.cols(); ++t )
  {
    if( problem.equality_constraints )
    {
      eq_violation_norm += problem.equality_constraints( states.col( t ), controls.col( t ) ).squaredNorm();
    }
    if( problem.inequality_constraints )
    {
      ineq_violation_norm += problem.inequality_constraints( states.col( t ), controls.col( t ) ).cwiseMax( 0 ).squaredNorm();
    }
  }

  eq_violation_norm   = std::sqrt( eq_violation_norm );
  ineq_violation_norm = std::sqrt( ineq_violation_norm );

  if( eq_violation_norm > tolerance || ineq_violation_norm > tolerance )
  {
    penalty_parameter *= 1.5;
  }
}

inline void
clamp_controls( ControlTrajectory& controls, const Control& lower_limit, const Control& upper_limit )
{
  for( int t = 0; t < controls.cols(); ++t )
  {
    controls.col( t ) = controls.col( t ).cwiseMin( upper_limit ).cwiseMax( lower_limit );
  }
}
}