#include <functional>
#include <iostream>

#include <Eigen/Dense>

#include "finite_differences.hpp"
#include "integrator.hpp"
#include "line_search.hpp"
#include "ocp.hpp"
#include "solver_output.hpp"
#include "types.hpp"

// Helper function to compute the augmented cost
inline double
compute_augmented_cost( const OCP& problem, const ConstraintViolations& equality_multipliers,
                        const ConstraintViolations& inequality_multipliers, double penalty_parameter, const StateTrajectory& states,
                        const ControlTrajectory& controls )
{
  double cost = problem.objective_function( states, controls );

  for( int t = 0; t < controls.cols(); ++t )
  {
    if( problem.equality_constraints )
    {
      ConstraintViolations eq_residuals  = problem.equality_constraints( states.col( t ), controls.col( t ) );
      cost                              += equality_multipliers.dot( eq_residuals ) + 0.5 * penalty_parameter * eq_residuals.squaredNorm();
    }

    if( problem.inequality_constraints )
    {
      ConstraintViolations ineq_residuals  = problem.inequality_constraints( states.col( t ), controls.col( t ) );
      ConstraintViolations slack           = ( ineq_residuals.array() > 0 ).select( ineq_residuals, 0 );
      cost                                += inequality_multipliers.dot( slack ) + 0.5 * penalty_parameter * slack.squaredNorm();
    }
  }

  return cost;
}

// Helper function to update Lagrange multipliers
inline void
update_lagrange_multipliers( const OCP& problem, const StateTrajectory& states, const ControlTrajectory& controls,
                             ConstraintViolations& equality_multipliers, ConstraintViolations& inequality_multipliers,
                             double penalty_parameter )
{
  for( int t = 0; t < controls.cols(); ++t )
  {
    if( problem.equality_constraints )
    {
      ConstraintViolations eq_residuals  = problem.equality_constraints( states.col( t ), controls.col( t ) );
      equality_multipliers              += penalty_parameter * eq_residuals;
    }

    if( problem.inequality_constraints )
    {
      ConstraintViolations ineq_residuals  = problem.inequality_constraints( states.col( t ), controls.col( t ) );
      inequality_multipliers              += penalty_parameter * ( ineq_residuals.array() > 0 ).select( ineq_residuals, 0 );
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
