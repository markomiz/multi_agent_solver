#pragma once
#include <cmath>

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

template<typename Scalar, typename Problem>
inline Scalar
compute_augmented_cost( const Problem& problem, const ConstraintViolationsT<Scalar>& equality_multipliers,
                        const ConstraintViolationsT<Scalar>& inequality_multipliers, Scalar penalty_parameter,
                        const StateTrajectoryT<Scalar>& states, const ControlTrajectoryT<Scalar>& controls )
{
  Scalar cost = problem.objective_function( states, controls );

  for( int t = 0; t < controls.cols(); ++t )
  {
    if( problem.equality_constraints )
    {
      ConstraintViolationsT<Scalar> eq_residuals = problem.equality_constraints( states.col( t ), controls.col( t ) );
      cost += equality_multipliers.dot( eq_residuals ) + static_cast<Scalar>( 0.5 ) * penalty_parameter * eq_residuals.squaredNorm();
    }

    if( problem.inequality_constraints )
    {
      ConstraintViolationsT<Scalar> ineq_residuals = problem.inequality_constraints( states.col( t ), controls.col( t ) );
      ConstraintViolationsT<Scalar> slack          = ineq_residuals.array().cwiseMax( Scalar( 0 ) ).matrix();
      cost += inequality_multipliers.dot( slack ) + static_cast<Scalar>( 0.5 ) * penalty_parameter * slack.squaredNorm();
    }
  }

  return cost;
}

template<typename Problem>
inline double
compute_augmented_cost( const Problem& problem, const ConstraintViolations& equality_multipliers,
                        const ConstraintViolations& inequality_multipliers, double penalty_parameter, const StateTrajectory& states,
                        const ControlTrajectory& controls )
{
  return compute_augmented_cost<double>( problem, equality_multipliers, inequality_multipliers, penalty_parameter, states, controls );
}

template<typename Scalar, typename Problem>
inline void
update_lagrange_multipliers( const Problem& problem, const StateTrajectoryT<Scalar>& states, const ControlTrajectoryT<Scalar>& controls,
                             ConstraintViolationsT<Scalar>& equality_multipliers, ConstraintViolationsT<Scalar>& inequality_multipliers,
                             Scalar penalty_parameter )
{
  for( int t = 0; t < controls.cols(); ++t )
  {
    if( problem.equality_constraints )
    {
      ConstraintViolationsT<Scalar> eq_residuals  = problem.equality_constraints( states.col( t ), controls.col( t ) );
      equality_multipliers                       += penalty_parameter * eq_residuals;
    }

    if( problem.inequality_constraints )
    {
      ConstraintViolationsT<Scalar> ineq_residuals  = problem.inequality_constraints( states.col( t ), controls.col( t ) );
      inequality_multipliers                       += penalty_parameter * ineq_residuals.array().cwiseMax( Scalar( 0 ) ).matrix();
    }
  }
}

template<typename Problem>
inline void
update_lagrange_multipliers( const Problem& problem, const StateTrajectory& states, const ControlTrajectory& controls,
                             ConstraintViolations& equality_multipliers, ConstraintViolations& inequality_multipliers,
                             double penalty_parameter )
{
  update_lagrange_multipliers<double>( problem, states, controls, equality_multipliers, inequality_multipliers, penalty_parameter );
}

template<typename Scalar, typename Problem>
inline void
increase_penalty_parameter( Scalar& penalty_parameter, const Problem& problem, const StateTrajectoryT<Scalar>& states,
                            const ControlTrajectoryT<Scalar>& controls, Scalar tolerance )
{
  Scalar eq_violation_norm   = static_cast<Scalar>( 0 );
  Scalar ineq_violation_norm = static_cast<Scalar>( 0 );

  for( int t = 0; t < controls.cols(); ++t )
  {
    if( problem.equality_constraints )
    {
      eq_violation_norm += problem.equality_constraints( states.col( t ), controls.col( t ) ).squaredNorm();
    }
    if( problem.inequality_constraints )
    {
      ineq_violation_norm
        += problem.inequality_constraints( states.col( t ), controls.col( t ) ).array().cwiseMax( Scalar( 0 ) ).matrix().squaredNorm();
    }
  }

  eq_violation_norm   = static_cast<Scalar>( std::sqrt( static_cast<double>( eq_violation_norm ) ) );
  ineq_violation_norm = static_cast<Scalar>( std::sqrt( static_cast<double>( ineq_violation_norm ) ) );

  if( eq_violation_norm > tolerance || ineq_violation_norm > tolerance )
  {
    penalty_parameter *= static_cast<Scalar>( 1.5 );
  }
}

template<typename Problem>
inline void
increase_penalty_parameter( double& penalty_parameter, const Problem& problem, const StateTrajectory& states,
                            const ControlTrajectory& controls, double tolerance )
{
  increase_penalty_parameter<double>( penalty_parameter, problem, states, controls, tolerance );
}

template<typename Scalar>
inline void
clamp_controls( ControlTrajectoryT<Scalar>& controls, const ControlT<Scalar>& lower_limit, const ControlT<Scalar>& upper_limit )
{
  for( int t = 0; t < controls.cols(); ++t )
  {
    controls.col( t ) = controls.col( t ).cwiseMin( upper_limit ).cwiseMax( lower_limit );
  }
}

inline void
clamp_controls( ControlTrajectory& controls, const Control& lower_limit, const Control& upper_limit )
{
  clamp_controls<double>( controls, lower_limit, upper_limit );
}

} // namespace mas
