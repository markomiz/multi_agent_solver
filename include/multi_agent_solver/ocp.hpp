#pragma once

#include <cassert>
#include <functional>
#include <limits>
#include <optional>

#include <Eigen/Dense>

#include "multi_agent_solver/finite_differences.hpp"
#include "multi_agent_solver/types.hpp"

namespace mas
{

template<typename Scalar>
inline Scalar
compute_trajectory_cost( const StateTrajectoryT<Scalar>& X, const ControlTrajectoryT<Scalar>& U,
                         const StageCostFunctionT<Scalar>& stage_cost, const TerminalCostFunctionT<Scalar>& terminal_cost )
{
  const int T   = U.cols();
  const int Tp1 = X.cols();

  Scalar cost = static_cast<Scalar>( 0 );
  for( int t = 0; t < T; ++t )
  {
    cost += stage_cost( X.col( t ), U.col( t ), static_cast<size_t>( t ) );
  }
  cost += terminal_cost( X.col( Tp1 - 1 ) );
  return cost;
}

inline double
compute_trajectory_cost( const StateTrajectory& X, const ControlTrajectory& U, const StageCostFunction& stage_cost,
                         const TerminalCostFunction& terminal_cost )
{
  return compute_trajectory_cost<double>( X, U, stage_cost, terminal_cost );
}

template<typename Scalar = double>
struct OCP
{
  using ScalarType             = Scalar;
  using State                  = StateT<Scalar>;
  using StateDerivative        = StateDerivativeT<Scalar>;
  using Control                = ControlT<Scalar>;
  using StateTrajectory        = StateTrajectoryT<Scalar>;
  using ControlTrajectory      = ControlTrajectoryT<Scalar>;
  using MotionModel            = MotionModelT<Scalar>;
  using ObjectiveFunction      = ObjectiveFunctionT<Scalar>;
  using StageCostFunction      = StageCostFunctionT<Scalar>;
  using TerminalCostFunction   = TerminalCostFunctionT<Scalar>;
  using ConstraintsFunction    = ConstraintsFunctionT<Scalar>;
  using ConstraintsJacobianFunction = ConstraintsJacobianFunctionT<Scalar>;
  using ConstraintViolations        = ConstraintViolationsT<Scalar>;
  using DynamicsStateJacobian       = DynamicsStateJacobianT<Scalar>;
  using DynamicsControlJacobian     = DynamicsControlJacobianT<Scalar>;
  using CostStateGradient           = CostStateGradientT<Scalar>;
  using CostControlGradient         = CostControlGradientT<Scalar>;
  using CostStateHessian            = CostStateHessianT<Scalar>;
  using CostControlHessian          = CostControlHessianT<Scalar>;
  using CostCrossTerm               = CostCrossTermT<Scalar>;
  using TerminalCostGradient        = TerminalCostGradientT<Scalar>;
  using TerminalCostHessian         = TerminalCostHessianT<Scalar>;

  // intial guess and output
  StateTrajectory   initial_states;
  ControlTrajectory initial_controls;

  StateTrajectory   best_states;
  ControlTrajectory best_controls;
  Scalar            best_cost = std::numeric_limits<Scalar>::max();

  // Dynamics and Objective
  State       initial_state;
  MotionModel dynamics;

  StageCostFunction    stage_cost
    = []( const State&, const Control&, size_t ) { return static_cast<Scalar>( 0 ); };
  TerminalCostFunction terminal_cost = []( const State& ) { return static_cast<Scalar>( 0 ); };
  // objective function is sum of all stage costs + terminal cost
  ObjectiveFunction objective_function;

  int    control_dim   = 0;
  int    state_dim     = 0;
  int    horizon_steps = 0;
  Scalar dt            = static_cast<Scalar>( 0 );

  // Static bounds
  std::optional<State>   state_lower_bounds = std::nullopt;
  std::optional<State>   state_upper_bounds = std::nullopt;
  std::optional<Control> input_lower_bounds = std::nullopt;
  std::optional<Control> input_upper_bounds = std::nullopt;

  // function constraints
  ConstraintsFunction equality_constraints;
  ConstraintsFunction inequality_constraints;

  ConstraintsJacobianFunction equality_constraints_state_jacobian;
  ConstraintsJacobianFunction equality_constraints_control_jacobian;
  ConstraintsJacobianFunction inequality_constraints_state_jacobian;
  ConstraintsJacobianFunction inequality_constraints_control_jacobian;

  // Optional analytical derivatives
  DynamicsStateJacobian   dynamics_state_jacobian;
  DynamicsControlJacobian dynamics_control_jacobian;
  CostStateGradient       cost_state_gradient;
  CostControlGradient     cost_control_gradient;
  CostStateHessian        cost_state_hessian;
  CostControlHessian      cost_control_hessian;
  CostCrossTerm           cost_cross_term;
  TerminalCostGradient    terminal_cost_gradient;
  TerminalCostHessian     terminal_cost_hessian;

  size_t id = 0;

  void
  reset()
  {
    initial_controls = ControlTrajectory::Zero( control_dim, horizon_steps );
    initial_states
      = integrate_horizon<Scalar>( initial_state, initial_controls, dt, dynamics, integrate_rk4<Scalar> );

    best_states   = initial_states;
    best_controls = initial_controls;

    best_cost = objective_function( initial_states, initial_controls );
  }

  void
  update_initial_with_best()
  {
    initial_controls = best_controls;
    initial_states   = best_states;
  }

  void
  initialize_problem()
  {
    // Ensure best_states and best_controls have correct sizes
    if( initial_controls.rows() != control_dim || initial_controls.cols() != horizon_steps )
    {
      initial_controls = ControlTrajectory::Zero( control_dim, horizon_steps );
    }
    initial_states = integrate_horizon<Scalar>( initial_state, initial_controls, dt, dynamics, integrate_rk4<Scalar> );

    best_states   = initial_states;
    best_controls = initial_controls;


    // use finite differences when derivatives are not specified
    if( !dynamics_state_jacobian )
      dynamics_state_jacobian = compute_dynamics_state_jacobian<Scalar>;
    if( !dynamics_control_jacobian )
      dynamics_control_jacobian = compute_dynamics_control_jacobian<Scalar>;
    if( !cost_state_gradient )
      cost_state_gradient = compute_cost_state_gradient<Scalar>;
    if( !cost_control_gradient )
      cost_control_gradient = compute_cost_control_gradient<Scalar>;
    if( !cost_state_hessian )
      cost_state_hessian = compute_cost_state_hessian<Scalar>;
    if( !cost_control_hessian )
      cost_control_hessian = compute_cost_control_hessian<Scalar>;
    if( !cost_cross_term )
      cost_cross_term = compute_cost_cross_term<Scalar>;

    if( !terminal_cost_gradient )
      terminal_cost_gradient = compute_terminal_cost_gradient<Scalar>;
    if( !terminal_cost_hessian )
      terminal_cost_hessian = compute_terminal_cost_hessian<Scalar>;

    if( equality_constraints )
    {
      if( !equality_constraints_state_jacobian )
      {
        auto equality_fn = equality_constraints;
        equality_constraints_state_jacobian = [equality_fn]( const State& state, const Control& control ) {
          return compute_constraints_state_jacobian<Scalar>( equality_fn, state, control );
        };
      }
      if( !equality_constraints_control_jacobian )
      {
        auto equality_fn = equality_constraints;
        equality_constraints_control_jacobian = [equality_fn]( const State& state, const Control& control ) {
          return compute_constraints_control_jacobian<Scalar>( equality_fn, state, control );
        };
      }
    }

    if( inequality_constraints )
    {
      if( !inequality_constraints_state_jacobian )
      {
        auto inequality_fn = inequality_constraints;
        inequality_constraints_state_jacobian = [inequality_fn]( const State& state, const Control& control ) {
          return compute_constraints_state_jacobian<Scalar>( inequality_fn, state, control );
        };
      }
      if( !inequality_constraints_control_jacobian )
      {
        auto inequality_fn = inequality_constraints;
        inequality_constraints_control_jacobian = [inequality_fn]( const State& state, const Control& control ) {
          return compute_constraints_control_jacobian<Scalar>( inequality_fn, state, control );
        };
      }
    }

    if( !objective_function )
    {
      auto stage_cost_local    = stage_cost;
      auto terminal_cost_local = terminal_cost;
      objective_function = [stage_cost_local, terminal_cost_local]( const StateTrajectory&   states,
                                                                    const ControlTrajectory& controls ) -> Scalar {
        return compute_trajectory_cost<Scalar>( states, controls, stage_cost_local, terminal_cost_local );
      };
    }
    best_cost = objective_function( initial_states, initial_controls );
  }

  // Verify that the problem's dimensions and outputs are consistent
  bool
  verify_problem() const
  {
    assert( state_dim != 0 && "No state dimension" );
    assert( control_dim != 0 && "No control dimension" );
    assert( horizon_steps != 0 && "No horizon dimension" );
    assert( dt != static_cast<Scalar>( 0 ) && "dt is 0.0" );

    assert( initial_state.size() == state_dim && "Initial state size does not match state dimension" );

    // Check bounds dimensions
    if( state_lower_bounds.has_value() )
    {
      assert( state_lower_bounds->size() == state_dim && "State lower bounds size mismatch" );
    }
    if( state_upper_bounds.has_value() )
    {
      assert( state_upper_bounds->size() == state_dim && "State upper bounds size mismatch" );
    }
    if( input_lower_bounds.has_value() )
    {
      assert( input_lower_bounds->size() == control_dim && "Input lower bounds size mismatch" );
    }
    if( input_upper_bounds.has_value() )
    {
      assert( input_upper_bounds->size() == control_dim && "Input upper bounds size mismatch" );
    }
    // Ensure cost functions are set
    assert( objective_function && "Objective cost function is not set." );

    // Test dynamics function
    StateDerivative dynamics_output = dynamics( best_states.col( 0 ), best_controls.col( 0 ) );
    assert( dynamics_output.size() == state_dim && "Dynamics output size mismatch" );

    // Test objective function
    Scalar cost = objective_function( best_states, best_controls );
    (void)cost;

    // If constraints exist, test them
    if( inequality_constraints )
    {
      ConstraintViolations violations = inequality_constraints( best_states.col( 0 ), best_controls.col( 0 ) );
      assert( violations.size() >= 0 && "Inequality constraints output invalid size" );
    }
    if( equality_constraints )
    {
      ConstraintViolations violations = equality_constraints( best_states.col( 0 ), best_controls.col( 0 ) );
      assert( violations.size() >= 0 && "Equality constraints output invalid size" );
    }

    return true;
  }
};

using OCPd = OCP<double>;
using OCPf = OCP<float>;

} // namespace mas
