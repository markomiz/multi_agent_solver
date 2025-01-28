
#pragma once
#include <functional>
#include <optional>

#include <Eigen/Dense>

#include "finite_differences.hpp"
#include "types.hpp"

struct OCP
{

  // Dynamics and Objective
  State             initial_state;
  MotionModel       dynamics;
  ObjectiveFunction objective_function;

  int    control_dim   = 0;
  int    state_dim     = 0;
  int    horizon_steps = 0;
  double dt;

  // Static bounds
  std::optional<State>   state_lower_bounds;
  std::optional<State>   state_upper_bounds;
  std::optional<Control> input_lower_bounds;
  std::optional<Control> input_upper_bounds;

  ConstraintsFunction equality_constraints;
  ConstraintsFunction inequality_constraints;

  // Optional analytical derivatives
  DynamicsStateJacobian   dynamics_state_jacobian;
  DynamicsControlJacobian dynamics_control_jacobian;
  CostStateGradient       cost_state_gradient;
  CostControlGradient     cost_control_gradient;
  CostStateHessian        cost_state_hessian;
  CostControlHessian      cost_control_hessian;
  CostCrossTerm           cost_cross_term;

  void
  initialize_derivatives()
  {
    // use finite differences when derivatives are not specified
    if( !dynamics_state_jacobian )
      dynamics_state_jacobian = compute_dynamics_state_jacobian;
    if( !dynamics_control_jacobian )
      dynamics_control_jacobian = compute_dynamics_control_jacobian;
    if( !cost_state_gradient )
      cost_state_gradient = compute_cost_state_gradient;
    if( !cost_control_gradient )
      cost_control_gradient = compute_cost_control_gradient;
    if( !cost_state_hessian )
      cost_state_hessian = compute_cost_state_hessian;
    if( !cost_control_hessian )
      cost_control_hessian = compute_cost_control_hessian;
    if( !cost_cross_term )
      cost_cross_term = compute_cost_cross_term;
  }

  // Verify that the problem's dimensions and outputs are consistent
  bool
  verify_problem() const
  {
    assert( state_dim != 0 && "No state dimension" );
    assert( control_dim != 0 && "No control dimension" );
    assert( horizon_steps != 0 && "No horizon dimension" );
    assert( dt != 0.0 && "dt is 0.0" );

    assert( initial_state.size() == state_dim && "Initial state size does not match state dimension" );

    // Check initial state dimension
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

    // Test dynamics function
    State           test_state      = State::Zero( state_dim );
    Control         test_control    = Control::Zero( control_dim );
    StateDerivative dynamics_output = dynamics( test_state, test_control );
    assert( dynamics_output.size() == state_dim && "Dynamics output size mismatch" );

    // Test objective function
    StateTrajectory         test_states   = StateTrajectory::Zero( state_dim, 10 ); // Example trajectory length
    ControlTrajectory       test_controls = ControlTrajectory::Zero( control_dim, 10 );
    [[maybe_unused]] double cost          = objective_function( test_states, test_controls );

    // If constraints exist, test them
    if( inequality_constraints )
    {
      ConstraintViolations violations = inequality_constraints( test_state, test_control );
      assert( violations.size() >= 0 && "Inequality constraints output invalid size" );
    }
    if( equality_constraints )
    {
      ConstraintViolations violations = equality_constraints( test_state, test_control );
      assert( violations.size() >= 0 && "Equality constraints output invalid size" );
    }

    std::cout << "OCP problem verified successfully." << std::endl;
    return true;
  }
};
