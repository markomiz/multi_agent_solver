#include <functional>
#include <iostream>

#include <Eigen/Dense>

#include "ocp.hpp"
#include "types.hpp"

struct UnconstrainedGradientDecent
{
  // Parameters for the solver
  double learning_rate  = 0.01; // Step size for gradient descent
  int    max_iterations = 100;  // Maximum number of iterations
  double tolerance      = 1e-6; // Convergence tolerance

  // Solve the OCP
  ControlTrajectory
  solve( const OCP& ocp, int horizon_length )
  {
    // Initialize control trajectory (zeros)
    ControlTrajectory controls = ControlTrajectory::Zero( ocp.initial_state.size(), horizon_length );

    for( int iter = 0; iter < max_iterations; ++iter )
    {
      // Forward simulate to get the state trajectory
      StateTrajectory states = forward_simulate( ocp, controls );

      // Evaluate the objective function
      double cost = ocp.objective_function( states, controls );

      // Compute gradients using finite differences
      ControlTrajectory gradients = compute_gradients( ocp, states, controls );

      // Update the controls using gradient descent
      controls -= learning_rate * gradients;

      // Check convergence
      if( gradients.norm() < tolerance )
      {
        std::cout << "Converged after " << iter + 1 << " iterations.\n";
        break;
      }
    }

    return controls;
  }

private:

  // Forward simulate the dynamics to get the state trajectory
  StateTrajectory
  forward_simulate( const OCP& ocp, const ControlTrajectory& controls )
  {
    int horizon_length = controls.cols();
    int state_dim      = ocp.initial_state.size();

    StateTrajectory states( state_dim, horizon_length + 1 );
    states.col( 0 ) = ocp.initial_state; // Set the initial state

    for( int t = 0; t < horizon_length; ++t )
    {
      states.col( t + 1 ) = states.col( t ) + ocp.dynamics( states.col( t ), controls.col( t ) );
    }

    return states;
  }

  // Compute gradients of the objective function using finite differences
  ControlTrajectory
  compute_gradients( const OCP& ocp, const StateTrajectory& states, const ControlTrajectory& controls )
  {
    double            epsilon   = 1e-4; // Finite difference step size
    ControlTrajectory gradients = ControlTrajectory::Zero( controls.rows(), controls.cols() );

    for( int t = 0; t < controls.cols(); ++t )
    {
      for( int i = 0; i < controls.rows(); ++i )
      {
        // Perturb the control
        ControlTrajectory perturbed_controls  = controls;
        perturbed_controls( i, t )           += epsilon;

        // Compute the perturbed objective
        StateTrajectory perturbed_states = forward_simulate( ocp, perturbed_controls );
        double          cost_perturbed   = ocp.objective_function( perturbed_states, perturbed_controls );

        // Compute the original objective
        double cost_original = ocp.objective_function( states, controls );

        // Approximate the gradient
        gradients( i, t ) = ( cost_perturbed - cost_original ) / epsilon;
      }
    }

    return gradients;
  }
};
