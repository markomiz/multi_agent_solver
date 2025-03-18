#pragma once

#include <omp.h>

#include <cassert>
#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

#include "ocp.hpp"

/**
 * @brief Struct to store each agent's dimensional offsets in the global vectors/matrices.
 */
struct AgentBlockInfo
{
  size_t               agent_id;
  int                  state_offset;
  int                  control_offset;
  int                  state_dim;
  int                  control_dim;
  std::shared_ptr<OCP> ocp_ptr;
};

class MultiAgentAggregator
{
public:

  using OCPPtr = std::shared_ptr<OCP>;

  std::unordered_map<size_t, OCPPtr> agent_ocps;
  std::optional<ObjectiveFunction>   cross_agent_cost                     = std::nullopt;
  std::optional<ConstraintsFunction> cross_agent_equality_constraints     = std::nullopt;
  std::optional<ConstraintsFunction> cross_agent_inequality_constraints   = std::nullopt;
  bool                               use_only_global_cost_and_constraints = false;
  std::vector<AgentBlockInfo>        agent_blocks;

  void
  compute_offsets()
  {
    agent_blocks.clear();
    agent_blocks.reserve( agent_ocps.size() );

    int state_offset   = 0;
    int control_offset = 0;
    for( const auto& [agent_id, ocp] : agent_ocps )
    {
      AgentBlockInfo info;
      info.agent_id       = agent_id;
      info.state_offset   = state_offset;
      info.control_offset = control_offset;
      info.state_dim      = ocp->state_dim;
      info.control_dim    = ocp->control_dim;
      info.ocp_ptr        = ocp;
      agent_blocks.push_back( info );

      state_offset   += ocp->state_dim;
      control_offset += ocp->control_dim;
    }
  }

  OCP
  create_global_ocp() const
  {
    assert( !agent_blocks.empty() && "No agent offsets computed. Call compute_offsets() first." );
    OCP global_ocp;

    set_total_dimensions( global_ocp );
    set_timing_info( global_ocp );
    set_initial_state( global_ocp );
    setup_dynamics( global_ocp );
    setup_objective_function( global_ocp );
    setup_equality_constraints( global_ocp );
    setup_inequality_constraints( global_ocp );

    global_ocp.initialize_problem();
    global_ocp.verify_problem();
    assert( global_ocp.objective_function && "❌ ERROR: Global OCP objective function was not set!" );

    return global_ocp;
  }

  double
  solve_centralized( const Solver& solver, int max_iterations, double tolerance )
  {
    std::cerr << "\n\nSolving Centralized..." << std::endl;

    if( agent_blocks.empty() )
      compute_offsets();

    OCP global_ocp = create_global_ocp();
    solver( global_ocp, max_iterations, tolerance );

    // Update individual agent OCPs with solved global trajectory
    for( const auto& block : agent_blocks )
    {
      block.ocp_ptr->best_states   = global_ocp.best_states.block( block.state_offset, 0, block.state_dim, global_ocp.best_states.cols() );
      block.ocp_ptr->best_controls = global_ocp.best_controls.block( block.control_offset, 0, block.control_dim,
                                                                     global_ocp.best_controls.cols() );

      block.ocp_ptr->best_cost = global_ocp.best_cost / agent_blocks.size();
    }
    return global_ocp.best_cost;
  }

  double
  agent_cost_sum()
  {
    // Compute total new cost
    double new_total_cost = 0;
    for( const auto& block : agent_blocks )
    {
      new_total_cost += block.ocp_ptr->best_cost;
    }
    return new_total_cost;
  }

  void
  solve_all_agents( const Solver& solver, int max_iterations, double tolerance )
  {
    // Run local optimization per agent
    // #pragma omp parallel for
    for( size_t i = 0; i < agent_blocks.size(); ++i )
    {
      solver( *agent_blocks[i].ocp_ptr, max_iterations, tolerance );
    }
  }

  double
  solve_decentralized_simple( const Solver& solver, int max_outer_iterations, int max_inner_iterations, double tolerance )
  {
    double total_cost = std::numeric_limits<double>::max();
    std::cerr << "\nSolving decentralized Simple..." << std::endl;

    for( int outer_iter = 0; outer_iter < max_outer_iterations; ++outer_iter )
    {

      solve_all_agents( solver, max_inner_iterations, tolerance );
      double new_total_cost = agent_cost_sum();

      // If the cost doesn't improve, terminate early
      if( total_cost > new_total_cost + tolerance )
      {
        total_cost = new_total_cost;
      }
      else
      {
        std::cerr << "Early termination at outer iteration " << outer_iter << std::endl;
        break;
      }
    }

    return total_cost;
  }

  double
  solve_decentralized_trust_region( const Solver& solver, int max_outer_iterations, int max_inner_iterations, double tolerance )
  {
    double total_cost = std::numeric_limits<double>::max();
    std::cerr << "\nSolving decentralized with trust region..." << std::endl;

    size_t              num_agents = agent_blocks.size();
    std::vector<double> delta( num_agents, 5.0 ); // Start with a large trust region
    const double        eta1 = 0.01, eta2 = 0.6;  // More lenient acceptance thresholds
    const double        shrink_factor = 0.8, expand_factor = 1.5;
    const double        min_delta = 1e-4, max_delta = 5.0;

    for( int outer_iter = 0; outer_iter < max_outer_iterations; ++outer_iter )
    {
      // Backup current best states and controls
      std::vector<ControlTrajectory> prev_controls( num_agents );
      std::vector<ControlTrajectory> new_controls( num_agents );
      for( size_t i = 0; i < num_agents; ++i )
      {
        prev_controls[i] = agent_blocks[i].ocp_ptr->best_controls;
      }

      // Run local optimization per agent
#pragma omp parallel for
      for( size_t i = 0; i < num_agents; ++i )
      {
        solver( *agent_blocks[i].ocp_ptr, max_inner_iterations, tolerance );
        new_controls[i] = agent_blocks[i].ocp_ptr->best_controls;
      }

      // Compute trial cost with full update
      double trial_cost = 0;
      for( auto& block : agent_blocks )
      {
        trial_cost += block.ocp_ptr->best_cost;
      }

      // Compute control deltas
      std::vector<ControlTrajectory> control_deltas( num_agents );
      for( size_t i = 0; i < num_agents; ++i )
      {
        control_deltas[i] = new_controls[i] - prev_controls[i];
      }

      // Perform trust-region updates
      double                         best_cost     = total_cost;
      std::vector<ControlTrajectory> best_controls = prev_controls;
      std::vector<double>            agent_rho( num_agents, -1.0 );

      for( size_t i = 0; i < num_agents; ++i )
      {
        double alpha             = delta[i]; // Start with the current trust-region size
        double best_alpha        = alpha;
        double agent_cost_before = total_cost;

        while( alpha > min_delta )
        {
          // Compute blended control update
          ControlTrajectory blended_control = prev_controls[i] + alpha * control_deltas[i];

          // Update agent's control and recompute trajectory
          auto& ocp         = *agent_blocks[i].ocp_ptr;
          ocp.best_controls = blended_control;
          ocp.best_states   = integrate_horizon( ocp.initial_state, blended_control, ocp.dt, ocp.dynamics, integrate_rk4 );

          // Compute new cost
          double new_cost = 0;
          for( auto& block : agent_blocks )
          {
            new_cost += block.ocp_ptr->best_cost;
          }

          // Compute actual vs predicted improvement
          double predicted_improvement = -( control_deltas[i].squaredNorm() ); // Quadratic approximation
          double actual_improvement    = total_cost - new_cost;
          double rho                   = actual_improvement / predicted_improvement;
          agent_rho[i]                 = rho; // Store for later adjustments
          // Adjust step size based on trust-region acceptance criteria
          if( rho > eta2 )
          {
            // Good improvement, expand trust region
            delta[i]         = std::min( max_delta, delta[i] * expand_factor );
            best_controls[i] = blended_control;
            best_alpha       = alpha;
            best_cost        = new_cost;
            break;
          }
          else if( rho > eta1 )
          {
            // Moderate improvement, accept update but do not change trust region
            best_controls[i] = blended_control;
            best_alpha       = alpha;
            best_cost        = new_cost;
            break;
          }
          else
          {
            // Poor improvement, shrink trust region and retry
            alpha *= shrink_factor;
          }
        }

        // If cost increased, shrink delta even more aggressively
        if( best_cost > agent_cost_before )
        {
          delta[i] *= 0.5;
        }
      }

      // Apply best found updates
      for( size_t i = 0; i < num_agents; ++i )
      {
        auto& ocp         = *agent_blocks[i].ocp_ptr;
        ocp.best_controls = best_controls[i];
        ocp.best_states   = integrate_horizon( ocp.initial_state, ocp.best_controls, ocp.dt, ocp.dynamics, integrate_rk4 );
      }

      // If overall cost worsened, revert back
      double total_new_cost = 0;
      for( auto& block : agent_blocks )
      {
        total_new_cost += block.ocp_ptr->best_cost;
      }

      if( total_new_cost > total_cost + tolerance )
      {
        std::cerr << "Global cost worsened! Reverting step..." << std::endl;
        for( size_t i = 0; i < num_agents; ++i )
        {
          auto& ocp         = *agent_blocks[i].ocp_ptr;
          ocp.best_controls = prev_controls[i];
          ocp.best_states   = integrate_horizon( ocp.initial_state, ocp.best_controls, ocp.dt, ocp.dynamics, integrate_rk4 );
        }
      }
      else
      {
        total_cost = total_new_cost;
      }

      // If no significant improvement, stop
      if( total_cost > best_cost - tolerance )
      {
        std::cerr << outer_iter << " outer iterations reached" << std::endl;
        break;
      }
    }

    return total_cost;
  }

  double
  solve_decentralized( const Solver& solver, int max_outer_iterations, int max_inner_iterations, double tolerance )
  {
    double total_cost = std::numeric_limits<double>::max();
    std::cerr << "\nSolving decentralized..." << std::endl;

    for( int outer_iter = 0; outer_iter < max_outer_iterations; ++outer_iter )
    {
      // Backup current best controls
      std::vector<ControlTrajectory> prev_controls( agent_blocks.size() );
      std::vector<ControlTrajectory> new_controls( agent_blocks.size() );
      std::vector<ControlTrajectory> control_delta( agent_blocks.size() );


      for( size_t i = 0; i < agent_blocks.size(); ++i )
      {
        prev_controls[i] = agent_blocks[i].ocp_ptr->best_controls;
      }
      new_controls = prev_controls;

      solve_all_agents( solver, max_inner_iterations, tolerance );

      for( size_t i = 0; i < agent_blocks.size(); ++i )
      {
        new_controls[i] = agent_blocks[i].ocp_ptr->best_controls;
      }

      // Compute per-agent control deltas
      for( size_t i = 0; i < agent_blocks.size(); ++i )
      {
        control_delta[i] = new_controls[i] - prev_controls[i];
      }

      double alpha      = 1.0;
      double best_alpha = alpha;
      double best_cost  = total_cost;

      while( alpha > 1e-6 )
      {
        double new_cost = 0.0;
        // Apply the best alpha found for each agent
        for( size_t i = 0; i < agent_blocks.size(); ++i )
        {
          auto& ocp          = *agent_blocks[i].ocp_ptr;
          ocp.best_controls  = prev_controls[i] + alpha * control_delta[i];
          ocp.best_states    = integrate_horizon( ocp.initial_state, ocp.best_controls, ocp.dt, ocp.dynamics, integrate_rk4 );
          new_cost          += ocp.objective_function( ocp.best_states, ocp.best_controls );
        }
        if( new_cost < best_cost )
        {
          best_alpha = alpha;
          best_cost  = new_cost;
          break;
        }

        alpha *= 0.5; // Reduce step size
      }
      // Apply the best alpha found for each agent
      for( size_t i = 0; i < agent_blocks.size(); ++i )
      {
        auto& ocp         = *agent_blocks[i].ocp_ptr;
        ocp.best_controls = prev_controls[i] + best_alpha * control_delta[i];
        ocp.best_states   = integrate_horizon( ocp.initial_state, ocp.best_controls, ocp.dt, ocp.dynamics, integrate_rk4 );
        ocp.best_cost     = ocp.objective_function( ocp.best_states, ocp.best_controls ); // Final cost update
      }

      // Compute total new cost
      double new_total_cost = agent_cost_sum();

      // If the cost doesn't improve, terminate early
      if( total_cost > new_total_cost + tolerance )
      {
        total_cost = new_total_cost;
      }
      else
      {
        std::cerr << "Early termination at outer iteration " << outer_iter << std::endl;
        break;
      }
    }

    return total_cost;
  }

  void
  reset()
  {
    for( auto& block : agent_blocks )
    {
      ( *block.ocp_ptr ).reset();
    }
  }

private:

  // Helper to set the total dimensions of the global OCP
  void
  set_total_dimensions( OCP& global_ocp ) const
  {
    int total_state_dim   = 0;
    int total_control_dim = 0;
    for( const auto& block : agent_blocks )
    {
      total_state_dim   += block.state_dim;
      total_control_dim += block.control_dim;
    }
    global_ocp.state_dim   = total_state_dim;
    global_ocp.control_dim = total_control_dim;
  }

  // Helper to copy timing information (horizon and dt) from the first agent
  void
  set_timing_info( OCP& global_ocp ) const
  {
    if( !agent_blocks.empty() )
    {
      auto& first_ocp          = agent_blocks.front().ocp_ptr;
      global_ocp.horizon_steps = first_ocp->horizon_steps;
      global_ocp.dt            = first_ocp->dt;
    }
  }

  // Helper to set the initial state for the global OCP
  void
  set_initial_state( OCP& global_ocp ) const
  {
    global_ocp.initial_state = State::Zero( global_ocp.state_dim );
    for( const auto& block : agent_blocks )
    {
      global_ocp.initial_state.segment( block.state_offset, block.state_dim ) = block.ocp_ptr->initial_state;
    }
  }

  // Helper to set up the global dynamics function by aggregating agent dynamics
  void
  setup_dynamics( OCP& global_ocp ) const
  {
    global_ocp.dynamics = [this]( const State& full_state, const Control& full_control ) {
      StateDerivative out = StateDerivative::Zero( full_state.size() );
      for( const auto& block : agent_blocks )
      {
        State   x_agent                                    = full_state.segment( block.state_offset, block.state_dim );
        Control u_agent                                    = full_control.segment( block.control_offset, block.control_dim );
        out.segment( block.state_offset, block.state_dim ) = block.ocp_ptr->dynamics( x_agent, u_agent );
      }
      return out;
    };
  }

  // Helper to set up the global objective function by summing agent objectives and cross-agent cost
  void
  setup_objective_function( OCP& global_ocp ) const
  {
    global_ocp.stage_cost = [this]( const State& full_x, const Control& full_u ) -> double {
      double total_cost = 0.0;

      if( !use_only_global_cost_and_constraints )
      {
        for( const auto& block : agent_blocks )
        {
          State   x_agent = full_x.segment( block.state_offset, block.state_dim );
          Control u_agent = full_u.segment( block.control_offset, block.control_dim );

          total_cost += block.ocp_ptr->stage_cost( x_agent, u_agent );
        }
      }

      if( cross_agent_cost.has_value() )
      {
        total_cost += ( *cross_agent_cost )( full_x, full_u );
      }

      return total_cost;
    };

    global_ocp.terminal_cost = [this]( const State& full_x ) -> double {
      double total_cost = 0.0;

      if( !use_only_global_cost_and_constraints )
      {
        for( const auto& block : agent_blocks )
        {
          State x_agent  = full_x.segment( block.state_offset, block.state_dim );
          total_cost    += block.ocp_ptr->terminal_cost( x_agent );
        }
      }

      return total_cost;
    };
  }

  // Helper to set up the global equality constraints function by concatenating agent and cross-agent constraints
  void
  setup_equality_constraints( OCP& global_ocp ) const
  {
    global_ocp.equality_constraints = [this]( const State& full_state, const Control& full_control ) {
      int total_eq_dim = 0;
      if( !use_only_global_cost_and_constraints )
      {
        for( const auto& block : agent_blocks )
        {
          auto& ocp = block.ocp_ptr;
          if( ocp->equality_constraints )
          {
            ConstraintViolations test  = ocp->equality_constraints( State::Zero( block.state_dim ), Control::Zero( block.control_dim ) );
            total_eq_dim              += test.size();
          }
        }
      }
      if( cross_agent_equality_constraints.has_value() )
      {
        ConstraintViolations test  = ( *cross_agent_equality_constraints )( State::Zero( full_state.size() ),
                                                                           Control::Zero( full_control.size() ) );
        total_eq_dim              += test.size();
      }

      ConstraintViolations eq_violations = ConstraintViolations::Zero( total_eq_dim );
      int                  offset        = 0;
      if( !use_only_global_cost_and_constraints )
      {
        for( const auto& block : agent_blocks )
        {
          auto& ocp = block.ocp_ptr;
          if( ocp->equality_constraints )
          {
            State                x_agent                      = full_state.segment( block.state_offset, block.state_dim );
            Control              u_agent                      = full_control.segment( block.control_offset, block.control_dim );
            ConstraintViolations agent_eq                     = ocp->equality_constraints( x_agent, u_agent );
            eq_violations.segment( offset, agent_eq.size() )  = agent_eq;
            offset                                           += agent_eq.size();
          }
        }
      }
      if( cross_agent_equality_constraints.has_value() )
      {
        ConstraintViolations c_eq                     = ( *cross_agent_equality_constraints )( full_state, full_control );
        eq_violations.segment( offset, c_eq.size() )  = c_eq;
        offset                                       += c_eq.size();
      }
      return eq_violations;
    };
  }

  // Helper to set up the global inequality constraints function by concatenating agent and cross-agent constraints
  void
  setup_inequality_constraints( OCP& global_ocp ) const
  {
    global_ocp.inequality_constraints = [this]( const State& full_state, const Control& full_control ) {
      int total_ineq_dim = 0;
      if( !use_only_global_cost_and_constraints )
      {
        for( const auto& block : agent_blocks )
        {
          auto& ocp = block.ocp_ptr;
          if( ocp->inequality_constraints )
          {
            ConstraintViolations test  = ocp->inequality_constraints( State::Zero( block.state_dim ), Control::Zero( block.control_dim ) );
            total_ineq_dim            += test.size();
          }
        }
      }
      if( cross_agent_inequality_constraints.has_value() )
      {
        ConstraintViolations test  = ( *cross_agent_inequality_constraints )( State::Zero( full_state.size() ),
                                                                             Control::Zero( full_control.size() ) );
        total_ineq_dim            += test.size();
      }

      ConstraintViolations ineq_violations = ConstraintViolations::Zero( total_ineq_dim );
      if( total_ineq_dim == 0 )
        return ineq_violations;

      int offset = 0;
      if( !use_only_global_cost_and_constraints )
      {
        for( const auto& block : agent_blocks )
        {
          auto& ocp = block.ocp_ptr;
          if( ocp->inequality_constraints )
          {
            State                x_agent                          = full_state.segment( block.state_offset, block.state_dim );
            Control              u_agent                          = full_control.segment( block.control_offset, block.control_dim );
            ConstraintViolations agent_ineq                       = ocp->inequality_constraints( x_agent, u_agent );
            ineq_violations.segment( offset, agent_ineq.size() )  = agent_ineq;
            offset                                               += agent_ineq.size();
          }
        }
      }
      if( cross_agent_inequality_constraints.has_value() )
      {
        ConstraintViolations c_ineq                       = ( *cross_agent_inequality_constraints )( full_state, full_control );
        ineq_violations.segment( offset, c_ineq.size() )  = c_ineq;
        offset                                           += c_ineq.size();
      }
      return ineq_violations;
    };
  }
};
