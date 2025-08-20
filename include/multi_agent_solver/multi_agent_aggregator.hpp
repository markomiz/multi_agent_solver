// MultiAgentAggregator refactored: templated solver type
#pragma once

#include <omp.h>

#include <algorithm>
#include <cassert>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

#include "multi_agent_solver/integrator.hpp"
#include "multi_agent_solver/ocp.hpp"
#include "multi_agent_solver/solvers/solver.hpp"
#include "multi_agent_solver/types.hpp"

namespace mas
{


struct AgentBlockInfo
{
  std::size_t             agent_id;
  int                     state_offset;
  int                     control_offset;
  int                     state_dim;
  int                     control_dim;
  std::shared_ptr<OCP>    ocp_ptr;
  std::shared_ptr<Solver> solver; // one per agent, lives forever
};

class MultiAgentAggregator
{
public:

  using OCPPtr = std::shared_ptr<OCP>;

  std::unordered_map<std::size_t, OCPPtr> agent_ocps;
  std::optional<ObjectiveFunction>        cross_agent_cost                     = std::nullopt;
  std::optional<ConstraintsFunction>      cross_agent_equality_constraints     = std::nullopt;
  std::optional<ConstraintsFunction>      cross_agent_inequality_constraints   = std::nullopt;
  bool                                    use_only_global_cost_and_constraints = false;
  std::vector<AgentBlockInfo>             agent_blocks;

  //---------------------- offset bookkeeping ---------------------------//
  void
  compute_offsets()
  {
    agent_blocks.clear();
    agent_blocks.reserve( agent_ocps.size() );
    int s_off = 0, u_off = 0;
    for( const auto& [id, ocp] : agent_ocps )
    {
      agent_blocks.push_back( { id, s_off, u_off, ocp->state_dim, ocp->control_dim, ocp } );
      s_off += ocp->state_dim;
      u_off += ocp->control_dim;
    }
  }

  template<typename SolverT>
  void
  initialize_solvers( const SolverParams& p )
  {
    for( auto& blk : agent_blocks )
    {
      blk.solver = create<SolverT>();
      set_params( *blk.solver, p );
    }
  }

  //---------------------------------------------------------------------
  //  Centralised solve  – single solver instance
  //---------------------------------------------------------------------
  template<typename SolverT>
  double
  solve_centralized( const SolverParams& params )
  {
    if( agent_blocks.empty() )
      compute_offsets();
    OCP global_ocp = create_global_ocp();

    SolverT solver; // one solver instance
    solver.set_params( params );
    solver.solve( global_ocp );

    for( const auto& blk : agent_blocks )
    {
      blk.ocp_ptr->best_states   = global_ocp.best_states.block( blk.state_offset, 0, blk.state_dim, global_ocp.best_states.cols() );
      blk.ocp_ptr->best_controls = global_ocp.best_controls.block( blk.control_offset, 0, blk.control_dim,
                                                                   global_ocp.best_controls.cols() );
      blk.ocp_ptr->best_cost     = global_ocp.best_cost / agent_blocks.size();
    }
    return global_ocp.best_cost;
  }

  //---------------------------------------------------------------------
  //  Per‑agent parallel solve – creates one SolverT per agent
  //---------------------------------------------------------------------
  template<typename SolverT>
  void
  solve_all_agents( const SolverParams& params )
  {
#pragma omp parallel for
    for( size_t i = 0; i < agent_blocks.size(); ++i )
    {
      solve( *agent_blocks[i].solver, *agent_blocks[i].ocp_ptr );
    }
#pragma omp parallel for
    for( size_t i = 0; i < agent_blocks.size(); ++i )
    {
      agent_blocks[i].ocp_ptr->update_initial_with_best();
    }
  }

  //---------------------------------------------------------------------
  //  Decentralised methods – template wrappers call solve_all_agents()
  //---------------------------------------------------------------------
  template<typename SolverT>
  double
  solve_decentralized_simple( int max_outer, const SolverParams& params )
  {
    double       total_cost = agent_cost_sum();
    const double tol        = params.at( "tolerance" );
    initialize_solvers<SolverT>( params );

    for( int k = 0; k < max_outer; ++k )
    {
      solve_all_agents<SolverT>( params );
      double new_cost = agent_cost_sum();
      if( total_cost > new_cost + tol )
        total_cost = new_cost;
      else
        break;
    }
    return total_cost;
  }

  template<typename SolverT>
  double
  solve_decentralized_trust_region( int max_outer, const SolverParams& params )
  {
    initialize_solvers<SolverT>( params );

    const double tol = params.at( "tolerance" );
    std::size_t  n   = agent_blocks.size();

    std::vector<double> delta( n, 1.0 );
    const double        eta1 = 0.01, eta2 = 0.5;
    const double        shrink = 0.8, expand = 1.5;
    const double        min_delta = 1e-3, max_delta = 1.0;

    double total_cost = agent_cost_sum();

    for( int outer = 0; outer < max_outer; ++outer )
    {
      auto prev_ctrl = backup_controls();
      solve_all_agents<SolverT>( params );
      auto   new_ctrl   = backup_controls();
      auto   ctrl_delta = compute_control_deltas( prev_ctrl, new_ctrl );
      double trial_cost = agent_cost_sum();

      std::vector<ControlTrajectory> best_ctrl = prev_ctrl;
      double                         best_cost = total_cost;

      for( std::size_t i = 0; i < n; ++i )
      {
        double alpha = delta[i];
        while( alpha > min_delta )
        {
          ControlTrajectory blended = prev_ctrl[i] + alpha * ctrl_delta[i];
          auto&             ocp     = *agent_blocks[i].ocp_ptr;
          ocp.best_controls         = blended;
          ocp.best_states           = integrate_horizon( ocp.initial_state, blended, ocp.dt, ocp.dynamics, integrate_rk4 );

          double new_cost  = agent_cost_sum();
          double predicted = ( total_cost - trial_cost ) * alpha;
          double actual    = total_cost - new_cost;
          double rho       = ( predicted == 0.0 ? 0.0 : actual / predicted );

          if( rho > eta2 )
          {
            delta[i]     = std::min( max_delta, delta[i] * expand );
            best_ctrl[i] = blended;
            best_cost    = new_cost;
            break;
          }
          if( rho > eta1 )
          {
            best_ctrl[i] = blended;
            best_cost    = new_cost;
            break;
          }
          alpha *= shrink;
        }
      }
      apply_control_updates( best_ctrl );
      double new_total = agent_cost_sum();
      if( new_total > total_cost + tol )
        restore_controls( prev_ctrl );
      else
        total_cost = new_total;
      // if( total_cost > best_cost - tol )
      //   break;

      std::cerr << "outer loop : " << outer << std::endl;
    }
    return total_cost;
  }

  template<typename SolverT>
  double
  solve_decentralized_line_search( int max_outer, const SolverParams& params )
  {
    initialize_solvers<SolverT>( params );


    const double c1        = 1e-4; // Armijo fraction
    const double shrink    = 0.5;  // α ← α·shrink
    const double min_alpha = 1e-5; // stop backtracking here
    const double tol       = params.at( "tolerance" );

    double      total_cost = agent_cost_sum(); // current best
    std::size_t n          = agent_blocks.size();

    /* -------- outer iterations ----------------------------------- */
    for( int outer = 0; outer < max_outer; ++outer )
    {
      /* (1) solve every agent once in parallel ------------------ */
      auto prev_ctrl = backup_controls();  // U_k
      solve_all_agents<SolverT>( params ); // produces U_trial
      auto trial_ctrl = backup_controls(); // U_k + ΔU
      auto delta_ctrl = compute_control_deltas( prev_ctrl, trial_ctrl );

      /* (2) line search *per agent* ----------------------------- */
      std::vector<ControlTrajectory> best_ctrl = prev_ctrl;
      double                         best_cost = total_cost;

      for( std::size_t i = 0; i < n; ++i )
      {
        double alpha = 1.0;
        while( alpha >= min_alpha )
        {
          ControlTrajectory blended = prev_ctrl[i] + alpha * delta_ctrl[i];

          /* write into agent i only ------------------------ */
          auto& ocp         = *agent_blocks[i].ocp_ptr;
          ocp.best_controls = blended;
          ocp.best_states   = integrate_horizon( ocp.initial_state, blended, ocp.dt, ocp.dynamics, integrate_rk4 );

          double new_cost  = agent_cost_sum(); // global
          double predicted = -c1 * alpha *     // Armijo
                             ( total_cost - ocp.best_cost );
          if( new_cost <= total_cost + predicted )
          {
            best_ctrl[i] = blended; // accept
            best_cost    = new_cost;
            break;
          }
          alpha *= shrink; // back-track
        }
      }

      /* (3) apply accepted updates ------------------------------ */
      apply_control_updates( best_ctrl );
      double new_total = agent_cost_sum();

      if( total_cost - new_total < tol ) // no significant progress
        return new_total;

      total_cost = new_total; // continue outer loop
    }
    return total_cost;
  }

  //---------------------------------------------------------------------
  void
  reset()
  {
    for( auto& blk : agent_blocks )
      blk.ocp_ptr->reset();
  }

  //---------------------------------------------------------------------
  // helpers identical to earlier version (agent_cost_sum, backup_controls, etc.)
  //---------------------------------------------------------------------
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

  std::vector<ControlTrajectory>
  backup_controls()
  {
    std::vector<ControlTrajectory> controls( agent_blocks.size() );
    for( size_t i = 0; i < agent_blocks.size(); ++i )
      controls[i] = agent_blocks[i].ocp_ptr->best_controls;
    return controls;
  }

  void
  restore_controls( const std::vector<ControlTrajectory>& controls )
  {
    for( size_t i = 0; i < agent_blocks.size(); ++i )
      agent_blocks[i].ocp_ptr->best_controls = controls[i];
  }

  std::vector<ControlTrajectory>
  compute_control_deltas( const std::vector<ControlTrajectory>& prev, const std::vector<ControlTrajectory>& new_controls )
  {
    std::vector<ControlTrajectory> deltas( agent_blocks.size() );
    for( size_t i = 0; i < agent_blocks.size(); ++i )
      deltas[i] = new_controls[i] - prev[i];
    return deltas;
  }

  void
  apply_control_updates( const std::vector<ControlTrajectory>& controls )
  {
    for( size_t i = 0; i < agent_blocks.size(); ++i )
    {
      auto ocp_ptr           = agent_blocks[i].ocp_ptr;
      ocp_ptr->best_controls = controls[i];
      ocp_ptr->best_states   = integrate_horizon( ocp_ptr->initial_state, ocp_ptr->best_controls, ocp_ptr->dt, ocp_ptr->dynamics,
                                                  integrate_rk4 );
      ocp_ptr->update_initial_with_best();
    }
  }

private:

  // Global‑OCP helper definitions (unchanged from previous version)
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
    global_ocp.stage_cost = [this]( const State& full_x, const Control& full_u, size_t time_index ) -> Scalar {
      Scalar total_cost = 0.0;

      if( !use_only_global_cost_and_constraints )
      {
        for( const auto& block : agent_blocks )
        {
          State   x_agent = full_x.segment( block.state_offset, block.state_dim );
          Control u_agent = full_u.segment( block.control_offset, block.control_dim );

          total_cost += block.ocp_ptr->stage_cost( x_agent, u_agent, time_index );
        }
      }

      if( cross_agent_cost.has_value() )
      {
        total_cost += ( *cross_agent_cost )( full_x, full_u );
      }

      return total_cost;
    };

    global_ocp.terminal_cost = [this]( const State& full_x ) -> Scalar {
      Scalar total_cost = 0.0;

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
};

} // namespace mas
