#pragma once

#include <algorithm>
#include <cctype>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "Eigen/Dense"

#include "multi_agent_solver/solvers/solver.hpp"
#include "multi_agent_solver/strategies/strategy.hpp"

namespace examples
{

inline std::string
normalize_key( std::string value )
{
  std::string result;
  result.reserve( value.size() );
  for( char ch : value )
  {
    if( std::isalnum( static_cast<unsigned char>( ch ) ) )
      result.push_back( static_cast<char>( std::tolower( static_cast<unsigned char>( ch ) ) ) );
  }
  return result;
}

inline std::string
canonical_solver_name( const std::string& name )
{
  const std::string key = normalize_key( name );
  if( key == "ilqr" )
    return "ilqr";
  if( key == "primaldualilqr" || key == "pdilqr" || key == "primal_dual_ilqr" )
    return "ilqr";
  if( key == "cgd" )
    return "cgd";
#ifdef MAS_HAVE_OSQP
  if( key == "osqp" )
    return "osqp";
  if( key == "osqpcollocation" )
    return "osqp_collocation";
#endif
  throw std::invalid_argument( "Unknown solver '" + name + "'." );
}

inline std::string
canonical_strategy_name( const std::string& name )
{
  const std::string key = normalize_key( name );
  if( key == "centralized" || key == "centralised" )
    return "centralized";
  if( key == "sequential" || key == "sequentialnash" )
    return "sequential";
  if( key == "linesearch" || key == "linesearchnash" )
    return "linesearch";
  if( key == "trustregion" || key == "trustregionnash" )
    return "trustregion";
  throw std::invalid_argument( "Unknown strategy '" + name + "'." );
}

inline std::vector<std::string>
available_solver_names()
{
  std::vector<std::string> names{ "ilqr", "cgd" };
#ifdef MAS_HAVE_OSQP
  names.push_back( "osqp" );
  names.push_back( "osqp_collocation" );
#endif
  return names;
}

inline mas::Solver
make_solver( const std::string& name )
{
  const std::string canonical = canonical_solver_name( name );
  if( canonical == "ilqr" )
    return mas::Solver{ std::in_place_type<mas::iLQR> };
  if( canonical == "cgd" )
    return mas::Solver{ std::in_place_type<mas::CGD> };
#ifdef MAS_HAVE_OSQP
  if( canonical == "osqp" )
    return mas::Solver{ std::in_place_type<mas::OSQP> };
  if( canonical == "osqp_collocation" )
    return mas::Solver{ std::in_place_type<mas::OSQPCollocation> };
#endif
  throw std::invalid_argument( "Unknown solver '" + name + "'." );
}

inline mas::Strategy
make_strategy( const std::string& name, mas::Solver solver, const mas::SolverParams& params, int max_outer )
{
  const std::string canonical = canonical_strategy_name( name );
  if( canonical == "centralized" )
  {
    mas::set_params( solver, params );
    return mas::Strategy{ mas::CentralizedStrategy{ std::move( solver ) } };
  }
  if( canonical == "sequential" )
    return mas::Strategy{ mas::SequentialNashStrategy{ max_outer, std::move( solver ), params } };
  if( canonical == "linesearch" )
    return mas::Strategy{ mas::LineSearchNashStrategy{ max_outer, std::move( solver ), params } };
  if( canonical == "trustregion" )
    return mas::Strategy{ mas::TrustRegionNashStrategy{ max_outer, std::move( solver ), params } };
  throw std::invalid_argument( "Unknown strategy '" + name + "'." );
}

inline void
print_available( std::ostream& os )
{
  const auto solvers = available_solver_names();
  os << "Available solvers:";
  for( const auto& solver : solvers )
    os << ' ' << solver;
  os << '\n';
  os << "Available strategies: centralized, sequential, linesearch, trustregion\n";
}

inline void
print_state_trajectory( std::ostream& os, const Eigen::MatrixXd& states, double dt, const std::string& label )
{
  if( states.size() == 0 )
    return;

  os << label << "_states\n";
  os << "time";
  for( int row = 0; row < states.rows(); ++row )
    os << ",x" << row;
  os << '\n';

  for( int col = 0; col < states.cols(); ++col )
  {
    const double time_value = dt > 0.0 ? static_cast<double>( col ) * dt : static_cast<double>( col );
    os << time_value;
    for( int row = 0; row < states.rows(); ++row )
      os << ',' << states( row, col );
    os << '\n';
  }
  os << '\n';
}

inline void
print_control_trajectory( std::ostream& os, const Eigen::MatrixXd& controls, double dt, const std::string& label )
{
  if( controls.size() == 0 )
    return;

  os << label << "_controls\n";
  os << "time";
  for( int row = 0; row < controls.rows(); ++row )
    os << ",u" << row;
  os << '\n';

  for( int col = 0; col < controls.cols(); ++col )
  {
    const double time_value = dt > 0.0 ? static_cast<double>( col ) * dt : static_cast<double>( col );
    os << time_value;
    for( int row = 0; row < controls.rows(); ++row )
      os << ',' << controls( row, col );
    os << '\n';
  }
  os << '\n';
}

} // namespace examples
