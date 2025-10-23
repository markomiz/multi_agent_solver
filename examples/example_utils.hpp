#pragma once

#include <algorithm>
#include <cctype>
#include <ostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "Eigen/Dense"

#include "multi_agent_solver/solvers/solver.hpp"
#include "multi_agent_solver/strategies/strategy.hpp"
#include "multi_agent_solver/types.hpp"

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

inline std::string
canonical_scalar_name( const std::string& name )
{
  const std::string key = normalize_key( name );
  if( key == "float" || key == "single" || key == "f32" )
    return "float";
  if( key == "double" || key == "doubleprecision" || key == "f64" )
    return "double";
  throw std::invalid_argument( "Unknown scalar type '" + name + "'." );
}

template<typename Scalar>
inline std::string
scalar_label()
{
  if constexpr( std::is_same_v<Scalar, float> )
    return "float";
  if constexpr( std::is_same_v<Scalar, double> )
    return "double";
  return "unknown";
}

template<typename Scalar>
inline std::vector<std::string>
available_solver_names()
{
  std::vector<std::string> names{ "ilqr", "cgd" };
#ifdef MAS_HAVE_OSQP
  if constexpr( std::is_same_v<Scalar, double> )
  {
    names.push_back( "osqp" );
    names.push_back( "osqp_collocation" );
  }
#endif
  return names;
}

inline std::vector<std::string>
available_solver_names()
{
  return available_solver_names<double>();
}

template<typename Scalar>
inline bool
solver_supported_for_scalar( const std::string& canonical )
{
#ifdef MAS_HAVE_OSQP
  if constexpr( !std::is_same_v<Scalar, double> )
  {
    if( canonical == "osqp" || canonical == "osqp_collocation" )
      return false;
  }
#else
  if( canonical == "osqp" || canonical == "osqp_collocation" )
    return false;
#endif
  return true;
}

template<typename Scalar>
inline mas::SolverVariant<Scalar>
make_solver( const std::string& name )
{
  const std::string canonical = canonical_solver_name( name );
  if( canonical == "ilqr" )
    return mas::SolverVariant<Scalar>{ std::in_place_type<mas::iLQR<Scalar>> };
  if( canonical == "cgd" )
    return mas::SolverVariant<Scalar>{ std::in_place_type<mas::CGD<Scalar>> };
#ifdef MAS_HAVE_OSQP
  if( canonical == "osqp" )
  {
    if constexpr( std::is_same_v<Scalar, double> )
      return mas::SolverVariant<Scalar>{ std::in_place_type<mas::OSQP<Scalar>> };
  }
  if( canonical == "osqp_collocation" )
  {
    if constexpr( std::is_same_v<Scalar, double> )
      return mas::SolverVariant<Scalar>{ std::in_place_type<mas::OSQPCollocation<Scalar>> };
  }
#endif
  if( !solver_supported_for_scalar<Scalar>( canonical ) )
    throw std::invalid_argument( "Solver '" + name + "' is not available for scalar type '" + scalar_label<Scalar>() + "'." );
  throw std::invalid_argument( "Unknown solver '" + name + "'." );
}

inline mas::Solver
make_solver( const std::string& name )
{
  return make_solver<double>( name );
}

template<typename Scalar>
inline mas::StrategyT<Scalar>
make_strategy( const std::string& name, mas::SolverVariant<Scalar> solver, const mas::SolverParamsT<Scalar>& params,
               int max_outer )
{
  const std::string canonical = canonical_strategy_name( name );
  if( canonical == "centralized" )
  {
    mas::set_params( solver, params );
    return mas::StrategyT<Scalar>{ mas::CentralizedStrategy<Scalar>{ std::move( solver ) } };
  }
  if( canonical == "sequential" )
    return mas::StrategyT<Scalar>{ mas::SequentialNashStrategy<Scalar>{ max_outer, std::move( solver ), params } };
  if( canonical == "linesearch" )
    return mas::StrategyT<Scalar>{ mas::LineSearchNashStrategy<Scalar>{ max_outer, std::move( solver ), params } };
  if( canonical == "trustregion" )
    return mas::StrategyT<Scalar>{ mas::TrustRegionNashStrategy<Scalar>{ max_outer, std::move( solver ), params } };
  throw std::invalid_argument( "Unknown strategy '" + name + "'." );
}

inline mas::Strategy
make_strategy( const std::string& name, mas::Solver solver, const mas::SolverParams& params, int max_outer )
{
  return make_strategy<double>( name, std::move( solver ), params, max_outer );
}

inline void
print_available( std::ostream& os )
{
  const auto solvers = available_solver_names();
  os << "Available solvers:";
  for( const auto& solver : solvers )
    os << ' ' << solver;
  os << " (float and double; OSQP variants require double)\n";
  os << "Available strategies: centralized, sequential, linesearch, trustregion\n";
  os << "Scalar precisions: float, double\n";
}

template<typename Scalar>
inline void
print_state_trajectory( std::ostream& os, const mas::StateTrajectoryT<Scalar>& states, Scalar dt, const std::string& label )
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
    const double time_value
      = dt > static_cast<Scalar>( 0 ) ? static_cast<double>( col ) * static_cast<double>( dt ) : static_cast<double>( col );
    os << time_value;
    for( int row = 0; row < states.rows(); ++row )
      os << ',' << static_cast<double>( states( row, col ) );
    os << '\n';
  }
  os << '\n';
}

template<typename Scalar>
inline void
print_control_trajectory( std::ostream& os, const mas::ControlTrajectoryT<Scalar>& controls, Scalar dt,
                          const std::string& label )
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
    const double time_value
      = dt > static_cast<Scalar>( 0 ) ? static_cast<double>( col ) * static_cast<double>( dt ) : static_cast<double>( col );
    os << time_value;
    for( int row = 0; row < controls.rows(); ++row )
      os << ',' << static_cast<double>( controls( row, col ) );
    os << '\n';
  }
  os << '\n';
}

inline void
print_state_trajectory( std::ostream& os, const Eigen::MatrixXd& states, double dt, const std::string& label )
{
  print_state_trajectory<double>( os, states, dt, label );
}

inline void
print_control_trajectory( std::ostream& os, const Eigen::MatrixXd& controls, double dt, const std::string& label )
{
  print_control_trajectory<double>( os, controls, dt, label );
}

} // namespace examples
