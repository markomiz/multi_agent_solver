#pragma once

#include <cstddef>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "Eigen/Dense"

#include "multi_agent_solver/solution.hpp"

namespace examples
{

struct SolutionExportAgentView
{
  std::size_t                    id = 0;
  double                         dt = 0.0;
  const mas::StateTrajectory*    states = nullptr;
};

inline void
write_solution_json( const std::string& path,
                     const std::string& solver_name,
                     const std::vector<SolutionExportAgentView>& agents,
                     const std::optional<std::string>& strategy_name = std::nullopt )
{
  std::ofstream out( path, std::ios::trunc );
  if( !out.is_open() )
    throw std::runtime_error( "Failed to open dump path '" + path + "'" );

  out.setf( std::ios::fixed, std::ios::floatfield );
  out.precision( 16 );

  out << "{\n";
  out << "  \"solver\": \"" << solver_name << "\"";
  if( strategy_name )
    out << ",\n  \"strategy\": \"" << *strategy_name << "\"";
  out << ",\n  \"agents\": [\n";

  for( std::size_t idx = 0; idx < agents.size(); ++idx )
  {
    const auto& view = agents[idx];
    if( view.states == nullptr )
      throw std::runtime_error( "Null trajectory passed to write_solution_json" );

    const auto& mat = *view.states;
    const auto  rows = static_cast<std::size_t>( mat.rows() );
    const auto  cols = static_cast<std::size_t>( mat.cols() );

    out << "    {\n";
    out << "      \"id\": " << view.id << ",\n";
    out << "      \"dt\": " << view.dt << ",\n";
    out << "      \"state_dim\": " << rows << ",\n";
    out << "      \"horizon_steps\": " << ( cols == 0 ? 0 : cols - 1 ) << ",\n";
    out << "      \"states\": [\n";

    for( std::size_t col = 0; col < cols; ++col )
    {
      out << "        [";
      for( std::size_t row = 0; row < rows; ++row )
      {
        out << mat( static_cast<Eigen::Index>( row ), static_cast<Eigen::Index>( col ) );
        if( row + 1 < rows )
          out << ", ";
      }
      out << "]";
      if( col + 1 < cols )
        out << ",";
      out << "\n";
    }

    out << "      ]\n";
    out << "    }";
    if( idx + 1 < agents.size() )
      out << ",";
    out << "\n";
  }

  out << "  ]\n";
  out << "}\n";
}

} // namespace examples
