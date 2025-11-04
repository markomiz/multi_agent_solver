#pragma once

#include <algorithm>
#include <charconv>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

namespace examples
{
namespace cli
{

inline std::string
normalize_option( std::string option )
{
  if( option.rfind( "--", 0 ) != 0 )
    return option;

  const auto eq_pos = option.find( '=' );
  const auto end    = eq_pos == std::string::npos ? option.size() : eq_pos;
  std::replace( option.begin() + 2, option.begin() + static_cast<std::ptrdiff_t>( end ), '_', '-' );
  return option;
}

inline bool
is_positional( std::string_view arg )
{
  return !arg.empty() && arg.front() != '-';
}

inline int
parse_int( const std::string& label, const std::string& value )
{
  int         result   = 0;
  const char* begin    = value.data();
  const char* end      = begin + value.size();
  const auto [ptr, ec] = std::from_chars( begin, end, result );
  if( ec != std::errc() || ptr != end )
    throw std::invalid_argument( "Invalid value for " + label + ": '" + value + "'" );
  return result;
}

class ArgParser
{
public:
  ArgParser( int argc, char** argv ) : argc_( argc ), argv_( argv ) {}

  bool
  empty() const
  {
    return index_ >= argc_;
  }

  std::string_view
  peek() const
  {
    if( empty() )
      return std::string_view{};
    return argv_[index_];
  }

  std::string
  take()
  {
    if( empty() )
      throw std::out_of_range( "No more arguments to consume" );
    return std::string( argv_[index_++] );
  }

  bool
  consume_flag( std::string_view long_name, std::string_view short_name = {} )
  {
    if( empty() )
      return false;

    std::string current = normalize_option( std::string( peek() ) );
    if( current == long_name || ( !short_name.empty() && current == short_name ) )
    {
      ++index_;
      return true;
    }
    return false;
  }

  bool
  consume_option( std::string_view long_name, std::string& value )
  {
    if( empty() )
      return false;

    std::string current = normalize_option( std::string( peek() ) );
    const std::string prefix = std::string( long_name ) + "=";
    if( current == long_name )
    {
      ++index_;
      if( empty() )
        throw std::invalid_argument( "Missing value for option '" + std::string( long_name ) + "'" );
      value = take();
      return true;
    }
    if( current.rfind( prefix, 0 ) == 0 )
    {
      value = current.substr( prefix.size() );
      ++index_;
      return true;
    }
    return false;
  }

private:
  int    argc_  = 0;
  char** argv_  = nullptr;
  int    index_ = 1;
};

} // namespace cli
} // namespace examples

namespace examples
{
namespace cli
{

struct SolverOptions
{
  bool        show_help = false;
  std::string solver    = "ilqr";
};

inline SolverOptions
parse_solver_options( int argc, char** argv, std::string default_solver = "ilqr" )
{
  SolverOptions          options;
  options.solver = std::move( default_solver );
  ArgParser args( argc, argv );

  while( !args.empty() )
  {
    const std::string raw_arg = std::string( args.peek() );
    if( args.consume_flag( "--help", "-h" ) )
    {
      options.show_help = true;
      continue;
    }

    std::string value;
    if( args.consume_option( "--solver", value ) )
    {
      options.solver = value;
      continue;
    }

    throw std::invalid_argument( "Unknown argument '" + raw_arg + "'" );
  }

  return options;
}

struct MultiAgentOptions
{
  bool        show_help = false;
  int         agents    = 10;
  int         max_outer = 10;
  std::string solver    = "ilqr";
  std::string strategy  = "centralized";
};

inline MultiAgentOptions
parse_multi_agent_options( int argc, char** argv, MultiAgentOptions defaults = {} )
{
  MultiAgentOptions options = std::move( defaults );
  ArgParser         args( argc, argv );
  bool              positional_agents = false;

  while( !args.empty() )
  {
    const std::string raw_arg = std::string( args.peek() );
    if( args.consume_flag( "--help", "-h" ) )
    {
      options.show_help = true;
      continue;
    }

    std::string value;
    if( args.consume_option( "--agents", value ) )
    {
      options.agents = parse_int( "--agents", value );
      continue;
    }
    if( args.consume_option( "--solver", value ) )
    {
      options.solver = value;
      continue;
    }
    if( args.consume_option( "--strategy", value ) )
    {
      options.strategy = value;
      continue;
    }
    if( args.consume_option( "--max-outer", value ) )
    {
      options.max_outer = parse_int( "--max-outer", value );
      continue;
    }

    if( is_positional( raw_arg ) && !positional_agents )
    {
      args.take();
      options.agents    = parse_int( "agents", raw_arg );
      positional_agents = true;
      continue;
    }

    throw std::invalid_argument( "Unknown argument '" + raw_arg + "'" );
  }

  return options;
}

struct RocketOptions
{
  bool        show_help   = false;
  bool        dump_traces = false;
  std::string solver      = "osqp";
};

inline RocketOptions
parse_rocket_options( int argc, char** argv, RocketOptions defaults = {} )
{
  RocketOptions options = std::move( defaults );
  ArgParser     args( argc, argv );

  while( !args.empty() )
  {
    const std::string raw_arg = std::string( args.peek() );
    if( args.consume_flag( "--help", "-h" ) )
    {
      options.show_help = true;
      continue;
    }
    if( args.consume_flag( "--dump" ) )
    {
      options.dump_traces = true;
      continue;
    }

    std::string value;
    if( args.consume_option( "--solver", value ) )
    {
      options.solver = value;
      continue;
    }

    throw std::invalid_argument( "Unknown argument '" + raw_arg + "'" );
  }

  return options;
}

} // namespace cli
} // namespace examples

