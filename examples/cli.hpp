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

