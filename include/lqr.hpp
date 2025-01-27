#pragma once
#include <iostream>

#include <Eigen/Dense>

#include "ocp.hpp"
#include "solver_ouput.hpp"
#include "types.hpp"

// Function to compute the LQR gain matrix
inline Eigen::MatrixXd
compute_lqr_gain( const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R,
                  int max_iterations = 1000, double tolerance = 1e-6 )
{
  Eigen::MatrixXd P = Q;
  Eigen::MatrixXd K;

  for( int i = 0; i < max_iterations; ++i )
  {
    Eigen::MatrixXd P_next = Q + A.transpose() * P * A
                           - A.transpose() * P * B * ( R + B.transpose() * P * B ).inverse() * B.transpose() * P * A;
    if( ( P_next - P ).norm() < tolerance )
    {
      P = P_next;
      break;
    }
    P = P_next;
  }

  K = ( R + B.transpose() * P * B ).inverse() * B.transpose() * P * A;
  return K;
}
