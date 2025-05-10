# dynamic_solver

### Overview

MultiAgentSolver is a high-performance C++ library designed to solve multi-agent optimization problems. The library includes solvers such as:

* **CGD (Constrained Gradient Descent)** 
* **iLQR (Iterative Linear Quadratic Regulator)** 
* **OSQP Solver** utilizing the osqp library

Additionally, it supports multi-agent coordination through Nash Equilibrium-based optimization and for comparison combining multiple agents into one big optimization problem.

---

### Dependencies

The following dependencies are required:

* [Eigen3](https://eigen.tuxfamily.org/dox/) - Linear Algebra Library
* [OSQP](https://github.com/osqp/osqp) - Quadratic Programming Solver
* [OsqpEigen](https://github.com/robotology/OsqpEigen) - Eigen wrapper for OSQP
* [OpenMP](https://www.openmp.org/resources/openmp-compilers-tools/) - Multithreading support

**Optional:**

* [MKL (Intel Math Kernel Library)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html) - for optimized multithreading

Install these dependencies before building the project.

---

## Build & Running Examples

The project can be built easily using the provided Makefile.

To buld and run the Release Version:

`make run`

To build in Debug Mode with AddressSanitizer:

`make debug`

To build in Release Mode with Debug Symbols (for profiling):

`make release_debug`

To clean the build files:

`make clean`

To run the Debug Version with GDB:

`make test`

The main executable will execute the following examples:

Single Track Test: Optimizes a single agent trajectory.

Multi-Agent LQR Example: Runs an LQR optimization for multiple agents.

Multi-Agent Circular Test: Simulates multiple agents in a circular path with different configurations.

You can find the examples in code/examples/:

`multi_agent_lqr.hpp`

`multi_agent_single_track.hpp`

`single_track_ocp.hpp`

### Usage

The main entry point is in `main.cpp`. You can adjust which tests are executed by commenting/uncommenting the lines:

```cpp
single_track_test();
multi_agent_lqr_example();
multi_agent_circular_test(10, 15);
```

You can modify the agent count and parameters directly to experiment with different optimization settings.

---

## Results

Times and costs for different methods in the example code:
```
========================================================================================
Outer Method                            Solver              Time (ms)           Cost                
========================================================================================
 Centralized                            CGD                 117.26              86.9948             
 Centralized                            OSQP                3685.16             75.9386             
 Centralized                            iLQR                2079.98             77.6986             
 Decentralized_Line_Search              CGD                 106.85              77.3723             
 Decentralized_Line_Search              OSQP                34.80               74.5437             
 Decentralized_Line_Search              iLQR                11.27               77.7030             
 Decentralized_Simple                   CGD                 138.99              77.7806             
 Decentralized_Simple                   OSQP                43.13               74.9514             
 Decentralized_Simple                   iLQR                10.17               78.1107             
 Decentralized_Trust_Region             CGD                 3.02                87.4026             
 Decentralized_Trust_Region             OSQP                9.96                76.3330             
 Decentralized_Trust_Region             iLQR                6.02                78.1107             
=======================================================================================
```


## License
Apache 2.0

## Project status
WIP 
