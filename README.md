# MultiAgentSolver

MultiAgentSolver is a high-performance C++ library designed to solve multi-agent optimization problems. The library includes solvers such as:

* **CGD (Constrained Gradient Descent)** 
* **iLQR (Iterative Linear Quadratic Regulator)** 
* **OSQP Solver** utilizing the osqp library

Additionally, it supports multi-agent coordination through Nash Equilibrium-based optimization and for comparison combining multiple agents into one big optimization problem.

## 🛠️ **Dependencies**

* **Eigen3** - Linear algebra
* **OSQP** - Optimization solver
* **OsqpEigen** - Eigen wrapper for OSQP
* **OpenMP** - Multi-threading support

All dependencies can be automatically installed with the setup_dependencies.sh script 

---

## 📦 **Building with Docker**

The recommended way to build and run the project is using Docker. This ensures a clean environment and reproducible builds.

```bash
./scripts/run_docker.sh
```

This will:

* Clean up any existing Docker containers of the same name
* Rebuild the Docker image
* Run the container interactively


## 📝 **Manual Build (Without Docker)**

If you prefer to build manually:

```bash
./scripts/setup_dependencies.sh
./scripts/build.sh
```

### **Run the Examples:**

```bash
./scripts/run.sh
```

## 📂 **Project Structure**

```
├── CMakeLists.txt             # CMake build configuration
├── Dockerfile                 # Docker setup
├── include                    # Header files
│   └── MultiAgentSolver
├── src                        # Source files
├── examples                   # Example applications
├── scripts                    # Helper scripts
│   ├── build.sh
│   ├── run.sh
│   └── setup_dependencies.sh
├── cmake                      # CMake configuration files
│   └── MultiAgentSolverConfig.cmake.in
├── build                      # Build artifacts
└── README.md                  # Project documentation
```



## Results

Times and costs for different methods in the example code:
```
🚗 Single-Track Lane Following Test 🚗
---------------------------------------------
Solver              Cost           Time (ms)
     ---------------------------------------------
CGD                 24.0465        20.6444        
OSQP                30.1889        2.33275        
OSQP Collocation    23.9809        5.11993        
iLQR                24.4039        1.06887 
```

```

Multi-Agent Single Track Test

Method                                  Cost           Time (ms)      
----------------------------------------------------------------------
Centralized CGD                         7928.151       1214.919       
Centralized iLQR                        7928.501       135.472        
Centralized OSQP                        7929.011       285.711        
Centralized OSQP-collocation            7929.392       1071.582       
Nash Sequential CGD                     7928.153       26.612         
Nash Sequential iLQR                    7928.327       11.053         
Nash Sequential OSQP                    7928.384       38.514         
Nash Sequential OSQP-collocation        7928.158       2098.299       
Nash LineSearch CGD                     7928.153       27.597         
Nash LineSearch iLQR                    7928.327       13.807         
Nash LineSearch OSQP                    7928.384       40.643         
Nash LineSearch OSQP-collocation        7928.152       2010.733       
Nash TrustRegion CGD                    7928.153       34.767         
Nash TrustRegion iLQR                   7928.199       14.093         
Nash TrustRegion OSQP                   7928.417       46.087         
Nash TrustRegion OSQP-collocation       7928.152       1596.460 
```

## License
Apache 2.0

## Project status
WIP  - still earlyl in development but working.
