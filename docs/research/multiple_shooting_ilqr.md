# Multiple-Shooting iLQR Research Notes

## Overview
Multiple shooting augments the classical single-shooting iLQR (iterative Linear Quadratic Regulator) by optimizing over both control variables and intermediate state nodes. The method breaks the horizon into segments, introduces continuity constraints ("defects") between segments, and solves the resulting structured constrained problem to improve numerical robustness and parallelism compared to single shooting.【F:docs/research/multiple_shooting_ilqr.md†L4-L7】

## Key References
- **Giftthaler et al., 2018** – Introduces Multiple-Shooting DDP (MSDDP) for robotics, showing how to add state decision variables and enforce defects with Lagrange multipliers while retaining quadratic subproblems.【F:docs/research/multiple_shooting_ilqr.md†L9-L12】
- **Verscheure et al., 2009** – Discusses direct multiple shooting transcription for optimal control and benefits for handling constraints and stiff dynamics.【F:docs/research/multiple_shooting_ilqr.md†L13-L15】
- **Freeman et al., 2021** – Explores combining iLQR-style backward passes with multiple shooting in Model Predictive Path Integral control, highlighting improvements in warm starting and parallelization.【F:docs/research/multiple_shooting_ilqr.md†L16-L18】

## Algorithmic Structure
1. **Segment the horizon:** Partition the time horizon into `M` shooting intervals with decision variables for the initial state of each segment `x_k` and control sequences `u_k` over local grids.【F:docs/research/multiple_shooting_ilqr.md†L20-L23】
2. **Introduce defect constraints:** Enforce `x_{k+1} - f(x_k, u_k) = 0` at the segment boundaries. In practice, relax these constraints via Lagrange multipliers and quadratic penalties to keep the backward pass linear.【F:docs/research/multiple_shooting_ilqr.md†L24-L27】
3. **Linearize dynamics per segment:** For each segment, linearize dynamics around nominal trajectories to obtain local matrices `A_k`, `B_k`, and evaluate cost expansions, mirroring classical iLQR but per segment.【F:docs/research/multiple_shooting_ilqr.md†L28-L31】
4. **Form block-sparse KKT system:** The quadratic approximation yields a block-tridiagonal system in `[δx_k, δu_k, λ_k]` with multipliers `λ_k` enforcing defects. Solve using Riccati-like sweeps that incorporate multiplier updates (forward) and state-control gains (backward).【F:docs/research/multiple_shooting_ilqr.md†L32-L36】
5. **Compute policy updates:** Extract feedback and feedforward gains `K_k`, `k_k` that respect defect constraints. Update states and controls via line search with defect correction (project defects or integrate segments forward).【F:docs/research/multiple_shooting_ilqr.md†L37-L40】
6. **Dual updates:** Update multipliers using augmented Lagrangian or projected gradient steps to reduce defect residuals, analogous to SQP multiple shooting schemes.【F:docs/research/multiple_shooting_ilqr.md†L41-L43】

## Implementation Recommendations
- **State decision variables:** Maintain explicit state nodes `x_k` per segment with bounds/constraints, enabling integration resets and contact handling.【F:docs/research/multiple_shooting_ilqr.md†L45-L47】
- **Defect handling:** Use augmented Lagrangian with penalty parameter adaptation to stabilize convergence; embed penalty terms in the cost expansion to modify Hessians locally.【F:docs/research/multiple_shooting_ilqr.md†L48-L51】
- **Linear algebra:** Exploit block structure (e.g., Schur complement or block LDLᵀ) for efficient backward passes; parallelize segment linearizations and forward simulations.【F:docs/research/multiple_shooting_ilqr.md†L52-L55】
- **Initialization:** Warm-start with single-shooting iLQR trajectory to provide consistent state-control sequences, then introduce defects gradually (continuation on penalty).【F:docs/research/multiple_shooting_ilqr.md†L56-L59】
- **Constraints & contacts:** Multiple shooting naturally allows inclusion of state/control inequality constraints at nodes; use projected gradient or quadratic programming (box-constrained) per stage.【F:docs/research/multiple_shooting_ilqr.md†L60-L63】
- **MPC deployment:** For receding-horizon use, cache factorizations and reuse multipliers across iterations to accelerate warm starts; apply partial condensing when communication between segments is expensive.【F:docs/research/multiple_shooting_ilqr.md†L64-L67】

## Practical Considerations
- **Segment length:** Balance integration accuracy vs. system size; shorter segments improve robustness for stiff dynamics but increase KKT dimension. Adaptive segmentation can target contact transitions.【F:docs/research/multiple_shooting_ilqr.md†L69-L72】
- **Convergence criteria:** Monitor cost decrease, defect norms, and multiplier updates. Apply merit functions that combine cost and defect reduction to drive line-search acceptance.【F:docs/research/multiple_shooting_ilqr.md†L73-L76】
- **Software architecture:** Abstract segment operations (linearization, cost expansion) to reuse existing single-shooting iLQR components. Introduce interfaces for defect assembly and block-sparse solves to minimize refactoring.【F:docs/research/multiple_shooting_ilqr.md†L77-L80】

## Summary
By embedding multiple shooting into iLQR, you trade additional state decision variables and constraint handling for better numerical conditioning, easier handling of constraints, and potential parallelism. The approach mirrors SQP multiple shooting while retaining iLQR's efficient quadratic approximations through specialized Riccati-like solvers for the block-structured KKT system.【F:docs/research/multiple_shooting_ilqr.md†L82-L85】
