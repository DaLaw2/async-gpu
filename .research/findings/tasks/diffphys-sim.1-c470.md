# diffphys-sim.1: Differentiable simulation design
**Cycle**: 470 | **Theme**: diffphys-sim | **Kind**: investigation | **Status**: done

## Summary
Designed a 2D N-body spring-mass simulation. GPU kernel for pairwise force computation
+ Euler integration. Manual analytical backward through timesteps (chain rule on Euler updates).
Optimization finds initial velocities that hit target positions.

## Findings
### Q: How to make physics simulation differentiable?
A: Euler integration creates a computation graph: pos_{t+1} = pos_t + vel_t * dt,
vel_{t+1} = vel_t + F(pos_t) * dt / mass. Backward: reverse chain rule through steps.
dL/d(pos_0) accumulates gradients from all timesteps.
**Confidence**: high

## Design Decision
- **Spring-mass system** (simpler than gravity, numerically stable): F = -k * (r - rest_length) * r_hat
- **GPU forward**: custom kernel for pairwise spring forces, elementwise Euler updates
- **CPU backward**: analytical gradient through Euler steps (no autograd tape needed)
- **Optimization**: gradient descent on initial velocities to reach target final positions

## Impact on Downstream Tasks
- diffphys-sim.2: implement GPU kernel + forward
- diffphys-sim.3: backward + optimization
