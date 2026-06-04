// Physics simulation kernels: spring forces + Euler integration.

use core::arch::nvptx;

/// Compute spring forces between connected particles.
///
/// Each thread processes one spring. Uses atomic add for force accumulation
/// since multiple springs contribute to the same particle.
///
/// Grid: (ceil(n_springs / 256), 1, 1), Block: (256, 1, 1).
#[no_mangle]
pub unsafe extern "gpu-kernel" fn spring_forces(
    pos: *const f32,
    forces: *mut f32,
    spring_a: *const u32,
    spring_b: *const u32,
    n_springs: u32,
    spring_k: f32,
    rest_length: f32,
    status: *mut u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;

    #[cfg(target_arch = "nvptx64")]
    {
        let block_x = nvptx::_block_idx_x() as u32;
        let global_id = block_x * 256 + tid;

        if global_id < n_springs {
            let a = *spring_a.add(global_id as usize) as usize;
            let b = *spring_b.add(global_id as usize) as usize;

            let dx = *pos.add(b * 2) - *pos.add(a * 2);
            let dy = *pos.add(b * 2 + 1) - *pos.add(a * 2 + 1);

            // Distance
            let dist_sq = dx * dx + dy * dy;
            let dist: f32;
            core::arch::asm!(
                "sqrt.approx.f32 {out}, {in_};",
                out = out(reg32) dist,
                in_ = in(reg32) dist_sq,
            );

            if dist > 1e-8 {
                let stretch = dist - rest_length;
                let f = spring_k * stretch / dist;
                let fx = f * dx;
                let fy = f * dy;

                // Atomic add to forces (multiple springs write to same particle)
                core::arch::asm!(
                    "atom.global.add.f32 {tmp}, [{addr}], {val};",
                    tmp = out(reg32) _,
                    addr = in(reg64) forces.add(a * 2),
                    val = in(reg32) fx,
                );
                core::arch::asm!(
                    "atom.global.add.f32 {tmp}, [{addr}], {val};",
                    tmp = out(reg32) _,
                    addr = in(reg64) forces.add(a * 2 + 1),
                    val = in(reg32) fy,
                );
                let neg_fx = -fx;
                let neg_fy = -fy;
                core::arch::asm!(
                    "atom.global.add.f32 {tmp}, [{addr}], {val};",
                    tmp = out(reg32) _,
                    addr = in(reg64) forces.add(b * 2),
                    val = in(reg32) neg_fx,
                );
                core::arch::asm!(
                    "atom.global.add.f32 {tmp}, [{addr}], {val};",
                    tmp = out(reg32) _,
                    addr = in(reg64) forces.add(b * 2 + 1),
                    val = in(reg32) neg_fy,
                );
            }
        }
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (pos, forces, spring_a, spring_b, n_springs, spring_k, rest_length);
    }

    if tid == 0 {
        *status = 0;
    }
}

/// Compute pairwise gravitational forces (O(N²)).
///
/// Each thread computes the total force on one particle from all others.
/// F_i = sum_j(-G * m_i * m_j / r_ij² * r_hat_ij) for j != i.
///
/// Grid: (ceil(n / 256), 1, 1), Block: (256, 1, 1).
#[no_mangle]
pub unsafe extern "gpu-kernel" fn gravity_forces(
    pos: *const f32,
    forces: *mut f32,
    mass: *const f32,
    n: u32,
    gravity_g: f32,
    softening: f32,
    status: *mut u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;

    #[cfg(target_arch = "nvptx64")]
    {
        let block_x = nvptx::_block_idx_x() as u32;
        let i = block_x * 256 + tid;

        if i < n {
            let px_i = *pos.add((i * 2) as usize);
            let py_i = *pos.add((i * 2 + 1) as usize);
            let m_i = *mass.add(i as usize);
            let mut fx = 0.0f32;
            let mut fy = 0.0f32;

            for j in 0..n {
                if j == i {
                    continue;
                }
                let px_j = *pos.add((j * 2) as usize);
                let py_j = *pos.add((j * 2 + 1) as usize);
                let m_j = *mass.add(j as usize);

                let dx = px_j - px_i;
                let dy = py_j - py_i;
                let dist_sq = dx * dx + dy * dy + softening;
                let dist: f32;
                core::arch::asm!(
                    "sqrt.approx.f32 {out}, {in_};",
                    out = out(reg32) dist,
                    in_ = in(reg32) dist_sq,
                );
                let inv_dist3 = 1.0 / (dist * dist_sq);
                let f = gravity_g * m_i * m_j * inv_dist3;
                fx += f * dx;
                fy += f * dy;
            }

            *forces.add((i * 2) as usize) = fx;
            *forces.add((i * 2 + 1) as usize) = fy;
        }
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (pos, forces, mass, n, gravity_g, softening);
    }

    if tid == 0 {
        *status = 0;
    }
}

/// Euler integration step: update positions and velocities.
///
/// vel[i] = vel[i] * (1 - damping*dt) + forces[i/2] * dt / mass[i/2]
/// pos[i] += vel[i] * dt
///
/// Grid: (ceil(n*2 / 256), 1, 1), Block: (256, 1, 1).
/// Each thread updates one component (x or y) of one particle.
#[no_mangle]
pub unsafe extern "gpu-kernel" fn euler_step(
    pos: *mut f32,
    vel: *mut f32,
    forces: *const f32,
    mass: *const f32,
    n: u32,
    dt: f32,
    damping: f32,
    status: *mut u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;

    #[cfg(target_arch = "nvptx64")]
    {
        let block_x = nvptx::_block_idx_x() as u32;
        let global_id = block_x * 256 + tid;
        let total = n * 2;

        if global_id < total {
            let particle = global_id / 2;
            let m = *mass.add(particle as usize);
            let f = *forces.add(global_id as usize);

            let v = *vel.add(global_id as usize);
            let new_v = v * (1.0 - damping * dt) + f * dt / m;
            *vel.add(global_id as usize) = new_v;

            let p = *pos.add(global_id as usize);
            *pos.add(global_id as usize) = p + new_v * dt;
        }
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (pos, vel, forces, mass, n, dt, damping);
    }

    if tid == 0 {
        *status = 0;
    }
}
