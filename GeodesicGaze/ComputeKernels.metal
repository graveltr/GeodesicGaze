//
//  ComputeKernels.metal
//  GeodesicGaze
//
//  Created by Trevor Gravely on 9/1/24.
//

#include <metal_stdlib>
#include "MathFunctions.h"
#include "Physics.h"

using namespace metal;

kernel void ellint_F_compute_kernel(const device float *angles [[buffer(0)]],
                                    const device float *moduli [[buffer(1)]],
                                    device EllintResult *results [[buffer(2)]],
                                    uint id [[thread_position_in_grid]]) {
    results[id] = ellint_F(angles[id], sqrt(moduli[id]), 1e-5, 1e-5);
}

kernel void radial_roots_compute_kernel(const device float *M [[buffer(0)]],
                                        const device float *b [[buffer(1)]],
                                        device float4 *results [[buffer(2)]],
                                        uint id [[thread_position_in_grid]]) {
    results[id] = radialRoots(M[id], b[id]);
}

kernel void phiS_compute_kernel(const device float *M [[buffer(0)]],
                                const device float *ro [[buffer(1)]],
                                const device float *rs [[buffer(2)]],
                                const device float *b [[buffer(3)]],
                                device PhiSResult *results [[buffer(4)]],
                                uint id [[thread_position_in_grid]]) {
    results[id] = phiS(M[id], ro[id], rs[id], b[id]);
}

kernel void normalize_angle_compute_kernel(const device float *phi [[buffer(0)]],
                                           device float *results [[buffer(1)]],
                                           uint id [[thread_position_in_grid]]) {
    results[id] = normalizeAngle(phi[id]);
}

kernel void schwarzschild_lense_compute_kernel(const device float *M [[buffer(0)]],
                                               const device float *ro [[buffer(1)]],
                                               const device float *rs [[buffer(2)]],
                                               const device float *varphi [[buffer(3)]],
                                               device SchwarzschildLenseResult *results [[buffer(4)]],
                                               uint id [[thread_position_in_grid]]) {
    results[id] = schwarzschildLense(M[id], ro[id], rs[id], varphi[id]);
}
