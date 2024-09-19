//
//  ComputeKernels.metal
//  GeodesicGaze
//
//  Created by Trevor Gravely on 9/1/24.
//

#include <metal_stdlib>
#include "MathFunctions.h"
#include "Physics.h"
#include "ComplexMath.h"

using namespace metal;

struct KerrRadialRootsResultForSwift {
    float2 r1;
    float2 r2;
    float2 r3;
    float2 r4;
    int status;
};

kernel void ellint_F_compute_kernel(const device float *angles [[buffer(0)]],
                                    const device float *moduli [[buffer(1)]],
                                    device EllintResult *results [[buffer(2)]],
                                    uint id [[thread_position_in_grid]]) {
    results[id] = ellint_F(angles[id], sqrt(moduli[id]), 1e-5, 1e-5);
}

kernel void ellint_E_compute_kernel(const device float *angles [[buffer(0)]],
                                    const device float *moduli [[buffer(1)]],
                                    device EllintResult *results [[buffer(2)]],
                                    uint id [[thread_position_in_grid]]) {
    results[id] = ellint_E(angles[id], sqrt(moduli[id]), 1e-5, 1e-5);
}

kernel void ellint_P_compute_kernel(const device float *angles [[buffer(0)]],
                                    const device float *moduli [[buffer(1)]],
                                    const device float *n [[buffer(2)]],
                                    device EllintResult *results [[buffer(3)]],
                                    uint id [[thread_position_in_grid]]) {
    results[id] = ellint_P(angles[id], sqrt(moduli[id]), -1 * n[id], 1e-5, 1e-5);
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
                                               const device float *b [[buffer(3)]],
                                               device SchwarzschildLenseResult *results [[buffer(4)]],
                                               uint id [[thread_position_in_grid]]) {
    results[id] = schwarzschildLense(M[id], ro[id], rs[id], b[id]);
}

kernel void kerr_radial_roots_compute_kernel(const device float *a [[buffer(0)]],
                                             const device float *M [[buffer(1)]],
                                             const device float *eta [[buffer(2)]],
                                             const device float *lambda [[buffer(3)]],
                                             device KerrRadialRootsResultForSwift *results [[buffer(4)]],
                                             uint id [[thread_position_in_grid]]) {
    KerrRadialRootsResultForSwift result;
    
    KerrRadialRootsResult packedResult = kerrRadialRoots(a[id], M[id], eta[id], lambda[id]);
    
    result.r1 = packedResult.roots[0];
    result.r2 = packedResult.roots[1];
    result.r3 = packedResult.roots[2];
    result.r4 = packedResult.roots[3];
    
    result.status = packedResult.status;

    results[id] = result;
}

kernel void compute_abc_compute_kernel(const device float *a                  [[buffer(0)]],
                                             const device float *M                  [[buffer(1)]],
                                             const device float *eta                [[buffer(2)]],
                                             const device float *lambda             [[buffer(3)]],
                                             device float3 *results                 [[buffer(4)]],
                                             uint id [[thread_position_in_grid]]) {
    results[id] = computeABC(a[id], M[id], eta[id], lambda[id]);
}

kernel void compute_pq_compute_kernel(const device float *a                  [[buffer(0)]],
                                             const device float *M                  [[buffer(1)]],
                                             const device float *eta                [[buffer(2)]],
                                             const device float *lambda             [[buffer(3)]],
                                             device float2 *results                 [[buffer(4)]],
                                             uint id [[thread_position_in_grid]]) {
    results[id] = computePQ(a[id], M[id], eta[id], lambda[id]);
}

kernel void pow1over3_compute_kernel(const device float *zx [[buffer(0)]],
                                     const device float *zy [[buffer(1)]],
                                     device float2 *results [[buffer(2)]],
                                     uint id [[thread_position_in_grid]]) {
    results[id] = pow1over3(float2(zx[id], zy[id]));
}
