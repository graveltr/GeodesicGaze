//
//  ComputeKernels.metal
//  GeodesicGaze
//
//  Created by Trevor Gravely on 9/1/24.
//

#include <metal_stdlib>
#include "MathFunctions.h"

using namespace metal;

kernel void ellint_F_compute_kernel(const device float *angles [[buffer(0)]],
                                    const device float *moduli [[buffer(1)]],
                                    device EllintResult *results [[buffer(2)]],
                                    uint id [[thread_position_in_grid]]) {
    
    float phi = angles[id];
    float k = moduli[id];
    float errtol = 1e-5;
    float prec = 1e-5;

    EllintResult result = ellint_F(phi, k, errtol, prec);
    results[id] = result;
}
