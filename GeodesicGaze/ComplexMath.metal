//
//  ComplexMath.metal
//  GeodesicGaze
//
//  Created by Trevor Gravely on 9/18/24.
//

#include <metal_stdlib>
using namespace metal;
#include "MathFunctions.h"

bool isReal(float2 z) {
    return fEqual(z.y, 0);
}

float2 zWithMaxRealPart(float2 z1, float2 z2, float2 z3) {
    float2 max = z1;
    
    if (z2.x > max.x) {
        max = z2;
    }
    if (z3.x > max.x) {
        max = z3;
    }
    
    return max;
}

float2 cartToPolar(float2 z) {
    return float2(sqrt(z.x*z.x + z.y*z.y), atan2(z.y, z.x));
}

float2 polarToCart(float2 rtheta) {
    return float2(rtheta[0] * cos(rtheta[1]), rtheta[0] * sin(rtheta[1]));
}

float2 pow1over3(float2 z) {
    float q = 1.0 / 3.0;
    
    float2 rtheta = cartToPolar(z);
    
    float rtilde = pow(rtheta[0], q);
    
    float2 z1 = polarToCart(float2(rtilde, q * rtheta[1]));
    float2 z2 = polarToCart(float2(rtilde, q * (rtheta[1] + 2.0 * M_PI_F)));
    float2 z3 = polarToCart(float2(rtilde, q * (rtheta[1] + 4.0 * M_PI_F)));
    
    return zWithMaxRealPart(z1, z2, z3);
}
