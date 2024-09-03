//
//  Utilities.metal
//  GeodesicGaze
//
//  Created by Trevor Gravely on 9/2/24.
//


#include <metal_stdlib>
#include "Utilities.h"

using namespace metal;

float3 yuvToRgb(float y, float u, float v) {
    float3 rgb;
    rgb.r = y + 1.402 * (v - 0.5);
    rgb.g = y - 0.344136 * (u - 0.5) - 0.714136 * (v - 0.5);
    rgb.b = y + 1.722 * (u - 0.5);
    return rgb;
}
