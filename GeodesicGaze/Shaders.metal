//
//  Shaders.metal
//  MultiCamDemo
//
//  Created by Trevor Gravely on 7/16/24.
//

#include <metal_stdlib>
using namespace metal;

struct VertexOut {
    float4 position [[position]];
    float2 texCoord;
};

vertex VertexOut vertexShader(uint vertexID [[vertex_id]]) {
    float4 positions[4] = {
        float4(-1.0,  1.0, 0.0, 1.0),
        float4(-1.0, -1.0, 0.0, 1.0),
        float4( 1.0,  1.0, 0.0, 1.0),
        float4( 1.0, -1.0, 0.0, 1.0)
    };
    
    float2 texCoords[4] = {
        float2(0.0, 0.0),
        float2(1.0, 0.0),
        float2(0.0, 1.0),
        float2(1.0, 1.0)
    };
    
    VertexOut out;
    out.position = positions[vertexID];
    out.texCoord = texCoords[vertexID];
    return out;
}

float3 yuvToRgb(float y, float u, float v) {
    float3 rgb;
    rgb.r = y + 1.402 * (v - 0.5);
    rgb.g = y - 0.344136 * (u - 0.5) - 0.714136 * (v - 0.5);
    rgb.b = y + 1.722 * (u - 0.5);
    return rgb;
}

fragment float4 fragmentShader(VertexOut in [[stage_in]],
                               texture2d<float, access::sample> frontYTexture [[texture(0)]],
                               texture2d<float, access::sample> frontUVTexture [[texture(1)]],
                               texture2d<float, access::sample> backYTexture [[texture(2)]],
                               texture2d<float, access::sample> backUVTexture [[texture(3)]],
                               sampler s [[sampler(0)]]) {
    constexpr sampler textureSampler(coord::normalized, address::clamp_to_edge, filter::linear);
    
    float yFront = frontYTexture.sample(textureSampler, in.texCoord).r;
    float2 uvFront = frontUVTexture.sample(textureSampler, in.texCoord).rg;
    float3 rgbFront = yuvToRgb(yFront, uvFront.x, uvFront.y);
    
    float yBack = backYTexture.sample(textureSampler, in.texCoord).r;
    float2 uvBack = backUVTexture.sample(textureSampler, in.texCoord).rg;
    float3 rgbBack = yuvToRgb(yBack, uvBack.x, uvBack.y);
    
    float3 mixedRgb = (rgbFront + rgbBack) / 2.0;

    return float4(mixedRgb, 1.0);
}
