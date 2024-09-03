//
//  Shaders.metal
//  MultiCamDemo
//
//  Created by Trevor Gravely on 7/16/24.
//

#include <metal_stdlib>

#include "Utilities.h"
#include "MathFunctions.h"
#include "Physics.h"

using namespace metal;

struct VertexOut {
    float4 position [[position]];
    float2 texCoord;
};

struct Uniforms {
    int frontTextureWidth;
    int frontTextureHeight;
    int backTextureWidth;
    int backTextureHeight;
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

float3 sampleYUVTexture(texture2d<float, access::sample> YTexture,
                        texture2d<float, access::sample> UVTexture,
                        float2 texCoord) {
    // The sampler to be used for obtaining pixel colors
    constexpr sampler textureSampler(coord::normalized, address::clamp_to_edge, filter::linear);
    
    float y = YTexture.sample(textureSampler, texCoord).r;
    float2 uv = UVTexture.sample(textureSampler, texCoord).rg;
    
    return yuvToRgb(y, uv.x, uv.y);
}

// TODO: Make sure it does the math properly for either orientation
// TODO: Ensure consistency of camel case and underscores
// TODO: Add front facing camera logic -> don't need to do multiple ray traces, just need to
fragment float4 fragmentShader(VertexOut in [[stage_in]],
                               texture2d<float, access::sample> frontYTexture [[texture(0)]],
                               texture2d<float, access::sample> frontUVTexture [[texture(1)]],
                               texture2d<float, access::sample> backYTexture [[texture(2)]],
                               texture2d<float, access::sample> backUVTexture [[texture(3)]],
                               constant Uniforms &uniforms [[buffer(0)]],
                               sampler s [[sampler(0)]]) {
    // The focal length of the camera in pixels
    // TODO: should be one for each camera!
    float focalLength = 3825.0;
    
    // Some other parameters.
    // TODO: set these parameters via the UI
    float M = 1.0;
    float rs = 1000.0;
    float ro = 30.0;
    
    // TODO: parse what follows into a function that can be called on both front and back texture
    
    // Calculate the pixel coordinates of the current fragment
    float2 pixelCoords = in.texCoord * float2(uniforms.backTextureWidth, uniforms.backTextureHeight);
    
    // Calculate the pixel coordinates of the center of the image
    float2 center = float2(uniforms.backTextureWidth / 2.0, uniforms.backTextureHeight / 2.0);
    
    // Place the center at the origin
    float2 relativePixelCoords = pixelCoords - center;
    
    // Compute the distance the fragment sits away from the center in pixels.
    float dist = distance(pixelCoords, center);
    
    // Compute the angle of incidence and the rotation in the image plane.
    float varphi = atan2(dist, focalLength);
    float psi = atan2(relativePixelCoords.y, relativePixelCoords.x);
    
    SchwarzschildLenseResult lenseResult = schwarzschildLense(M, ro, rs, varphi);
    if (lenseResult.status == FAILURE) {
        // color errored pixels solid red
        return float4(1.0, 0.0, 0.0, 1.0);
    } else if (lenseResult.status == EMITTED_FROM_BLACK_HOLE) {
        return float4(0.0, 0.0, 0.0, 1.0);
    }
    float varphitilde = lenseResult.varphitilde;
    bool ccw = lenseResult.ccw;
    
    // Using the transformed incidence angle and focal length
    // obtain the distance in pixels from the center.
    float transformedDist = 0.0;
    float2 transformedRelativePixelCoordinates = 0.0;
    bool isFrontFacing = false;
    if (fEqual(varphitilde, M_PI_F)) {
        isFrontFacing = true;
        return float4(0.0, 0.0, 0.0, 1.0);
    } else if (fEqual(varphitilde, 0.0)) {
        isFrontFacing = false;
        transformedDist = 0.0;
    } else if (M_PI_F / 2.0 < varphitilde) {
        // If the rotation from the line of sight is greater
        // than pi / 2 rad then the incidence corresponds to
        // the front facing camera.
        // TODO: implement this
        isFrontFacing = true;
        return float4(0.0, 1.0, 0.0, 1.0);
    } else {
        isFrontFacing = false;
        transformedDist = focalLength * tan(varphitilde);
        transformedRelativePixelCoordinates = (ccw ? -1.0 : 1.0) * float2(transformedDist * cos(psi),
                                                                          transformedDist * sin(psi));
    }
    
    // Invert back to texture coordinates
    float2 transformedPixelCoords = transformedRelativePixelCoordinates + center;
    float2 transformedTexCoord = transformedPixelCoords / float2(uniforms.backTextureWidth, uniforms.backTextureHeight);
    
    // Ensure that the texture coordinate is inbounds
    if (transformedTexCoord.x < 0.0 || 1.0 < transformedTexCoord.x ||
        transformedTexCoord.y < 0.0 || 1.0 < transformedTexCoord.y) {
        return float4(0.0, 0.0, 1.0, 1.0);
    }
    
    float3 rgb = float3(0.0, 0.0, 0.0);
    if (isFrontFacing) {
        rgb = sampleYUVTexture(frontYTexture, frontUVTexture, transformedTexCoord);
    } else {
        rgb = sampleYUVTexture(backYTexture, backUVTexture, transformedTexCoord);
    }
    
    return float4(rgb, 1.0);
}
