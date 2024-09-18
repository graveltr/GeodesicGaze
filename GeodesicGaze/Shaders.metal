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
        float4( 1.0,  1.0, 0.0, 1.0),
        float4(-1.0, -1.0, 0.0, 1.0),
        float4( 1.0, -1.0, 0.0, 1.0)
    };
    
    float2 texCoords[4] = {
        float2(0.0, 1.0),
        float2(0.0, 0.0),
        float2(1.0, 1.0),
        float2(1.0, 0.0)
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

fragment float4 fragmentShader(VertexOut in [[stage_in]],
                               texture2d<float, access::sample> frontYTexture [[texture(0)]],
                               texture2d<float, access::sample> frontUVTexture [[texture(1)]],
                               texture2d<float, access::sample> backYTexture [[texture(2)]],
                               texture2d<float, access::sample> backUVTexture [[texture(3)]],
                               constant Uniforms &uniforms [[buffer(0)]],
                               sampler s [[sampler(0)]]) {
    // Estimate of the focal length in units of pixels.
    // We assume that the front-facing and rear-facing
    // cameras have the same focal length.
    float focalLength = 3825.0;

    // Some other parameters.
    // TODO: set these parameters via the UI
    float M = 0.1;
    float rs = 100.0;
    float ro = 30.0;
    
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
    
    /*
    * In flat space, geodesics traverse straight lines. From trig,
    * it is obvious that ro sin(varphi) is the distance of closest
    * approach for a light ray launched from ro with incidence varphi.
    * The distance of closest approach, by definition is a radial turning
    * point, so it must lie in the set of roots of R(r).
    * Further, the roots of the radial potential are +- lambda, from
    * which we deduce that b = | lambda | = ro sin(varphi).
    */
    
    float b = ro * sin(varphi);

    SchwarzschildLenseResult lenseResult = schwarzschildLense(M, ro, rs, b);
    if (lenseResult.status == FAILURE) {
        // color errored pixels solid red
        return float4(1.0, 0.0, 0.0, 1.0);
    } else if (lenseResult.status == EMITTED_FROM_BLACK_HOLE) {
        return float4(0.0, 0.0, 0.0, 1.0);
    }
    float varphitilde = lenseResult.varphitilde;
    bool ccw = lenseResult.ccw;
    
    // Need to ensure that we don't call tan(pi / 2).
    if (fEqual(varphitilde, M_PI_F / 2.0)) {
        return float4(0.0, 0.0, 1.0, 1.0);
    }
    
    // Using the transformed incidence angle and focal length
    // obtain the distance in pixels from the center.
    bool isFrontFacing = false;
    if (M_PI_F / 2.0 < varphitilde) {
        isFrontFacing = true;
        varphitilde = M_PI_F - varphitilde;
    }
    float transformedDist = focalLength * tan(varphitilde);
    float2 transformedRelativePixelCoords = (ccw ? -1.0 : 1.0) * float2(transformedDist * cos(psi),
                                                                 transformedDist * sin(psi));
    
    // We also need the front-facing camera center now.
    float2 frontFacingCenter = float2(uniforms.frontTextureWidth / 2.0, uniforms.frontTextureHeight / 2.0);

    // Invert back to texture coordinates
    float2 transformedCenter = isFrontFacing ? frontFacingCenter : center;
    float2 imageDimensions = isFrontFacing ? float2(uniforms.frontTextureWidth, uniforms.frontTextureHeight)
                                           : float2(uniforms.backTextureWidth, uniforms.backTextureHeight);
    
    float2 transformedPixelCoords = transformedRelativePixelCoords + transformedCenter;
    float2 transformedTexCoord = transformedPixelCoords / imageDimensions;
    
    // Ensure that the texture coordinate is inbounds
    if (transformedTexCoord.x < 0.0 || 1.0 < transformedTexCoord.x ||
        transformedTexCoord.y < 0.0 || 1.0 < transformedTexCoord.y) {
        return float4(0.0, 0.0, 1.0, 1.0);
    }
    
    float3 rgb = isFrontFacing  ? sampleYUVTexture(frontYTexture, frontUVTexture, transformedTexCoord)
                                : sampleYUVTexture(backYTexture, backUVTexture, transformedTexCoord);
    
    return float4(rgb, 1.0);
}

fragment float4 dominicFragmentShader(VertexOut in [[stage_in]],
                               texture2d<float, access::sample> frontYTexture [[texture(0)]],
                               texture2d<float, access::sample> frontUVTexture [[texture(1)]],
                               texture2d<float, access::sample> backYTexture [[texture(2)]],
                               texture2d<float, access::sample> backUVTexture [[texture(3)]],
                               constant Uniforms &uniforms [[buffer(0)]],
                               sampler s [[sampler(0)]]) {
    // We let rs and ro be large in this set up.
    // This will allow for the usage of an approximation to the
    // elliptic integrals during lensing.
    float M = 1.0;
    float rs = 1000.0;
    float ro = rs;
    
    // Calculate the pixel coordinates of the current fragment
    float2 pixelCoords = in.texCoord * float2(uniforms.backTextureWidth, uniforms.backTextureHeight);
    
    // Calculate the pixel coordinates of the center of the image
    float2 center = float2(uniforms.backTextureWidth / 2.0, uniforms.backTextureHeight / 2.0);
    
    // Place the center at the origin
    float2 relativePixelCoords = pixelCoords - center;
    
    // Convert the pixel coordinates to coordinates in the image plane
    float lengthPerPixel = 0.1;
    float2 imagePlaneCoords = lengthPerPixel * relativePixelCoords;
   
    // Obtain the polar coordinates of this image plane location
    float b = length(imagePlaneCoords);
    float psi = atan2(imagePlaneCoords.y, imagePlaneCoords.x);

    SchwarzschildLenseResult lenseResult = schwarzschildLense(M, ro, rs, b);
    if (lenseResult.status == FAILURE) {
        // Color errored pixels solid red
        return float4(1.0, 0.0, 0.0, 1.0);
    } else if (lenseResult.status == EMITTED_FROM_BLACK_HOLE) {
        return float4(0.0, 0.0, 0.0, 1.0);
    }
    float varphitilde = lenseResult.varphitilde;
    bool ccw = lenseResult.ccw;

    // Unwind through the inverse transformation to texture coordinates.
    // Note that because ro = rs, we don't need to worry about the front-facing
    // camera.
    float btilde = ro * sin(varphitilde);
    float2 transformedImagePlaneCoords = (ccw ? -1.0 : 1.0) * float2(btilde * cos(psi), btilde * sin(psi));
    float2 transformedRelativePixelCoords = transformedImagePlaneCoords / lengthPerPixel;
    float2 transformedPixelCoords = transformedRelativePixelCoords + center;
    float2 transformedTexCoord = transformedPixelCoords / float2(uniforms.backTextureWidth, uniforms.backTextureHeight);

    // Ensure that the texture coordinate is inbounds
    if (transformedTexCoord.x < 0.0 || 1.0 < transformedTexCoord.x ||
        transformedTexCoord.y < 0.0 || 1.0 < transformedTexCoord.y) {
        return float4(0.0, 0.0, 1.0, 1.0);
    }
    
    float3 rgb = sampleYUVTexture(backYTexture, backUVTexture, transformedTexCoord);
    return float4(rgb, 1.0);
}
