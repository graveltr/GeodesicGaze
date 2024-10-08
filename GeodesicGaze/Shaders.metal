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

// TODO: fix status code overlaps with EMITTED_FROM_BLACK_HOLE

#define SUCCESS_BACK_TEXTURE 5
#define SUCCESS_FRONT_TEXTURE 4
#define OUTSIDE_FOV 1
#define ERROR 2
#define VORTICAL 10

#define FULL_FOV_MODE 0
#define ACTUAL_FOV_MODE 1


struct LenseTextureCoordinateResult {
    float2 coord;
    int status;
};

struct VertexOut {
    float4 position [[position]];
    float2 texCoord;
};

struct Uniforms {
    int frontTextureWidth;
    int frontTextureHeight;
    int backTextureWidth;
    int backTextureHeight;
    int mode;
};

struct PreComputeUniforms {
    int mode;
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

fragment float4 screenProjectionFragmentShader(VertexOut in [[stage_in]],
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

fragment float4 kerrFragmentShader(VertexOut in [[stage_in]],
                                   texture2d<float, access::sample> frontYTexture [[texture(0)]],
                                   texture2d<float, access::sample> frontUVTexture [[texture(1)]],
                                   texture2d<float, access::sample> backYTexture [[texture(2)]],
                                   texture2d<float, access::sample> backUVTexture [[texture(3)]],
                                   constant Uniforms &uniforms [[buffer(0)]],
                                   sampler s [[sampler(0)]]) {
    /*
     * The convention we use is to call the camera screen the "source" since we
     * ray trace from this location back into the geometry.
     */
    float M = 1.0;
    float a = 0.99;
    float thetas = M_PI_F / 2.0;
    float rs = 1000.0;
    float ro = rs;
    
    // Calculate the pixel coordinates of the current fragment
    float2 pixelCoords = in.texCoord * float2(uniforms.backTextureWidth, uniforms.backTextureHeight);
    
    // Calculate the pixel coordinates of the center of the image
    float2 center = float2(uniforms.backTextureWidth / 2.0, uniforms.backTextureHeight / 2.0);
    
    // Place the center at the origin
    float2 relativePixelCoords = pixelCoords - center;
    
    // Convert the pixel coordinates to coordinates in the image plane (alpha, beta)
    float lengthPerPixel = 0.1;
    float2 imagePlaneCoords = lengthPerPixel * relativePixelCoords;
    float alpha = imagePlaneCoords.x;
    float beta = imagePlaneCoords.y;

    // Convert (alpha, beta) -> (lambda, eta)
    float lambda = -1.0 * alpha * sin(thetas);
    float eta = (alpha * alpha - a * a) * cos(thetas) * cos(thetas) + beta * beta;
    float nuthetas = sign(beta);

    // We don't currently handle the case of vortical geodesics
    if (eta <= 0.0) {
        return float4(1.0, 1.0, 1.0, 1.0);
    }
    
    // Do the actual lensing. The result is a final theta and phi.
    KerrLenseResult kerrLenseResult = kerrLense(a, M, thetas, nuthetas, ro, rs, eta, lambda);
    if (kerrLenseResult.status != SUCCESS) {
        return float4(1.0, 0.0, 0.0, 1.0);
    }
    float phif = kerrLenseResult.phif;
    float thetaf = acos(kerrLenseResult.costhetaf);
    
    // Obtain the corresponding values of eta_flat, lambda_flat.
    FlatSpaceEtaLambdaResult flatSpaceEtaLambdaResult = flatSpaceEtaLambda(rs, thetas, 0, ro, thetaf, phif);
    if (flatSpaceEtaLambdaResult.status != SUCCESS) {
        return float4(1.0, 0.0, 0.0, 1.0);
    }
    float etaflat = flatSpaceEtaLambdaResult.etaflat;
    float lambdaflat = flatSpaceEtaLambdaResult.lambdaflat;
    float pthetaSign = flatSpaceEtaLambdaResult.uthetaSign;
    
    // Map back to screen coordinates
    float alphaflat = -1.0 * lambdaflat / sin(thetas);
    float termUnderRadical = etaflat - lambdaflat * lambdaflat * (1.0 / tan(thetas)) * (1.0 / tan(thetas));
    if (termUnderRadical < 0.0) {
        return float4(0.0, 1.0, 0.0, 1.0);
    }
    float betaflat = pthetaSign * sqrt(termUnderRadical);
    
    // Unwind through the texture -> screen coordinate mappings
    float2 transformedImagePlaneCoords = float2(alphaflat, betaflat);
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

float2 getPipCoord(float2 pipOrigin, float pipHeight, float pipWidth, float2 coord) {
    float2 displacement = coord - pipOrigin;
    float2 renormalizedCoord = float2(displacement.x / pipWidth, displacement.y / pipHeight);
    
    return renormalizedCoord;
}


fragment float4 preComputedFragmentShader(VertexOut in [[stage_in]],
                                          texture2d<float, access::sample> frontYTexture [[texture(0)]],
                                          texture2d<float, access::sample> frontUVTexture [[texture(1)]],
                                          texture2d<float, access::sample> backYTexture [[texture(2)]],
                                          texture2d<float, access::sample> backUVTexture [[texture(3)]],
                                          texture2d<float, access::sample> lutTexture [[texture(4)]],
                                          constant Uniforms &uniforms [[buffer(0)]]) {
    constexpr sampler s(coord::normalized, address::clamp_to_edge, filter::linear);

    float2 backPipOrigin = float2(0.05, 0.7);
    float2 frontPipOrigin = float2(0.05, 0.05);
    
    float aRatio = 1.00;
    float pipHeight = 0.22;
    float pipWidth = pipHeight * aRatio;
    
    float2 backPipCoord = getPipCoord(backPipOrigin, pipHeight, pipWidth, in.texCoord);
    if (    0.0 < backPipCoord.x && backPipCoord.x < 1.0
        &&  0.0 < backPipCoord.y && backPipCoord.y < 1.0) {
        float3 rgb = sampleYUVTexture(backYTexture, backUVTexture, backPipCoord);
        return float4(rgb, 1.0);
    }
    
    float2 frontPipCoord = getPipCoord(frontPipOrigin, pipHeight, pipWidth, in.texCoord);
    if (    0.0 < frontPipCoord.x && frontPipCoord.x < 1.0
        &&  0.0 < frontPipCoord.y && frontPipCoord.y < 1.0) {
        float3 rgb = sampleYUVTexture(frontYTexture, frontUVTexture, frontPipCoord);
        return float4(rgb, 1.0);
    }

    float4 lutSample = lutTexture.sample(s, in.texCoord);
    
    float2 transformedTexCoord = lutSample.xy;
    float2 statusCode = lutSample.zw;
    
    if (uniforms.mode == FULL_FOV_MODE) {
        if (fEqual(statusCode[0], 0.0) && fEqual(statusCode[1], 0.0)) {
            float3 rgb = sampleYUVTexture(backYTexture, backUVTexture, transformedTexCoord);
            return float4(rgb, 1.0);
        } else if (fEqual(statusCode[0], 0.0) && fEqual(statusCode[1], 1.0)) {
            float3 rgb = sampleYUVTexture(frontYTexture, frontUVTexture, transformedTexCoord);
            return float4(rgb, 1.0);
        } else if (fEqual(statusCode[0], 1.0) && fEqual(statusCode[1], 0.0)) {
            return float4(0.0, 1.0, 0.0, 1.0);
        } else if (fEqual(statusCode[0], 1.0) && fEqual(statusCode[1], 1.0)) {
            return float4(0.0, 0.0, 0.0, 1.0);
        } else if (fEqual(statusCode[0], 0.5) && fEqual(statusCode[1], 0.5)) {
            return float4(1.0, 1.0, 0.0, 1.0);
        } else {
            return float4(0.0, 0.0, 1.0, 1.0);
        }
    }
    
    if (uniforms.mode == ACTUAL_FOV_MODE) {
        if (fEqual(statusCode[0], 0.0) && fEqual(statusCode[1], 0.0)) {
            float3 rgb = sampleYUVTexture(backYTexture, backUVTexture, transformedTexCoord);
            return float4(rgb, 1.0);
        } else if (fEqual(statusCode[0], 0.0) && fEqual(statusCode[1], 1.0)) {
            return float4(0.0, 0.0, 0.0, 1.0);
        } else if (fEqual(statusCode[0], 1.0) && fEqual(statusCode[1], 0.0)) {
            return float4(0.0, 1.0, 0.0, 1.0);
        } else if (fEqual(statusCode[0], 1.0) && fEqual(statusCode[1], 1.0)) {
            return float4(0.0, 0.0, 0.0, 1.0);
        } else if (fEqual(statusCode[0], 0.5) && fEqual(statusCode[1], 0.5)) {
            return float4(1.0, 1.0, 0.0, 1.0);
        } else {
            return float4(0.0, 0.0, 1.0, 1.0);
        }
    }
    
    return float4(1.0, 1.0, 1.0, 1.0);
}

LenseTextureCoordinateResult schwarzschildLenseTextureCoordinate(float2 inCoord) {
    LenseTextureCoordinateResult result;
    
    /*
     * The convention we use is to call the camera screen the "source" since we
     * ray trace from this location back into the geometry.
     */
    float backTextureWidth = 1920.0;
    float backTextureHeight = 1080.0;
    
    // We let rs and ro be large in this set up.
    // This will allow for the usage of an approximation to the
    // elliptic integrals during lensing.
    float M = 1.0;
    float rs = 1000.0;
    float ro = rs;
    
    // Calculate the pixel coordinates of the current fragment
    float2 pixelCoords = inCoord * float2(backTextureWidth, backTextureHeight);
    
    // Calculate the pixel coordinates of the center of the image
    float2 center = float2(backTextureWidth / 2.0, backTextureHeight / 2.0);
    
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
        result.status = ERROR;
        return result;
    } else if (lenseResult.status == EMITTED_FROM_BLACK_HOLE) {
        result.status = EMITTED_FROM_BLACK_HOLE;
        return result;
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
    float2 transformedTexCoord = transformedPixelCoords / float2(backTextureWidth, backTextureHeight);

    // Ensure that the texture coordinate is inbounds
    if (transformedTexCoord.x < 0.0 || 1.0 < transformedTexCoord.x ||
        transformedTexCoord.y < 0.0 || 1.0 < transformedTexCoord.y) {
        result.status = OUTSIDE_FOV;
        return result;
    }

    result.coord = transformedTexCoord;
    result.status = SUCCESS;
    return result;
}

float2 pixelToScreen(float2 pixelCoords) {
    float base = 0.04;
    float rcrit = 300.0;
    float alpha = 1000.0;
    int n = 1;
    
    float r = sqrt(pixelCoords.x * pixelCoords.x + pixelCoords.y * pixelCoords.y);
    
    if (r < rcrit) {
        return base * pixelCoords;
    }
    
    return ((1.0 / alpha) * pow(r - rcrit, n) + base) * pixelCoords;
}

float2 pixelToScreenZoomAboutCriticalCurve(float2 pixelCoords) {
    float base = 0.04;
    float rcrit = 180.0;
    
    float r = sqrt(pixelCoords.x * pixelCoords.x + pixelCoords.y * pixelCoords.y);
    float theta = atan2(pixelCoords.y, pixelCoords.x);
    
    if (r < rcrit) {
        return base * pixelCoords;
    }
    
    float alpha = 0.01;
    float rtilde = rcrit + alpha * (r - rcrit);
    
    return base * float2(rtilde * cos(theta), rtilde * sin(theta));
}


float3 rotateSphericalCoordinate(float3 vsSpherical, float3 voSpherical) {
    float3 vsCartesian = sphericalToCartesian(vsSpherical);
    float3 voCartesian = sphericalToCartesian(voSpherical);
    
    float3 zhat = float3(0.0, 0.0, 1.0);

    float3 n1 = vsCartesian / length(vsCartesian);
    
    float3 v2 = zhat - dot(zhat, n1) * n1;
    float3 n2;
    if (fEqual(length(v2), 0.0)) {
        // When polar observer, n2 = 0 -> degenerate, arbitrary
        // up direction, just pick one in the plane.
        n2 = float3(0.0, 1.0, 0.0);
    } else {
        n2 = v2 / length(v2);
    }
    
    float3 n3 = cross(n2, n1);
    
    // This is just matrix multiplication by the matrix
    // whose rows are {n1, n3, n2}.
    float3 voHatCartesian = float3(dot(n1, voCartesian),
                                   dot(n3, voCartesian),
                                   dot(n2, voCartesian));
    
    return cartesianToSpherical(voHatCartesian);
}

LenseTextureCoordinateResult kerrLenseTextureCoordinate(float2 inCoord, int mode) {
    LenseTextureCoordinateResult result;
    
    float backTextureWidth = 1920.0;
    float backTextureHeight = 1080.0;

    /*
     * The convention we use is to call the camera screen the "source" since we
     * ray trace from this location back into the geometry.
     */
    float M = 1.0;
    float a = 0.1;
    float thetas = M_PI_F / 2.0;
    float rs = 1000.0;
    float ro = rs;
    
    // Calculate the pixel coordinates of the current fragment
    float2 pixelCoords = inCoord * float2(backTextureWidth, backTextureHeight);
    
    // Calculate the pixel coordinates of the center of the image
    float2 center = float2(backTextureWidth / 2.0, backTextureHeight / 2.0);
    
    // Place the center at the origin
    float2 relativePixelCoords = pixelCoords - center;
    
    // Convert the pixel coordinates to coordinates in the image plane (alpha, beta)
    float2 imagePlaneCoords = pixelToScreen(relativePixelCoords);
    float alpha = imagePlaneCoords.x;
    float beta = imagePlaneCoords.y;

    // Convert (alpha, beta) -> (lambda, eta)
    float lambda = -1.0 * alpha * sin(thetas);
    float eta = (alpha * alpha - a * a) * cos(thetas) * cos(thetas) + beta * beta;
    float nuthetas = sign(beta);

    // We don't currently handle the case of vortical geodesics
    if (eta <= 0.0) {
        result.status = VORTICAL;
        return result;
    }
    
    // Do the actual lensing. The result is a final theta and phi.
    KerrLenseResult kerrLenseResult = kerrLense(a, M, thetas, nuthetas, ro, rs, eta, lambda);
    if (kerrLenseResult.status != SUCCESS) {
        result.status = ERROR;
        return result;
    }
    float phif = kerrLenseResult.phif;
    float thetaf = acos(kerrLenseResult.costhetaf);
    
    if (mode == FULL_FOV_MODE) {
        float3 rotatedSphericalCoordinates = rotateSphericalCoordinate(float3(rs, thetas, 0.0),
                                                                       float3(ro, thetaf, phif));
        
        float phifNormalized = normalizeAngle(rotatedSphericalCoordinates.z);
        thetaf = rotatedSphericalCoordinates.y;
        
        float oneTwoBdd = M_PI_F / 2.0;
        float threeFourBdd = 3.0 * M_PI_F / 2.0;
        
        float v = thetaf / M_PI_F;
        float u = 0.0;
        
        // If in quadrant I
        if (0.0 <= phifNormalized && phifNormalized <= oneTwoBdd) {
            u = 0.5 + 0.5 * (phifNormalized / oneTwoBdd);
            result.status = SUCCESS_FRONT_TEXTURE;
        } else if (threeFourBdd <= phifNormalized && phifNormalized <= 2.0 * M_PI_F) { // quadrant IV
            u = 0.5 * ((phifNormalized - threeFourBdd) / (2.0 * M_PI_F - threeFourBdd));
            result.status = SUCCESS_FRONT_TEXTURE;
        } else { // II or III
            u = (phifNormalized - oneTwoBdd) / (threeFourBdd - oneTwoBdd);
            result.status = SUCCESS_BACK_TEXTURE;
        }
        float2 transformedTexCoord = float2(u, v);
        
        result.coord = transformedTexCoord;
        return result;
    }
    
    // Obtain the corresponding values of eta_flat, lambda_flat.
    FlatSpaceEtaLambdaResult flatSpaceEtaLambdaResult = flatSpaceEtaLambda(rs, thetas, 0, ro, thetaf, phif);
    if (flatSpaceEtaLambdaResult.status != SUCCESS) {
        result.status = ERROR;
        return result;
    }
    float etaflat = flatSpaceEtaLambdaResult.etaflat;
    float lambdaflat = flatSpaceEtaLambdaResult.lambdaflat;
    float pthetaSign = flatSpaceEtaLambdaResult.uthetaSign;
    
    // Map back to screen coordinates
    float alphaflat = -1.0 * lambdaflat / sin(thetas);
    float termUnderRadical = etaflat - lambdaflat * lambdaflat * (1.0 / tan(thetas)) * (1.0 / tan(thetas));
    if (termUnderRadical < 0.0) {
        result.status = ERROR;
        return result;
    }
    float betaflat = pthetaSign * sqrt(termUnderRadical);
    
    float lengthPerPixelInverse = 0.5;
    
    // Unwind through the texture -> screen coordinate mappings
    float2 transformedImagePlaneCoords = float2(alphaflat, betaflat);
    float2 transformedRelativePixelCoords = transformedImagePlaneCoords / lengthPerPixelInverse;
    float2 transformedPixelCoords = transformedRelativePixelCoords + center;
    float2 transformedTexCoord = transformedPixelCoords / float2(backTextureWidth, backTextureHeight);
    
    // Ensure that the texture coordinate is inbounds
    if (transformedTexCoord.x < 0.0 || 1.0 < transformedTexCoord.x ||
        transformedTexCoord.y < 0.0 || 1.0 < transformedTexCoord.y) {
        result.status = OUTSIDE_FOV;
        return result;
    }
    
    result.coord = transformedTexCoord;
    result.status = SUCCESS;
    return result;
}

/*
 * To avoid computing the same lensing map every frame, we compute once
 * and store the result in a look-up table (LUT). The LUT is then passed
 * to the fragment shader on subsequent render passes (per frame updates)
 * and simpled sampled.
 */
kernel void precomputeLut(texture2d<float, access::write> lut [[texture(0)]],
                          constant PreComputeUniforms &uniforms [[buffer(0)]],
                          uint2 gid [[thread_position_in_grid]]) {
    // This is normalizing to texture coordinate between 0 and 1
    float2 originalCoord = float2(gid) / float2(lut.get_width(), lut.get_height());
    LenseTextureCoordinateResult result = kerrLenseTextureCoordinate(originalCoord, uniforms.mode);
    
    // Need to pass the status code within the look-up table. We do so in the
    // zw components with binary strings (00, 01, 10, 11)
    if (uniforms.mode == FULL_FOV_MODE) {
        if (result.status == SUCCESS_BACK_TEXTURE) {
            lut.write(float4(result.coord, 0.0, 0.0), gid); // 00
        }
        if (result.status == SUCCESS_FRONT_TEXTURE) {
            lut.write(float4(result.coord, 0.0, 1.0), gid); // 01
        }
        if (result.status == ERROR) {
            lut.write(float4(0.0, 0.0, 1.0, 0.0), gid); // 10
        }
        if (result.status == EMITTED_FROM_BLACK_HOLE) {
            lut.write(float4(0.0, 0.0, 1.0, 1.0), gid); // 11
        }
        if (result.status == VORTICAL) {
            lut.write(float4(0.0, 0.0, 0.5, 0.5), gid);
        }
    }
    
    if (uniforms.mode == ACTUAL_FOV_MODE) {
        if (result.status == SUCCESS) {
            lut.write(float4(result.coord, 0.0, 0.0), gid);
        }
        if (result.status == ERROR) {
            lut.write(float4(0.0, 0.0, 0.0, 1.0), gid);
        }
        if (result.status == OUTSIDE_FOV) {
            lut.write(float4(0.0, 0.0, 1.0, 0.0), gid);
        }
        if (result.status == EMITTED_FROM_BLACK_HOLE) {
            lut.write(float4(0.0, 0.0, 1.0, 1.0), gid);
        }
        if (result.status == VORTICAL) {
            lut.write(float4(0.0, 0.0, 0.5, 0.5), gid); // 11
        }
    }
}
