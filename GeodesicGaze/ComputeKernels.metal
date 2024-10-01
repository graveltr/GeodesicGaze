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

struct KerrLenseResultForSwift {
    float alpha;
    float beta;
    float eta;
    float lambda;
    float phif;
    float thetaf;
    int kerrLenseStatus;
};

struct ResultForSwift {
    float IrValue;
    float cosThetaObserverValue;
    float GphiValue;
    float mathcalGphisValue;
    float psiTauValue;
    float mathcalGthetasValue;
    int IrStatus;
    int cosThetaObserverStatus;
    int GphiStatus;
    int mathcalGphisStatus;
    int psiTauStatus;
    int mathcalGthetasStatus;
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

kernel void kerr_lense_compute_kernel(const device float *dummyData [[buffer(0)]],
                                      device KerrLenseResultForSwift *results [[buffer(1)]],
                                      uint id [[thread_position_in_grid]]) {
    KerrLenseResultForSwift result;

    float2 texCoord = float2(0.4, 0.4);
    float backTextureWidth  = 1920.0;
    float backTextureHeight = 1080.0;

    float M = 1.0;
    float a = 0.6;
    float thetas = M_PI_F / 4.0;
    float rs = 1000.0;
    float ro = rs;

    // Calculate the pixel coordinates of the current fragment
    float2 pixelCoords = texCoord * float2(backTextureWidth, backTextureHeight);

    // Calculate the pixel coordinates of the center of the image
    float2 center = float2(backTextureWidth / 2.0, backTextureHeight / 2.0);

    // Place the center at the origin
    float2 relativePixelCoords = pixelCoords - center;

    // Convert the pixel coordinates to coordinates in the image plane (alpha, beta)
    float lengthPerPixel = 0.1;
    float2 imagePlaneCoords = lengthPerPixel * relativePixelCoords;
    float alpha = imagePlaneCoords.x;
    float beta = imagePlaneCoords.y;

    result.alpha = alpha;
    result.beta = beta;

    // Convert (alpha, beta) -> (lambda, eta)
    float lambda = -1.0 * alpha * sin(thetas);
    float eta = (alpha * alpha - a * a) * cos(thetas) * cos(thetas) + beta * beta;
    float nuthetas = sign(beta);

    result.eta = eta;
    result.lambda = lambda;

    // We don't currently handle the case of vortical geodesics
    if (eta <= 0.0) {
    return;
    }

    // Do the actual lensing. The result is a final theta and phi.
    KerrLenseResult kerrLenseResult = kerrLense(a, M, thetas, nuthetas, ro, rs, eta, lambda);
    result.kerrLenseStatus = kerrLenseResult.status;
    
    float phif = kerrLenseResult.phif;
    float thetaf = acos(kerrLenseResult.costhetaf);
    
    result.phif = phif;
    result.thetaf = thetaf;

    // Obtain the corresponding values of eta_flat, lambda_flat.
    FlatSpaceEtaLambdaResult flatSpaceEtaLambdaResult = flatSpaceEtaLambda(rs, thetas, 0, ro, thetaf, phif);
    if (flatSpaceEtaLambdaResult.status != SUCCESS) {
    }
    float etaflat = flatSpaceEtaLambdaResult.etaflat;
    float lambdaflat = flatSpaceEtaLambdaResult.lambdaflat;
    float pthetaSign = flatSpaceEtaLambdaResult.uthetaSign;

    // Map back to screen coordinates
    float alphaflat = -1.0 * lambdaflat / sin(thetas);
    float termUnderRadical = etaflat - lambdaflat * lambdaflat * (1.0 / tan(thetas)) * (1.0 / tan(thetas));
    if (termUnderRadical < 0.0) {
    }
    float betaflat = pthetaSign * sqrt(termUnderRadical);

    // Unwind through the texture -> screen coordinate mappings
    float2 transformedImagePlaneCoords = float2(alphaflat, betaflat);
    float2 transformedRelativePixelCoords = transformedImagePlaneCoords / lengthPerPixel;
    float2 transformedPixelCoords = transformedRelativePixelCoords + center;
    float2 transformedTexCoord = transformedPixelCoords / float2(backTextureWidth, backTextureHeight);

    // Ensure that the texture coordinate is inbounds
    if (transformedTexCoord.x < 0.0 || 1.0 < transformedTexCoord.x ||
    transformedTexCoord.y < 0.0 || 1.0 < transformedTexCoord.y) {
    }

    results[id] = result;
}

kernel void tau_compute_kernel(const device float *dummyData [[buffer(0)]],
                               device ResultForSwift *results [[buffer(1)]],
                               uint id [[thread_position_in_grid]]) {
    ResultForSwift result;
    
    float M = 1.0;
    float a = 0.6;
    float thetas = M_PI_F / 4.0;
    float rs = 1000.0;
    float ro = rs;
    float eta = 300.78;
    float lambda = 13.57645;
    
    float r1 = -22.562675;
    float r2 = 0.12738323;
    float r3 = 1.8230124;
    float r4 = 20.612282;
    
    IrResult IrResult = computeIr(a, M, ro, rs, r1, r2, r3, r4);
    result.IrStatus = IrResult.status;
    result.IrValue = IrResult.val;
    
    float tau = IrResult.val;
    
    CosThetaObserverResult cosThetaObserverResult = cosThetaObserver(1, tau, a, M, thetas, eta, lambda);
    result.cosThetaObserverValue = cosThetaObserverResult.val;
    result.cosThetaObserverStatus = cosThetaObserverResult.status;
    
    // TODO: Resolve this error!
    // START Gphi
    /*
    Result GphiResult = computeGphi(1, tau, a, M, thetas, eta, lambda);
    result.GphiValue = GphiResult.val;
    result.GphiStatus = GphiResult.status;
    */
    
    float deltaTheta = (1.0 / 2.0) * (1.0 - (eta + lambda * lambda) / (a * a));
    
    float uplus = deltaTheta + sqrt(deltaTheta * deltaTheta + (eta / (a * a)));
    float uminus = deltaTheta - sqrt(deltaTheta * deltaTheta + (eta / (a * a)));
    
    Result mathcalGphiResult = mathcalGphi(a, thetas, uplus, uminus);
    result.mathcalGphisValue = mathcalGphiResult.val;
    result.mathcalGphisStatus = mathcalGphiResult.status;
    
    // TODO: Resolve this error!
    /*
    Result psiTauResult = Psitau(a, uplus, uminus, tau, thetas, 1);
    result.psiTauValue = psiTauResult.val;
    result.psiTauStatus = psiTauResult.status;
    */
    
    // START PSITAU
    MathcalGResult mathcalGthetasResult = mathcalGtheta(a, thetas, uplus, uminus);
    result.mathcalGthetasValue = mathcalGthetasResult.val;
    result.mathcalGthetasStatus = mathcalGthetasResult.status;
    float mathcalGthetas = mathcalGthetasResult.val;
    
    float u = sqrt(-1.0 * uminus * a * a) * (tau + 1 * mathcalGthetas);
    float m = uplus / uminus;
    
    EllamResult amResult = jacobiam(u, m);
    result.psiTauValue = amResult.am;
    result.psiTauStatus = amResult.status;
    // END PSITAU
    
    // END Gphi

    results[id] = result;
}
kernel void jacobiam_compute_kernel(const device float *u [[buffer(0)]],
                               const device float *m [[buffer(1)]],
                               device float *results [[buffer(2)]],
                               uint id [[thread_position_in_grid]]) {
    results[id] = jacobiam(u[id], m[id]).am;
}
