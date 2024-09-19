//
//  Physics.metal
//  GeodesicGaze
//
//  Created by Trevor Gravely on 9/2/24.
//

#include <metal_stdlib>
#include "Physics.h"
#include "MathFunctions.h"
#include "ComplexMath.h"

using namespace metal;

// The roots of the radial potential.
// ASSUMPTION: We assume that the caller
// ensures bc < b => bc / b < 1 such that
// we don't encounter domain issues with
// arctrig functions.
float4 radialRoots(float M, float b) {
    float bc = 3.0 * sqrt(3.0) * M;
    
    assert(bc < b);
    assert(bc < b);

    float r1 = (-2.0 * b / sqrt(3.0)) * cos((1.0 / 3.0) * acos(bc / b));
    float r2 = 0.0;
    float r3 = (2.0 * b / sqrt(3.0)) * sin((1.0 / 3.0) * asin(bc / b));
    float r4 = (2.0 * b / sqrt(3.0)) * cos((1.0 / 3.0) * acos(-1.0 * bc / b));
    
    return float4(r1, r2, r3, r4);
}

float3 computeABC(float a, float M, float eta, float lambda) {
    float A = a * a - eta - lambda * lambda;
    float B = 2.0 * M * (eta + (lambda - a) * (lambda - a));
    float C = -1.0 * a * a * eta;
    
    return float3(A,B,C);
}

float2 computePQ(float a, float M, float eta, float lambda) {
    float A = a * a - eta - lambda * lambda;
    float B = 2.0 * M * (eta + (lambda - a) * (lambda - a));
    float C = -1.0 * a * a * eta;
    
    float P = -1.0 * (A * A / 12.0) - C;
    float Q = -1.0 * (A / 3.0) * ((A / 6.0) * (A / 6.0) - C) - B * B / 8.0;

    return float2(P, Q);
}

KerrRadialRootsResult kerrRadialRoots(float a, float M, float eta, float lambda) {
    KerrRadialRootsResult result;
    
    float A = a * a - eta - lambda * lambda;
    float B = 2.0 * M * (eta + (lambda - a) * (lambda - a));
    float C = -1.0 * a * a * eta;
    
    float P = -1.0 * (A * A / 12.0) - C;
    float Q = -1.0 * (A / 3.0) * ((A / 6.0) * (A / 6.0) - C) - B * B / 8.0;
    
    float2 term;
    float termUnderRadical = pow(P / 3.0, 3.0) + pow(Q / 2.0, 2.0);
    if (termUnderRadical < 0.0) {
        term = float2(0, 1) * sqrt(-1 * termUnderRadical);
    } else {
        term = float2(1, 0) * sqrt(termUnderRadical);
    }
    
    float2 otherTerm = (-1.0 * Q / 2.0) * float2(1.0, 0.0);
    float2 omegaplus = pow1over3(otherTerm + term);
    float2 omegaminus = pow1over3(otherTerm - term);
    
    float2 xi0 = omegaplus + omegaminus - (A / 3.0) * float2(1.0, 0.0);
    
    
    bool imaginaryPartVanishes = fEqual(FLT_EPSILON, xi0.y);
    assert(imaginaryPartVanishes);
    
    float z = sqrt(xi0.x / 2.0);
    
    float2 r1, r2;
    termUnderRadical = (-1.0 * A / 2.0) - z * z + (B / (4.0 * z));
    if (termUnderRadical < 0.0) {
        r1 = -1.0 * z * float2(1.0, 0.0) - float2(0.0, 1.0) * sqrt(-1.0 * termUnderRadical);
        r2 = -1.0 * z * float2(1.0, 0.0) + float2(0.0, 1.0) * sqrt(-1.0 * termUnderRadical);
    } else {
        r1 = -1.0 * z * float2(1.0, 0.0) - float2(1.0, 0.0) * sqrt(termUnderRadical);
        r2 = -1.0 * z * float2(1.0, 0.0) + float2(1.0, 0.0) * sqrt(termUnderRadical);
    }
    
    float2 r3, r4;
    termUnderRadical = (-1.0 * A / 2.0) - z * z - (B / (4.0 * z));
    if (termUnderRadical < 0.0) {
        r3 = z * float2(1.0, 0.0) - float2(0.0, 1.0) * sqrt(-1.0 * termUnderRadical);
        r4 = z * float2(1.0, 0.0) + float2(0.0, 1.0) * sqrt(-1.0 * termUnderRadical);
    } else {
        r3 = z * float2(1.0, 0.0) - float2(1.0, 0.0) * sqrt(termUnderRadical);
        r4 = z * float2(1.0, 0.0) + float2(1.0, 0.0) * sqrt(termUnderRadical);
    }

    result.status = SUCCESS;
    result.roots[0] = r1;
    result.roots[1] = r2;
    result.roots[2] = r3;
    result.roots[3] = r4;
    
    return result;
}

EllintResult mathcalF(float M, float r, float b, float modulus) {
    assert(0 <= modulus);
    
    float4 roots = radialRoots(M, b);
    float r1 = roots[0];
    float r2 = roots[1];
    float r3 = roots[2];
    float r4 = roots[3];
    
    float arg = asin(sqrt(((r - r4) / (r - r3)) * ((r3 - r1) / (r4 - r1))));
    
    return ellint_F(arg, sqrt(modulus), 1e-5, 1e-5);
}

PhiSResult phiS(float M, float ro, float rs, float b) {
    PhiSResult result;
    
    // Note that phiS is only called when bc < b, such that
    // radialRoots requirements are met.
    float4 roots = radialRoots(M, b);
    
    float r1 = roots[0];
    float r2 = roots[1];
    float r3 = roots[2];
    float r4 = roots[3];

    float modulus = ((r3 - r2) * (r4 - r1)) / ((r3 - r1) * (r4 - r2));
    float prefactor = (2 * b) / (sqrt((r3 - r1) * (r4 - r2)));
    
    EllintResult roF = mathcalF(M, ro, b, modulus);
    EllintResult rsF = mathcalF(M, rs, b, modulus);
    
    if (roF.status != SUCCESS || rsF.status != SUCCESS) {
        result.val = 0.0;
        result.status = FAILURE;
        return result;
    }
    
    result.val = prefactor * (roF.val + rsF.val);
    result.status = SUCCESS;
    return result;
}

SchwarzschildLenseResult schwarzschildLense(float M, float ro, float rs, float b) {
    SchwarzschildLenseResult result;
    
    // If we have b < bc, then the photon trajectory enters the horizon.
    float bc = 3.0 * sqrt(3.0) * M;
    if (b < bc) {
        result.status = EMITTED_FROM_BLACK_HOLE;
        return result;
    }
    
    /*
    * The value of lambda is the initial angular momenta of the light ray.
    * We use this initial condition to ray trace the light ray into the
    * Schwarzschild geometry. In particular, we compute the accrued azimuthal
    * rotation of the resulting ray as it traverses out to rs.
     */
    
    PhiSResult phiSResult = phiS(M, ro, rs, b);
    if (phiSResult.status == FAILURE) {
        result.status = FAILURE;
        return result;
    }
    float phiS = phiSResult.val;
    
    /*
    * We now consider the triangle formed via A = (r = rs, \phi = \Delta \phi),
    * B = the origin, and C = (r = ro, \phi = 0). This defines an angle BCA,
    * which is the incidence angle of the corresponding flat space light ray
    * that interects the source point at rs. The angle ABC is easily computed
    * from the value of \Delta \phi. The triangle can then be solved for the angle
    * of interest, BCA.
    */
    
    // Normalize the angle to lie between 0 and 2 pi
    float normalizedAngle = normalizeAngle(phiS);
    
    // Obtain the angle ABC, along with the direction of the angle
    // away from the line of sight.
    float ABC = 0.0;
    bool ccw = false;
    
    if (fEqual(normalizedAngle, M_PI_F)) {
        result.varphitilde = 0;
        result.ccw = true;
        result.status = SUCCESS;
        return result;
    } else if (fEqual(normalizedAngle, 0) || fEqual(normalizedAngle, 2.0 * M_PI_F)) {
        result.varphitilde = M_PI_F;
        result.ccw = true;
        result.status = SUCCESS;
        return result;
    } else if (0 < normalizedAngle < M_PI_F) {
        // If the source location is in the upper half plane, then
        // the obtained angle of the triangle is a rotation cw.
        ABC = normalizedAngle;
        ccw = false;
    } else if (M_PI_F < normalizedAngle < 2.0 * M_PI_F) {
        // If the source location is in the lower half plane, then
        // the obtained angle of the triangle is a rotation ccw.
        ABC = 2 * M_PI_F - normalizedAngle;
        ccw = true;
    } else {
        assert(false);
    }
    result.ccw = ccw;

    // Use the law of cosines to obtain the missing side length
    float AC = sqrt(ro * ro + rs * rs - 2 * ro * rs * cos(ABC));
    
    // Two possibilities when using asin
    float BCA1 = asin((sin(ABC) / AC) * rs);
    float BCA2 = M_PI_F - BCA1;
    
    // To determine which is the angle of the triangle, check
    // the law of sines for all sides. You will run into
    // precision issues when checking for equality, so pick the
    // one that is closest.
    float diff1 = fabs((AC / sin(ABC)) - (ro / sin(M_PI_F - (BCA1 + ABC))));
    float diff2 = fabs((AC / sin(ABC)) - (ro / sin(M_PI_F - (BCA2 + ABC))));
    
    result.varphitilde = diff1 < diff2 ? BCA1 : BCA2;
    result.status = SUCCESS;
    return result;
}
