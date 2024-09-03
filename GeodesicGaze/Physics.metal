//
//  Physics.metal
//  GeodesicGaze
//
//  Created by Trevor Gravely on 9/2/24.
//

#include <metal_stdlib>
#include "Physics.h"
#include "MathFunctions.h"

using namespace metal;


struct PhiSResult {
    float val;
    int status;
};

// The roots of the radial potential
float4 radialRoots(float M, float b) {
    float bc = 3.0 * sqrt(3.0) * M;
    
    float r1 = (-2.0 * b / sqrt(3.0)) * cos((1.0 / 3.0) * acos(bc / b));
    float r2 = 0.0;
    float r3 = (2.0 * b / sqrt(3.0)) * sin((1.0 / 3.0) * asin(bc / b));
    float r4 = (2.0 * b / sqrt(3.0)) * cos((1.0 / 3.0) * acos(-1.0 * bc / b));
    
    return float4(r1, r2, r3, r4);
}

// TODO: verify that this expression, including conventions, is exactly correct
// TODO: ensure proper error handling, no divide by zeros, etc.
EllintResult mathcalF(float M, float r, float b, float modulus) {
    float4 roots = radialRoots(M, b);
    
    float r1 = roots[0];
    float r2 = roots[1];
    float r3 = roots[2];
    float r4 = roots[3];
    
    float arg = asin(sqrt(((r - r4) / (r - r3)) * ((r3 - r1) / (r4 - r1))));
    
    return ellint_F(arg, modulus, 1e-5, 1e-5);
}

PhiSResult phiS(float M, float ro, float rs, float b) {
    PhiSResult result;
    
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

SchwarzschildLenseResult schwarzschildLense(float M, float ro, float rs, float varphi) {
    SchwarzschildLenseResult result;
    
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
    
    // Use the law of cosines to obtain the missing side length
    float AC = sqrt(ro * ro + rs * rs - 2 * ro * rs * cos(ABC));
    
    // Two possibilities when using asin
    float BCA = 0.0;
    float BCA1 = asin((sin(normalizedAngle) / AC) * rs);
    float BCA2 = M_PI_F - BCA1;
    
    // To determine which is the angle of the triangle, check
    // the law of sines for all sides.
    if (fEqual(AC / sin(normalizedAngle), rs / sin(BCA1)) &&
        fEqual(AC / sin(normalizedAngle), ro / sin(M_PI_F - (BCA1 + normalizedAngle)))) {
        BCA = BCA1;
    } else {
        BCA = BCA2;
    }
    
    result.varphitilde = BCA;
    result.status = SUCCESS;
    return result;
}
