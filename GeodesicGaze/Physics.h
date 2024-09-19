//
//  Physics.h
//  GeodesicGaze
//
//  Created by Trevor Gravely on 9/2/24.
//

#ifndef Physics_h
#define Physics_h

#define EMITTED_FROM_BLACK_HOLE 3
#define IMAGINARY_VALUE_ENCOUNTERED 4

struct SchwarzschildLenseResult {
    float varphitilde;
    bool ccw;
    int status;
};

struct PhiSResult {
    float val;
    int status;
};

struct KerrRadialRootsResult {
    float2 roots[4];
    int status;
};

float4 radialRoots(float M, float b);
float3 computeABC(float a, float M, float eta, float lambda);
KerrRadialRootsResult kerrRadialRoots(float a, float M, float eta, float lambda);

PhiSResult phiS(float M, float ro, float rs, float b);

/*
* Given a conserved value of b (angular momentum), an observer radius, and a source radius,
* determine the point at which a null geodesic in Schwarzschild would intersect
* the source radius, if fired from the observer.
*
* Then compute the angle of incidence that a light ray would have in flat space,
* such that it intersects this point, and return that value.
*
* The returned angle of incidence is about the line of sight. The boolean value
* ccw indicates whether it is clockwise or counter-clockwise.
*/
SchwarzschildLenseResult schwarzschildLense(float M, float ro, float rs, float b);

#endif /* Physics_h */
