//
//  Physics.h
//  GeodesicGaze
//
//  Created by Trevor Gravely on 9/2/24.
//

#ifndef Physics_h
#define Physics_h

#define EMITTED_FROM_BLACK_HOLE 3

struct SchwarzschildLenseResult {
    float varphitilde;
    bool ccw;
    int status;
};

struct PhiSResult {
    float val;
    int status;
};

float4 radialRoots(float M, float b);

PhiSResult phiS(float M, float ro, float rs, float b);

/*
* Given an angle of incidence varphi, an observer radius, and a source radius,
* determine the point at which a null geodesic in Schwarzschild would intersect
* the source radius, if fired from the observer.
*
* Then compute the angle of incidence that a light ray would have in flat space,
* such that it intersects this point, and return that value.
*/
SchwarzschildLenseResult schwarzschildLense(float M, float ro, float rs, float varphi);

#endif /* Physics_h */
