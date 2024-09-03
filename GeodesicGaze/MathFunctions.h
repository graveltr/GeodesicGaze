//
//  MathFunctions.h
//  GeodesicGaze
//
//  Created by Trevor Gravely on 9/1/24.
//

#ifndef MathFunctions_h
#define MathFunctions_h

#define DOMAIN_ERROR -1
#define MAXITER_ERROR -2
#define SUCCESS 0
#define FAILURE 1

#define locMAX3(a,b,c) max(max(a, b), c)
#define locMAX4(a,b,c,d) max(max(max(a, b), c), d)

struct EllintResult {
    float val;
    float err;
    int status;
};

struct ElljacResult {
    float sn;
    float cn;
    float dn;
    int status;
};

EllintResult ellint_F(float phi, float k, float errtol, float prec);

float normalizeAngle(float phi);

bool fEqual(float x, float y);

#endif /* MathFunctions_h */
