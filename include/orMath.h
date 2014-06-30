/*
 * File:   orMath.h
 * Author: fib
 *
 * Created on 26 December 2012, 12:20
 */

#ifndef ORMATH_H
#define	ORMATH_H

# include <cmath>
# include <Eigen/Eigen>
EIGEN_USING_MATRIX_TYPEDEFS;

#include "constants.h"

struct orVec2 {
  orVec2() {
    for (int i = 0; i < 2; ++i) {
      data[i] = 0;
    }
  }

  orVec2(double x, double y) {
    data[0] = x;
    data[1] = y;
  }

  orVec2(Eigen::Vector2d v) {
    double const* vdata = v.data();
    for (int i = 0; i < 2; ++i) {
      data[i] = vdata[i];
    }
  }

  double&       operator[] (int i)       { return data[i]; }
  double const& operator[] (int i) const { return data[i]; }
  double data[2];
}; // struct orVec2

struct orVec3 {
  orVec3() {
    for (int i = 0; i < 3; ++i) {
      data[i] = 0;
    }
  }

  orVec3(double x, double y, double z) {
    data[0] = x;
    data[1] = y;
    data[2] = z;
  }

  orVec3(Eigen::Vector3d v) {
    double const* vdata = v.data();
    for (int i = 0; i < 3; ++i) {
      data[i] = vdata[i];
    }
  }

  double&       operator[] (int i)       { return data[i]; }
  double const& operator[] (int i) const { return data[i]; }

  operator Eigen::Vector3d() const {
    return Eigen::Vector3d(data);
  }

  double data[3];
}; // struct orVec3

template <typename T, typename A>
static T orLerp(T const _x0, T const _x1, A const _a) {
    return _x0 * (1 - _a) + _x1 * _a;
}

struct orOrbitParams
{
  orOrbitParams(): p(0), e(0), theta(0), x_dir(), y_dir() {}

  // TODO document these parameters
  double p; // p is the 'semi-latus rectum' of the conic section
  double e; // e is the 'orbital eccentricity'
  double theta; // theta is the 'true anomaly'
  orVec3 x_dir;
  orVec3 y_dir;
};

// TODO want to support multiple ways of computing pos+vel of an object at a
// given time:
// - the JPL approximations we already have,
// - 'on-rails' keplerian orbits (but we want to be able to get the pos and vel at a time in the future, so existing code is bad; see Chap 4 in Orbital Mechanics textbook maybe?
// Want to get the keplerian params of the moon from http://ssd.jpl.nasa.gov/?sat_elem and ideally offset the earth from its barycenter too
// - RK4 propagator (maybe others in future, STI Astrogator recommends adaptive RK7 with RK8 error correction - of course that takes more forces into account)
// TODO will want to include the fuel/propellant mass in calculations; ship mass should lower after each maneuver

inline void orbitParamsFromPosAndVel(
  Vector3d const& r2, // body pos relative to parent
  Vector3d const& v, // body vel relative to parent
  double const M, // parent body mass
  // TODO want to also return some representation of the current position in the orbit
  orOrbitParams& o_params
) {
  // Compute Kepler orbit

  double const G = GRAV_CONSTANT;

  double const mu = M * G;

  Vector3d const r = -r2; // TODO substitute through

  double const r_mag = r.norm();
  Vector3d const r_dir = r/r_mag;

  double const vr_mag = r_dir.dot(v);
  Vector3d const vr = r_dir * vr_mag; // radial velocity
  Vector3d const vt = v - vr; // tangent velocity
  double const vt_mag = vt.norm();
  Vector3d const t_dir = vt/vt_mag;

  // h is the 'specific relative angular momentum' of the body with respect to the parent
  // (actually, this is the magnitude; including direction it is r.cross(vt))
  double const h = r_mag * vt_mag;

  // L is the magnitude of the total angular momentum
  // double const L = h * mu;
  // p = h * h / mu = (L/mu) * (L/mu) / mu = (L*L) / (mu*mu*mu)

  // p is the 'semi-latus rectum' of the conic section
  // p = a * (1 - e * e)?
  // p = b * b / a for an ellipse
  double const p = (h * h) / mu; // TODO why pow(x, 2) instead of x * x?
  double const v0 = sqrt(mu/p); // TODO compute more accurately/efficiently? // TODO where did I suspect inaccuracy or efficiency here?

  Vector3d const ex = ((vt_mag - v0) * r_dir - vr_mag * t_dir) / v0;
  // e is the 'orbital eccentricity'
  double const e = ex.norm();

  double const ec = (vt_mag / v0) - 1;
  double const es = (vr_mag / v0);

  // theta is the 'true anomaly'
  double const theta = atan2(es, ec);

  Vector3d const x_dir = cos(theta) * r_dir - sin(theta) * t_dir;
  Vector3d const y_dir = sin(theta) * r_dir + cos(theta) * t_dir;

  o_params.e = e;
  o_params.p = p;
  o_params.theta = theta;

  o_params.x_dir = x_dir;

  o_params.y_dir = y_dir;
}

inline void sampleOrbit(
  orOrbitParams const& params,
  orVec3 const& origin,
  int const num_steps,
  orVec3* const o_posData
) {
    double const delta = .0001;
    double const HAX_RANGE = .9; // limit range to stay out of very large values

    double range;
    if (params.e < 1 - delta) { // ellipse
        range = .5 * M_TAU;
    } else if (params.e < 1 + delta) { // parabola
        range = .5 * M_TAU * HAX_RANGE;
    } else { // hyperbola
        range = acos(-1/params.e) * HAX_RANGE;
    }
    double const mint = -range;
    double const maxt = range;

#if 0
    Vector3d const orbit_x(params.x_dir);
    Vector3d const orbit_y(params.y_dir);
    Vector3d const orbit_pos(origin);
#endif

    for (int i = 0; i < num_steps; ++i) {
        double const ct = orLerp(mint, maxt, (double)i / num_steps);
        double const cr = params.p / (1 + params.e * cos(ct));
        double const x_len = cr * -cos(ct);
        double const y_len = cr * -sin(ct);

#if 0 // Original version (correct implementation)
        Vector3d const pos = (orbit_x * x_len) + (orbit_y * y_len) + orbit_pos;
        o_posData[i] = orVec3(pos);
#else // No vector version (correct implementation, faster...)
        o_posData[i][0] = (params.x_dir[0] * x_len) + (params.y_dir[0] * y_len) + origin[0];
        o_posData[i][1] = (params.x_dir[1] * x_len) + (params.y_dir[1] * y_len) + origin[1];
        o_posData[i][2] = (params.x_dir[2] * x_len) + (params.y_dir[2] * y_len) + origin[2];
#endif
    }
}

namespace orMath {

// returns eccentric anomaly in radians
// from http://ssd.jpl.nasa.gov/txt/aprx_pos_planets.pdf

// "If this iteration formula won't converge, the eccentricity is probably too
// close to one. Then you should instead use the formulae for near-parabolic or
// parabolic orbits."
// http://astro.if.ufrgs.br/trigesf/position.html
inline double computeEccentricAnomaly(
  double mean_anomaly_rad,
  double eccentricity
) {
  double const tolerance_rad = 10e-6 * (M_TAU / 360.0);
  double eccentric_anomaly_rad = mean_anomaly_rad + eccentricity * sin(mean_anomaly_rad);
  double delta_eccentric_anomaly_rad = 0;
  do
  {
    double const delta_mean_anomaly_rad = mean_anomaly_rad - (eccentric_anomaly_rad - eccentricity * sin(eccentric_anomaly_rad));
    double const delta_eccentric_anomaly_rad = delta_mean_anomaly_rad / (1 - eccentricity * cos(eccentric_anomaly_rad));
    eccentric_anomaly_rad += delta_eccentric_anomaly_rad;
  } while (delta_eccentric_anomaly_rad > tolerance_rad);

  return eccentric_anomaly_rad;
}

} // namespace orMath

#endif	/* ORMATH_H */

