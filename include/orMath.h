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

// Boost

#include "boost_begin.h"
#include <boost/date_time/posix_time/posix_time.hpp>
#include "boost_end.h"

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


// Floating-point modulus - handles negative values correctly
// From http://stackoverflow.com/questions/4633177/c-how-to-wrap-a-float-to-the-interval-pi-pi
template <typename T>
static T orFMod(T const _a, T const _b)
{
  if (_b == 0) { return _a; } // Welp

  return _a - _b * floor(_a / _b);
}

/* Wrap a value into the given range, for example -.5f*M_TAU to +.5f*M_TAU */
template <typename T>
static T orWrap(T const _x, T const _min, T const _max) {
    return orFMod(_x - _min, _max - _min) + _min;
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


inline boost::posix_time::ptime posixTimeFromSimTime(float simTime) {
  using namespace boost::posix_time;
  using namespace boost::gregorian;
  typedef boost::posix_time::ptime posix_time;

  // Astronomical Epoch: 1200 hours, 1 January 2000
  posix_time epoch(date(2000, Jan, 1), hours(12));
  // Game start date: 1753 hours, Mar 15 2025 (Did I pick this for any reason?)
  posix_time gameStart(date(2025, Mar, 15), hours(1753));
  // Note: the (long) here limits us to ~68 years game time.
  // Should be enough, otherwise just need to keep adding seconds to the
  // dateTime to match the simTime.
  posix_time curDateTime = gameStart + seconds((long)simTime);
  return curDateTime;
}

inline std::string calendarDateFromSimTime(float simTime) {
  using namespace boost::posix_time;
  return to_simple_string(posixTimeFromSimTime(simTime));
}

inline double julianDateFromPosixTime(
  boost::posix_time::ptime const& ptime
) {
  using namespace boost::posix_time;
  using namespace boost::gregorian;
  typedef boost::posix_time::ptime posix_time;
  // wikipedia: posix_time = (julian_date - 2440587.5) * 86400
  // => (posix_time / 86400.0) + 2440587.5 = julian_date
  posix_time posix_epoch(date(1970, Jan, 1), hours(0));
  boost::posix_time::time_duration d = (ptime - posix_epoch);
  double posix_time_s = (double)d.ticks() / (double)d.ticks_per_second();
  return (posix_time_s / SECONDS_PER_DAY) + 2440587.5;
}

struct orEphemerisHybrid
{
  orEphemerisHybrid(): p(0), e(0), theta(0), x_dir(), y_dir() {}

  double p; // p is the 'semi-latus rectum' of the conic section
  double e; // e is the 'orbital eccentricity'
  double theta; // theta is the 'true anomaly'
  // TODO document
  orVec3 x_dir;
  orVec3 y_dir;
};

// TODO convert to SI units
  struct orEphemerisJPL {
    // AU: astrononmical unit
    // C: century
    // deg: degree
    double semi_major_axis_AU;
    double eccentricity;
    double inclination_deg;
    double mean_longitude_deg;
    double longitude_of_perihelion_deg;
    double longitude_of_ascending_node_deg;
    double semi_major_axis_AU_per_C;
    double eccentricity_per_C;
    double inclination_deg_per_C;
    double mean_longitude_deg_per_C;
    double longitude_of_perihelion_deg_per_C;
    double longitude_of_ascending_node_deg_per_C;

    double error_b_deg;
    double error_c_deg;
    double error_s_deg;
    double error_f_deg;
  };

  struct orEphemerisCartesian {
    Eigen::Vector3d pos;
    Eigen::Vector3d vel;
  };

// TODO want to support multiple ways of computing pos+vel of an object at a
// given time:
// - the JPL approximations we already have,
// - 'on-rails' keplerian orbits (but we want to be able to get the pos and vel at a time in the future, so existing code is bad; see Chap 4 in Orbital Mechanics textbook maybe?
// Want to get the keplerian params of the moon from http://ssd.jpl.nasa.gov/?sat_elem and ideally offset the earth from its barycenter too
// - RK4 propagator (maybe others in future, STI Astrogator recommends adaptive RK7 with RK8 error correction - of course that takes more forces into account)
// TODO will want to include the fuel/propellant mass in calculations; ship mass should lower after each maneuver

inline void ephemerisHybridFromCartesian(
  orEphemerisCartesian const& cart,
  double const M, // parent body mass
  // TODO want to also return some representation of the current position in the orbit
  orEphemerisHybrid& o_params
) {
  // Compute Kepler orbit

  double const G = GRAV_CONSTANT;

  double const mu = M * G;

  Vector3d const r = -cart.pos; // TODO substitute through

  double const r_mag = r.norm();
  Vector3d const r_dir = r/r_mag;

  double const vr_mag = r_dir.dot(cart.vel);
  Vector3d const vr = r_dir * vr_mag; // radial velocity
  Vector3d const vt = cart.vel - vr; // tangent velocity
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

inline void ephemerisCartesianFromJPL(
  orEphemerisJPL const& elements_t0,
  boost::posix_time::ptime const& ptime,
  orEphemerisCartesian& o_cart
) {
  // Compute time in centuries since J2000

  // julian date in days
  double julian_date = julianDateFromPosixTime(ptime);
  double t_C = (julian_date - 2451545.0) / DAYS_PER_CENTURY;

  // Update elements for ephemerides
  orEphemerisJPL e(elements_t0);
  e.semi_major_axis_AU += e.semi_major_axis_AU_per_C * t_C;
  e.eccentricity += e.eccentricity_per_C * t_C;
  e.inclination_deg += e.inclination_deg_per_C * t_C;
  e.mean_longitude_deg += e.mean_longitude_deg_per_C * t_C;
  e.longitude_of_perihelion_deg += e.longitude_of_perihelion_deg_per_C * t_C;
  e.longitude_of_ascending_node_deg += e.longitude_of_ascending_node_deg_per_C * t_C;

  // arg: argument
  double const arg_of_perihelion_deg = e.longitude_of_perihelion_deg - e.longitude_of_ascending_node_deg;

  // NOTE assuming error_f needs deg->rad conversion, since all other angles in the paper needed it
  double const error_f_rad = e.error_f_deg * t_C * RAD_PER_DEG;

  double const mean_anomaly_deg = e.mean_longitude_deg - e.longitude_of_perihelion_deg
    + e.error_b_deg * t_C * t_C
    + e.error_c_deg * cos(error_f_rad)
    + e.error_s_deg * sin(error_f_rad);

  double const mean_anomaly_rad = orWrap(mean_anomaly_deg * RAD_PER_DEG, -0.5 * M_TAU, +0.5 * M_TAU);

  double const eccentric_anomaly_rad = orMath::computeEccentricAnomaly(mean_anomaly_rad, e.eccentricity);

  double const semi_major_axis_meters = METERS_PER_AU * e.semi_major_axis_AU;
  double const x_orbital = semi_major_axis_meters * (cos(eccentric_anomaly_rad) - e.eccentricity);
  double const y_orbital = semi_major_axis_meters * sqrt(1 - e.eccentricity * e.eccentricity) * sin(eccentric_anomaly_rad);

  Eigen::Vector3d r_orbital(x_orbital, y_orbital, 0);

  Eigen::Matrix3d rot_inertial_frame;
  rot_inertial_frame = Eigen::AngleAxisd(-e.longitude_of_ascending_node_deg * RAD_PER_DEG, Eigen::Vector3d::UnitZ())
                     * Eigen::AngleAxisd(-e.inclination_deg * RAD_PER_DEG, Eigen::Vector3d::UnitX())
                     * Eigen::AngleAxisd(-arg_of_perihelion_deg * RAD_PER_DEG, Eigen::Vector3d::UnitZ());

  // Mean motion n = sqrt(mu / (a*a*a))
  // n * n = mu / (a * a * a)
  // mu = n * n * a * a * a
  // n is d/dt mean anomaly
  // mean anomaly (outside corrections) is e.mean_longitude_rad - e.longitude_of_perihelion_rad
  // n is e.mean_longitude_rad_per_s - e.longitude_of_perihelion_rad_per_s?

  // Ignoring the error corrections - could differentiate wrt time for more precision!
  double const a = semi_major_axis_meters;
  double const mean_longitude_rad_per_s = e.mean_longitude_deg_per_C * RAD_PER_DEG / (SECONDS_PER_DAY * DAYS_PER_CENTURY);
  double const longitude_of_perihelion_rad_per_s = e.longitude_of_perihelion_deg_per_C * RAD_PER_DEG / (SECONDS_PER_DAY * DAYS_PER_CENTURY);
  // Mean motion in rad/sec
  double const n = mean_longitude_rad_per_s - longitude_of_perihelion_rad_per_s;
  double const mu = n * n * a * a * a;
  double const p = semi_major_axis_meters * (1 - e.eccentricity * e.eccentricity);
  // From wikipedia article on True Anomaly
  double const true_anomaly_rad = 2 * atan2(sqrt(1+e.eccentricity) * sin(eccentric_anomaly_rad / 2), sqrt(1-e.eccentricity) * cos(eccentric_anomaly_rad / 2));
  // From Orbital Mechanics Ch 3
  double const v0 = sqrt(mu/p); // or mu * h
  double const vr = v0 * e.eccentricity * sin(true_anomaly_rad);
  double const vn = v0 * (1 + e.eccentricity * cos(true_anomaly_rad));
  Eigen::Vector3d unit_radial = r_orbital.normalized();
  Eigen::Vector3d unit_normal = Eigen::AngleAxisd(M_TAU / 4.0, Eigen::Vector3d::UnitZ()) * unit_radial; // In this frame we have Z=0 as orbital plane
  Eigen::Vector3d v_orbital = vr * unit_radial + vn * unit_normal;

  o_cart.pos = rot_inertial_frame * r_orbital;
  o_cart.vel = rot_inertial_frame * v_orbital;
}

inline void sampleOrbit(
  orEphemerisHybrid const& params,
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

#endif	/* ORMATH_H */

