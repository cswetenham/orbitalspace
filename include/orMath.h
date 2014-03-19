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

#define M_TAU      6.28318530717958647693
#define M_TAU_F    6.28318530717958647693f

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
  double&       operator[] (int i)       { return data[i]; }
  double const& operator[] (int i) const { return data[i]; }

  operator Eigen::Vector3d() const {
    return Eigen::Vector3d(data);
  }

  double data[3];
}; // struct orVec3


#endif	/* ORMATH_H */

