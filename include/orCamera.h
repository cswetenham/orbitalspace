#pragma once

#include "orStd.h"
#include "orMath.h"
#include "orCore/orSystem.h"

#include <string>
#include <vector>

class CameraSystem {
public:

  struct Camera {
    Camera() : m_pos(), m_fov(0) {}
    orVec3 m_pos;
    float m_fov;
  };

  DECLARE_SYSTEM_TYPE(Camera, Cameras);

  int nextCamera(int cameraId) { return (cameraId + 1) % numCameras(); }

  struct Target {
    Target() : m_pos(), m_name() {}
    orVec3 m_pos;
    std::string m_name;
  };

  DECLARE_SYSTEM_TYPE(Target, Targets);

  int nextTarget(int targetId) { return (targetId % numTargets()) + 1; }

  Eigen::Matrix4d calcScreenMatrix( int width, int height ) const;
  Eigen::Matrix4d calcProjMatrix( int cameraId, int width, int height, double minZ, double maxZ, double aspect ) const;
  Eigen::Matrix4d calcCameraMatrix( int cameraId, int targetId, Vector3d up ) const;
}; // class CameraSystem