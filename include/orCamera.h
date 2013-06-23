#pragma once

#include "orStd.h"
#include "orMath.h"

#include <string>
#include <vector>

class CameraSystem {
public:

  struct Camera {
    double m_pos[3];
    float m_fov;
  };

  int numCameras() const { return (int)m_cameras.size(); }
  int makeCamera() { m_cameras.push_back(Camera()); return numCameras() - 1; }
  Camera&       getCamera(int cameraId)       { return m_cameras[cameraId]; }
  Camera const& getCamera(int cameraId) const { return m_cameras[cameraId]; }

  int nextCamera(int cameraId) { return (cameraId + 1) % numCameras(); }

  struct Target {
    double m_pos[3];
    std::string m_name;
  };

  int numTargets() const { return (int)m_targets.size(); }
  int makeTarget() { m_targets.push_back(Target()); return numTargets() - 1; }
  Target&       getTarget(int targetId)       { return m_targets[targetId]; }
  Target const& getTarget(int targetId) const { return m_targets[targetId]; }

  int nextTarget(int targetId) { return (targetId + 1) % numTargets(); }

  Eigen::Matrix4d calcScreenMatrix( int width, int height );
  Eigen::Matrix4d calcProjMatrix( int cameraId, int width, int height, double minZ, double maxZ );
  Eigen::Affine3d calcCameraMatrix( int cameraId, int targetId, Vector3d up  );

private:
  std::vector<Camera> m_cameras;
  std::vector<Target> m_targets;
}; // class CameraSystem