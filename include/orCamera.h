#pragma once

#include "orStd.h"
#include "orMath.h"

#include <string>
#include <vector>

class CameraSystem {
public:
  struct Camera {
    Vector3d m_pos;
  };
  int numCameras() { return (int)m_cameras.size(); }
  int makeCamera() { m_cameras.push_back(Camera()); return numCameras() - 1; }
  Camera& getCamera(int cameraId) { return m_cameras[cameraId]; }
  int nextCamera(int cameraId) { return (cameraId + 1) % numCameras(); }

  struct Target {
    Vector3d m_pos;
    std::string m_name;
  };
  int numTargets() { return (int)m_targets.size(); }
  int makeTarget() { m_targets.push_back(Target()); return numTargets() - 1; }
  Target& getTarget(int targetId) { return m_targets[targetId]; }
  int nextTarget(int targetId) { return (targetId + 1) % numTargets(); }

  void setCameraMatrix( int cameraId, int targetId, Vector3d up );

private:
  std::vector<Camera> m_cameras;
  std::vector<Target> m_targets;
}; // class CameraSystem