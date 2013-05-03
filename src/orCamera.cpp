#include "orCamera.h"

#include "orGfx.h"

Eigen::Affine3d CameraSystem::calcCameraMatrix( int cameraId, int targetId, Vector3d up )
{
  Camera& camera = getCamera( cameraId );
  Target& target = getTarget( targetId );

  Vector3d camF = (target.m_pos - camera.m_pos).normalized();
  Vector3d camR = camF.cross(up).normalized();
  Vector3d camU = camF.cross(camR).normalized();

  Matrix3d camRot;
  camRot.col(0) = camR;
  camRot.col(1) = -camU;
  camRot.col(2) = -camF;

  Eigen::Affine3d camT;
  camT.linear() = camRot;
  camT.translation() = camera.m_pos;

  return camT.inverse();
}