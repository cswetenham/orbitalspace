#include "orCamera.h"

#include "orGfx.h"

void CameraSystem::setCameraMatrix( int cameraId, int targetId, Vector3d up )
{
  Camera& camera = getCamera( cameraId );
  Target& target = getTarget( targetId );

  // TODO move to CameraSystem?

  Vector3d camF = (target.m_pos - camera.m_pos).normalized();
  Vector3d camR = camF.cross(up).normalized();
  Vector3d camU = camF.cross(camR).normalized();

  Matrix3d camMat;
  camMat.col(0) = camR;
  camMat.col(1) = -camU;
  camMat.col(2) = -camF;

  Eigen::Affine3d camT;
  camT.linear() = camMat;
  camT.translation() = camera.m_pos;

  glMultMatrix(camT.inverse());
}