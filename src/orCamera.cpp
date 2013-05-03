#include "orCamera.h"

#include "orGfx.h"

Eigen::Projective3d CameraSystem::calcProjMatrix(int cameraId, int width, int height, float minZ, float maxZ)
{
  Camera& camera = getCamera( cameraId );

  double const aspect = width / (double)height;

  double const heightZ = tan(0.5 * M_TAU * camera.m_fov / 360.0);
  double const widthZ = heightZ * aspect;

  Eigen::Projective3d proj;
  Eigen::Matrix4d& projMatrix = proj.matrix();
  projMatrix.setZero(4, 4);
  projMatrix.coeffRef(0, 0) = 1 / widthZ;
  projMatrix.coeffRef(1, 1) = 1 / heightZ;
  projMatrix.coeffRef(2, 2) = -(maxZ + minZ) / (maxZ - minZ);
  projMatrix.coeffRef(2, 3) = -2*maxZ*minZ / (maxZ - minZ);
  projMatrix.coeffRef(3, 2) = -1;

  return proj;
}

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