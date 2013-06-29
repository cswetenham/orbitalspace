#include "orCamera.h"

#include "orGfx.h"

// Normalised Device Coordinates -> Screen Space Coordinates
Eigen::Matrix4d CameraSystem::calcScreenMatrix(int width, int height)
{
  Eigen::Matrix4d screenMatrix;
  // NDC: -1 to 1
  // Screen: 0 to +width or 0 to +height
  double const halfWidth = width / 2.0;
  double const halfHeight = height / 2.0;
  
  screenMatrix.setZero(4, 4);
  screenMatrix.coeffRef(0, 0) = halfWidth;
  screenMatrix.coeffRef(0, 3) = halfWidth;
  screenMatrix.coeffRef(1, 1) = -halfHeight;
  screenMatrix.coeffRef(1, 3) = halfHeight;
  screenMatrix.coeffRef(3, 3) = 1.0;

  return screenMatrix;
}

// Camera Space Coordinates -> Normalised Device Coordinates
Eigen::Matrix4d CameraSystem::calcProjMatrix(int cameraId, int width, int height, double minZ, double maxZ, double aspect)
{
  Camera& camera = getCamera( cameraId );
  double const fov_y = camera.m_fov;
  
  double const heightZ = tan(0.5 * M_TAU * fov_y / 360.0);
  double const widthZ = heightZ * aspect;

  Eigen::Matrix4d projMatrix;
  projMatrix.setZero(4, 4);
  projMatrix.coeffRef(0, 0) = 1 / widthZ;
  projMatrix.coeffRef(1, 1) = 1 / heightZ;
  projMatrix.coeffRef(2, 2) = -(maxZ + minZ) / (maxZ - minZ);
  projMatrix.coeffRef(2, 3) = -2*maxZ*minZ / (maxZ - minZ);
  projMatrix.coeffRef(3, 2) = -1;

  return projMatrix;
}

// World Space Coordinates -> Camera Space Coordinates
Eigen::Affine3d CameraSystem::calcCameraMatrix( int cameraId, int targetId, Vector3d up )
{
  Camera& camera = getCamera( cameraId );
  Target& target = getTarget( targetId );

  Vector3d const targetPos(target.m_pos);
  Vector3d const cameraPos(camera.m_pos);

  Vector3d camF = (targetPos - cameraPos).normalized();
  Vector3d camR = camF.cross(up).normalized();
  Vector3d camU = camF.cross(camR).normalized();

  Matrix3d camRot;
  camRot.col(0) = camR;
  camRot.col(1) = -camU;
  camRot.col(2) = -camF;

  Eigen::Affine3d camT;
  camT.linear() = camRot;
  camT.translation() = cameraPos;

  return camT.inverse();
}