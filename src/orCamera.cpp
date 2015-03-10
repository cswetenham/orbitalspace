#include "orCamera.h"

#include "orGfx.h"

// TODO just an idea for rendering - instead of zooming out camera by moving it
// far away, scale down the entire world? Scale it around camera target.
// Third option is to modify FOV for a zooming effect.
// Whatever the method, it would also be nice to automatically pick z near/far planes
// based on what we send to the renderer - maybe track near (>0) / far (>0) for
// each orbit for now? Later could include near/far for spheres too since they
// can be much closer to the camera than their respective orbit.

// Worked through the math and found that scaling the world around the camera has
// no effect as long as the view frustum is scaled in the same way; if the view
// frustum isn't scaled then it's equivalent to changing the FOV and z planes.

// As for picking z planes, a better approach would be to follow the example of
// the blog posts I found (saved in Reference/Orbital Space) and use a log(z)
// buffer.

// For reference there is also the open-source Celestia project, which renders
// the solar system without issues and I've not yet looked into how.

// Normalised Device Coordinates -> Screen Space Coordinates
Eigen::Matrix4d CameraSystem::calcScreenMatrix(int width, int height) const
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
  screenMatrix.coeffRef(2, 2) = 1.0;
  screenMatrix.coeffRef(3, 3) = 1.0;

  return screenMatrix;
}

// Camera Space Coordinates -> Normalised Device Coordinates
Eigen::Matrix4d CameraSystem::calcProjMatrix(int cameraId, int width, int height, double minZ, double maxZ, double aspect) const
{
  // Projection matrix (GL_PROJECTION)
  // Simplified for symmetric case

  Camera const& camera = getCamera( cameraId );
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
Eigen::Matrix4d CameraSystem::calcCameraMatrix( int cameraId, int targetId, Vector3d up ) const
{
  // Camera matrix (GL_MODELVIEW)

  Camera const& camera = getCamera( cameraId );
  Target const& target = getTarget( targetId );

  Vector3d const targetPos(target.m_pos);
  Vector3d const cameraPos(camera.m_pos);

  Vector3d camF = (targetPos - cameraPos).normalized();
  Vector3d camR = camF.cross(up).normalized();
  Vector3d camU = camF.cross(camR).normalized();

  Eigen::Matrix3d camRot;
  camRot.col(0) = camR;
  camRot.col(1) = -camU;
  camRot.col(2) = -camF;

  Eigen::Affine3d camT;
  camT.linear() = camRot;
  // NOTE now we subtract the camera position from everything before sending it to the render system, instead
  // camT.translation() = cameraPos;

  return camT.inverse().matrix();
}
