import numpy as np
import copy
from pyquaternion import Quaternion

from .point_cloud import PointCloud
from .bounding_box import BoundingBox
from .pcd_utils import get_pcd_in_box_mask


def translate3d(pcd: PointCloud, box: BoundingBox, in_box_only=False):
    offset = np.random.uniform(low=-0.3, high=0.3, size=3)
    if in_box_only:
        mask = get_pcd_in_box_mask(pcd, box, offset=0.05, scale=1.0)
        if np.sum(mask) == 0:
            return
        pcd.points[:, mask] += np.expand_dims(offset, -1)
    else:
        pcd.points += np.expand_dims(offset, -1)
    box.center += offset


def flip3d(pcd: PointCloud, box: BoundingBox, axis=0):
    assert axis in [0, 1]
    pcd.points[axis, :] = -pcd.points[axis, :]
    box.center[axis] = -box.center[axis]
    box.orientation = box.orientation.inverse
    box.velocity[axis] = -box.velocity[axis]


def rotate3d(pcd: PointCloud, box: BoundingBox, in_box_only=True):
    offset = np.random.uniform(low=-10, high=10)
    if in_box_only:
        mask = get_pcd_in_box_mask(pcd, box, offset=0.05, scale=1.0)
        if np.sum(mask) == 0:
            return
        trans = - box.center
        pcd.translate(trans)
        box.translate(trans)
        quat = Quaternion(axis=[0, 0, 1], degrees=offset)
        mat = quat.rotation_matrix
        box.rotate(quat)
        pcd.points[:3, mask] = np.dot(mat, pcd.points[:3, mask])
        box.translate(-trans)
        pcd.translate(-trans)

    else:
        quat = Quaternion(axis=[0, 0, 1], degrees=offset)
        mat = quat.rotation_matrix
        box.rotate(quat)
        pcd.rotate(mat)


def apply_augmentation(pcd: PointCloud, box: BoundingBox, offset: np.array, degree: float, flip_x: bool, flip_y: bool):
    """
    Apply transformation to the box and its pcd insides. pcd should be inside the given box.
    """

    # get inverse transform
    rot_mat = box.rotation_matrix
    trans = box.center

    new_box = copy.deepcopy(box)
    new_pcd = copy.deepcopy(pcd)

    new_pcd.translate(-trans)
    new_box.translate(-trans)
    new_pcd.rotate(rot_mat.T)
    new_box.rotate(Quaternion(matrix=rot_mat.T))

    if flip_x:
        new_pcd.points[0, :] = -new_pcd.points[0, :]
        # rotate the box to make sure that the x-axis is point to the head
        new_box.rotate(Quaternion(axis=[0, 0, 1], degrees=180))
    if flip_y:
        new_pcd.points[1, :] = -new_pcd.points[1, :]

    # apply rotation
    rot_quat = Quaternion(axis=(0, 0, 1), degrees=degree)
    new_box.rotate(rot_quat)
    new_pcd.rotate(rot_quat.rotation_matrix)

    # apply translation
    new_box.translate(offset)
    new_pcd.translate(offset)

    # transform back
    new_box.rotate(Quaternion(matrix=rot_mat))
    new_pcd.rotate(rot_mat)
    new_box.translate(trans)
    new_pcd.translate(trans)
    return new_pcd, new_box


def augment3d(pcd: PointCloud, box: BoundingBox):
    mask = get_pcd_in_box_mask(pcd, box, offset=0.0, scale=1.25)
    if np.sum(mask) == 0:
        return copy.deepcopy(pcd), copy.deepcopy(box)
    in_box_pcd = PointCloud(pcd.points[:, mask])
    # np.random.seed(787)
    offset = np.random.uniform(low=-0.3, high=0.3, size=3)
    degree = np.random.uniform(low=-10, high=10)
    flip_x, flip_y = np.random.choice([True, False], size=2, replace=True)
    new_in_box_pcd, new_box = apply_augmentation(
        in_box_pcd, box, offset, degree, flip_x, flip_y)
    new_pcd = copy.deepcopy(pcd)
    new_pcd.points[:, mask] = new_in_box_pcd.points
    return new_pcd, new_box
