import numpy as np
from pytransform3d.rotations import euler_from_quaternion, quaternion_from_euler
from pytransform3d.transformations import transform, transform_from_pq


def normalize_angle(angle: float) -> float:
    """
    Normalize the angle to [-pi, pi]
    :param angle:
    :return:
    """
    normalized_angle = angle % (2.0 * np.pi)
    if normalized_angle > np.pi: normalized_angle -= 2.0 * np.pi
    return normalized_angle


def get_euler_angles(qw: float, qx: float, qy: float, qz: float):
    """

    :param qw:
    :param qx:
    :param qy:
    :param qz:
    :return: Roll, pitch, yaw; in that order
    """
    euler_angles = euler_from_quaternion([qw, qx, qy, qz], 0, 1, 2, True)
    roll, pitch, yaw = float(euler_angles[0]), float(euler_angles[1]), float(euler_angles[2])
    return roll, pitch, yaw


def get_quaternion_wxyz(roll: float, pitch: float, yaw: float):
    return quaternion_from_euler([roll, pitch, yaw], 0, 1, 2, True)


def get_quaternion_xyzw(roll: float, pitch: float, yaw: float):
    return np.roll(get_quaternion_wxyz(roll, pitch, yaw), shift=-1)


def get_pq(translation, rotation):
    """

    :param translation: (x, y, z)
    :param rotation: (qw, qx, qy, qz)
    :return: (x, y, z, qw, qx, qy, qz)
    """
    return np.hstack((translation, rotation))


def get_transform_from_pq(translation, rotation):
    """
    Generate a homogeneous 4x4 transformation matrix from translation and rotation vectors.
    :param translation: (x, y, z)
    :param rotation: (qw, qx, qy, qz)
    :return: 4x4 homogeneous transformation matrix
    """
    return transform_from_pq(get_pq(translation, rotation))


def get_transform(x: float, y: float, z: float, roll: float, pitch: float, yaw: float):
    """
    Generate a homogeneous 4x4 transformation matrix from x, y, z, roll, pitch, yaw through pytransform3d.
    :param x:
    :param y:
    :param z:
    :param roll:
    :param pitch:
    :param yaw:
    :return:
    """
    p = np.array([x, y, z])
    q = get_quaternion_wxyz(roll, pitch, yaw)
    return get_transform_from_pq(p, q)


def transform_point_cloud(point_cloud, transform_matrix) -> None:
    """
    Transforms the input point_cloud according to the given transform_matrix. This modifies the point cloud inplace.
    :param point_cloud: NumPy array of point cloud data, in which each row corresponds to a point, and the first 3
    columns correspond to X, Y, and Z coordinates, respectively.
    :param transform_matrix: Transformation matrix to be applied to the point cloud.
    :return:
    """
    nr_points = np.shape(point_cloud)[0]
    homogeneous_point_cloud_array = np.hstack((point_cloud[:, :3], np.ones((nr_points, 1))))
    point_cloud[:, :3] = transform(transform_matrix, homogeneous_point_cloud_array)[:, :3]
