import numpy as np
from pytransform3d.rotations import euler_from_matrix
from pytransform3d.transformations import transform, invert_transform
from transformation_utils import get_quaternion_xyzw, get_transform, transform_point_cloud


class MinimalInternalBox:
    """
    Minimal box in 2D created for efficiency purposes.
    """
    def __init__(self, x: float, y: float, yaw: float, length: float, width: float):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.size = np.array([length, width])
        self.tf_box_to_global = get_transform(x, y, 0.0, 0.0, 0.0, yaw)

        # Corners that define the box: base (rear-right), cx (front-right), and cy (rear-left)
        self.base = transform(self.tf_box_to_global, np.array([- self.size[0], - self.size[1], 0.0, 2.0]) / 2.0)[:2]
        self.cx = transform(self.tf_box_to_global, np.array([+ self.size[0], - self.size[1], 0.0, 2.0]) / 2.0)[:2]
        self.cy = transform(self.tf_box_to_global, np.array([- self.size[0], + self.size[1], 0.0, 2.0]) / 2.0)[:2]

        # Vectors that define the box
        self.vx = self.cx - self.base  # Vector base -> front-right-bottom (box's x-axis)
        self.vy = self.cy - self.base  # Vector base -> rear-left-bottom (box's y-axis)

        # Pre-calculate the constant vx_vx and vy_vy products for efficiency
        self.vx_vx = np.dot(self.vx, self.vx)
        self.vy_vy = np.dot(self.vy, self.vy)

    def which_points_in_box_vector_projection_2d(self, point_cloud: np.ndarray):
        """
        Calculate which points are within the box. Inspired from
        https://math.stackexchange.com/questions/1472049/check-if-a-point-is-inside-a-rectangular-shaped-area-3d/1552579#1552579
        :param point_cloud: NumPy array with shape (N_points, 2), where each row corresponds to a point and the 2
        columns correspond to x and y coordinates, respectively, of each point.
        :return: A mask with shape (N_points) telling which points are within the box (True) or outside (False).
        """
        # Vectors from the base corner to each point
        v = point_cloud - self.base

        # Project these vectors onto the box's x and y axes
        dot_vx = np.dot(v, self.vx)
        dot_vy = np.dot(v, self.vy)

        # Check if the projection falls within the box's bounds
        mask_vx = np.logical_and(0.0 <= dot_vx, dot_vx <= self.vx_vx)
        mask_vy = np.logical_and(0.0 <= dot_vy, dot_vy <= self.vy_vy)

        return mask_vx & mask_vy

    def calculate_points_in_box_2d(self, point_cloud: np.ndarray) -> int:
        """
        Calculate how many points are within the box.
        :param point_cloud: NumPy array with shape (N_points, 2), where each row corresponds to a point and the 2
        columns correspond to x and y coordinates, respectively, of each point.
        :return: Number of points within the box.
        """
        return np.sum(self.which_points_in_box_vector_projection_2d(point_cloud))


class InternalBox(MinimalInternalBox):
    """
    A complete bounding box in 3D.
    """
    def __init__(self, x: float, y: float, z: float, roll: float, pitch: float, yaw: float,
                 speed, yaw_rate, acceleration, length: float, width: float, height: float,
                 timestamp: int, category: str, track_id: str):

        super().__init__(x, y, yaw, length, width)

        self.z = z
        self.roll = roll
        self.pitch = pitch
        self.speed = speed if speed is not None else 0.0
        self.yaw_rate = yaw_rate if yaw_rate is not None else 0.0
        self.acceleration = acceleration if acceleration is not None else 0.0
        self.length = length
        self.width = width
        self.height = height
        self.has_dynamics = (speed is not None)
        self.timestamp = timestamp
        self.category = category
        self.track_id = track_id

        self.center = np.array([x, y, z])
        self.size = np.array([length, width, height])

        # Transforms
        self.tf_box_to_global = get_transform(x, y, z, roll, pitch, yaw)
        self.tf_global_to_box = invert_transform(self.tf_box_to_global)

        # Corners that define the box
        self.base, self.cx, self.cy, self.cz, self.cxy, self.cxz, self.cyz, self.cxyz = self._get_corners_global_frame()

        # Vectors that define the box
        self.vx = self.cx - self.base  # Vector base -> front-right-bottom (box's x-axis)
        self.vy = self.cy - self.base  # Vector base -> rear-left-bottom (box's y-axis)
        self.vz = self.cz - self.base  # Vector base -> rear-right-top (box's z-axis)

        # Pre-calculate the constant vx_vx, vy_vy and vz_vz products for efficiency
        self.vx_vx = np.dot(self.vx, self.vx)
        self.vy_vy = np.dot(self.vy, self.vy)
        self.vz_vz = np.dot(self.vz, self.vz)

    def get_quaternion(self):
        """
        Convenience function to get a quaternion representation of the box's orientation.
        :return: A (qx, qy, qz, qw) quaternion.
        """
        return get_quaternion_xyzw(self.roll, self.pitch, self.yaw)

    def _get_corners_global_frame(self):
        """
        Private method used to initialize the box's corners in global frame.
        :return: A tuple of 8 corners, in this order, following the right-hand rule: rear-right-bottom,
        front-right-bottom, rear-left-bottom, rear-right-top, front-left-bottom, front-right-top, rear-left-top,
        front-left-top.
        """
        base = np.array([- self.size[0], - self.size[1], - self.size[2], 2.0]) / 2.0
        cx = np.array([+ self.size[0], - self.size[1], - self.size[2], 2.0]) / 2.0
        cy = np.array([- self.size[0], + self.size[1], - self.size[2], 2.0]) / 2.0
        cz = np.array([- self.size[0], - self.size[1], + self.size[2], 2.0]) / 2.0
        cxy = np.array([+ self.size[0], + self.size[1], - self.size[2], 2.0]) / 2.0
        cxz = np.array([+ self.size[0], - self.size[1], + self.size[2], 2.0]) / 2.0
        cyz = np.array([- self.size[0], + self.size[1], + self.size[2], 2.0]) / 2.0
        cxyz = np.array([+ self.size[0], + self.size[1], + self.size[2], 2.0]) / 2.0
        return (transform(self.tf_box_to_global, base)[:3],
                transform(self.tf_box_to_global, cx)[:3],
                transform(self.tf_box_to_global, cy)[:3],
                transform(self.tf_box_to_global, cz)[:3],
                transform(self.tf_box_to_global, cxy)[:3],
                transform(self.tf_box_to_global, cxz)[:3],
                transform(self.tf_box_to_global, cyz)[:3],
                transform(self.tf_box_to_global, cxyz)[:3])

    def transform_annotations(self, transform_matrix) -> None:
        """
        Transforms the annotations according to the given transform_matrix inplace.
        :param transform_matrix: Transformation matrix to be applied to the annotations.
        :return:
        """
        # Transform centroid
        homogeneous_centroid = np.array([self.x, self.y, self.z, 1.0])
        transformed_centroid = transform(transform_matrix, homogeneous_centroid)[:3]
        self.x = transformed_centroid[0]
        self.y = transformed_centroid[1]
        self.z = transformed_centroid[2]

        # Rotate
        orientation_offset = euler_from_matrix(transform_matrix[:3, :3], 0, 1, 2, True)
        self.roll += orientation_offset[0]
        self.pitch += orientation_offset[1]
        self.yaw += orientation_offset[2]

        # Update other attributes
        self.center = np.array([self.x, self.y, self.z])
        self.tf_box_to_global = get_transform(self.x, self.y, self.z, self.roll, self.pitch, self.yaw)
        self.tf_global_to_box = invert_transform(self.tf_box_to_global)
        self.base, self.cx, self.cy, self.cz, self.cxy, self.cxz, self.cyz, self.cxyz = self._get_corners_global_frame()
        self.vx = self.cx - self.base
        self.vy = self.cy - self.base
        self.vz = self.cz - self.base
        self.vx_vx = np.dot(self.vx, self.vx)
        self.vy_vy = np.dot(self.vy, self.vy)
        self.vz_vz = np.dot(self.vz, self.vz)

    def which_points_in_box_vector_projection(self, point_cloud: np.ndarray):
        """
        Calculate which points are within the box. Inspired from
        https://math.stackexchange.com/questions/1472049/check-if-a-point-is-inside-a-rectangular-shaped-area-3d/1552579#1552579
        :param point_cloud: NumPy array with shape (N_points, 3), where each row corresponds to a point and the 3
        columns correspond to x, y, and z coordinates, respectively, of each point.
        :return: A mask with shape (N_points,) telling which points are within the box (True) or outside (False).
        """
        # Vectors from the base corner to each point
        v = point_cloud - self.base

        # Project these vectors onto the box's x, y, and z axes
        dot_vx = np.dot(v, self.vx)
        dot_vy = np.dot(v, self.vy)
        dot_vz = np.dot(v, self.vz)

        # Check if the projection falls within the box's bounds
        mask_vx = np.logical_and(0.0 <= dot_vx, dot_vx <= self.vx_vx)
        mask_vy = np.logical_and(0.0 <= dot_vy, dot_vy <= self.vy_vy)
        mask_vz = np.logical_and(0.0 <= dot_vz, dot_vz <= self.vz_vz)

        return mask_vx & mask_vy & mask_vz

    def which_points_in_box(self, point_cloud: np.ndarray, inflation_l = (0.0, 0.0), inflation_w = (0.0, 0.0), inflation_h = (0.0, 0.0)):
        """
        Calculate which points are within the inflated box by transforming the points to the box's reference frame and
        thresholding. This variant is slower than the vector projection variant, but allows for inflating the box.
        :param point_cloud: Expressed in the same frame as self (it does not matter which one). Must have shape
        (N, >=3), where the first 3 columns represent x, y, and z, respectively.
        :param inflation_l: Tuple of 2 elements for length of inflation in meters. The first element will inflate the rear and the second element will inflate the front.
        :param inflation_w: Tuple of 2 elements for width of inflation in meters. The first element will inflate the right and the second element will inflate the left.
        :param inflation_h: Tuple of 2 elements for height of inflation in meters. The first element will inflate the bottom and the second element will inflate the top.
        :return: A boolean mask with shape (N) for which points are within the inflated box.
        """
        # Transform point cloud to the target reference frame
        transform_point_cloud(point_cloud, self.tf_global_to_box)

        # Set box boundaries
        ll, lu = 0.5 * self.length + inflation_l[0], 0.5 * self.length + inflation_l[1]
        wl, wu = 0.5 * self.width + inflation_w[0], 0.5 * self.width + inflation_w[1]
        hl, hu = 0.5 * self.height + inflation_h[0], 0.5 * self.height + inflation_h[1]

        # Get which points are inside the inflated box
        pc_x, pc_y, pc_z = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2]
        mask = (-ll < pc_x) & (pc_x < lu) & (-wl < pc_y) & (pc_y < wu) & (-hl < pc_z) & (pc_z < hu)

        # Transform the point cloud back to the original frame
        transform_point_cloud(point_cloud, self.tf_box_to_global)

        return mask

    def calculate_points_in_box(self, point_cloud: np.ndarray) -> int:
        """
        Calculate how many points are within the box.
        :param point_cloud: NumPy array with shape (N_points, >=3), where each row corresponds to a point and the 3
        columns correspond to x, y, and z coordinates, respectively, of each point.
        :return: Number of points within the box.
        """
        mask = self.which_points_in_box_vector_projection(point_cloud[:, :3])
        return np.sum(mask)
