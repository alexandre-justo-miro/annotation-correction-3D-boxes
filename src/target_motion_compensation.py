from constants import CTRA_LOW_YAW_RATE_THRESHOLD
import numpy as np


def speed_dependent_threshold(speed):
    # For longitudinal box inflation: do speed-dependent, from 2.0 m when s=0.0 m/s to e.g. 12.0 m when s=50.0 m/s
    return 1.0 + 0.1 * np.abs(speed)  # [P, N_sf]


def ctra_motion(dt: float, th: float, s: float, w: float, a: float, approximation_threshold: float = CTRA_LOW_YAW_RATE_THRESHOLD):
    """
    Calculate the variation in global x and y coordinates for CTRA motion
    :param dt: [s] Time elapsed. Can be scalar or 1D vector
    :param th: [rad] Original yaw angle expressed in global coordinates. Must be a scalar
    :param s: [m/s] Linear speed. Must be a scalar
    :param w: [rad/s] Angular speed. Must be a scalar
    :param a: [m/s^2] Linear acceleration. Must be a scalar
    :param approximation_threshold: [rad/s] Angular speed threshold below which the approximated formulas are used for numerical stability
    :return: dx, dy, dth, ds: Variation in global x and y coordinates and in heading and speed, respectively
    """
    s_th = np.sin(th)
    c_th = np.cos(th)

    if w < approximation_threshold:
        tmp = s * dt + (a * np.power(dt, 2)) / 2
        dx             = tmp * c_th
        dy             = tmp * s_th
    else:
        wt   = w * dt
        vw   = s * w
        awt  = a * wt
        thwt = th + wt
        w2   = np.power(w, 2)

        s_thwt = np.sin(thwt)
        c_thwt = np.cos(thwt)

        dx = ((vw + awt) * s_thwt + a * c_thwt - vw * s_th - a * c_th) / w2
        dy = ((-vw - awt) * c_thwt + a * s_thwt + vw * c_th - a * s_th) / w2

    dth = w * dt
    ds = a * dt

    return dx, dy, dth, ds


def target_motion_compensation(point_cloud, corrected_annotations):
    """
    Do target motion compensation for track_id's movement in one sample. This modifies point_cloud by reference inplace.
    :param point_cloud: NumPy array. Must be expressed in global frame. Must have shape (N, >=4), where the first 3
     columns represent x, y, and z, respectively; and the 4th column represents delta time.
    :param corrected_annotations: Iterable of corrected annotations containing dynamics for different Track IDs within
     the sample. Must be expressed in global frame.
    :return:
    """

    # Prepare matrix with as many columns as number of points
    dx_dy_global = np.full((2, np.shape(point_cloud)[0]), np.nan)

    for internal_box in corrected_annotations:
        # Some objects may not have dynamics. For example, traffic signs. In that case, skip the object
        # if "Dynamics" not in box: continue

        # Find which points need to be target motion compensated
        # Assume 40 m/s (144 km/h) as maximum speed. Over 100 ms, this becomes 4.0 m offset along x-axis. Add a 2.0 m offset on all axes
        sdt = speed_dependent_threshold(internal_box.speed)
        mask = internal_box.which_points_in_box(point_cloud, (sdt, sdt), (1.0, 1.0), (1.0, 1.0))

        # Apply CTRA motion for each point at its delta timestamp with respect to the sample's timestamp
        for i in list(np.where(mask == True)[0]):
            dt = point_cloud[i, 3]
            dx_dy_global[0, i], dx_dy_global[1, i], _, _ = ctra_motion(-dt, internal_box.yaw, internal_box.speed, internal_box.yaw_rate, internal_box.acceleration)

    # Add delta x and y wherever it is applicable
    mask_to_be_compensated = np.bitwise_not(np.isnan(dx_dy_global[0, :]))
    point_cloud[mask_to_be_compensated, :2] += dx_dy_global[:, mask_to_be_compensated].T
