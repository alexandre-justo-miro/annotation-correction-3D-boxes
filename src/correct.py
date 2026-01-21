from box_types import InternalBox
from constants import CTRA_LOW_YAW_RATE_THRESHOLD, EGO_FRAME_NAME, GLOBAL_FRAME_NAME, LOW_SPEED_THRESHOLD
import numpy as np
import pandas as pd
from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultMultiObjectiveTermination
from target_motion_compensation import speed_dependent_threshold
from tqdm import tqdm
from transformation_utils import get_euler_angles, normalize_angle, transform_point_cloud


NR_STATES_OPTIMIZATION = 6  # x, y, heading, speed, heading rate, acceleration
NR_STATES_ANNOTATION = 3    # x, y, heading


def which_points_in_box(all_points, all_base, all_cx, all_cy, infl_x = 0.0, infl_y = 0.0):
    """

    :param all_points: Shape [N_points, P, N_s, 2].
    :param all_base: For each particle P and sample N_s, (x, y) coordinates of the base corner. Shape [P, N_s, 2].
    :param all_cx: For each particle P and sample N_s, (x, y) coordinates of the corner that is longitudinally
     opposite (along the x-axis) to the base corner. Shape [P, N_s, 2].
    :param all_cy: For each particle P and sample N_s, (x, y) coordinates of the corner that is laterally opposite
     (along the y-axis) to the base corner. Shape [P, N_s, 2].
    :param infl_x: Shape [P, N_s]. Half of it will inflate the rear and the other half will inflate the front.
    :param infl_y: Shape [P, N_s]. Half of it will inflate the right and the other half will inflate the left.
    :return: Projections of points onto the x and y vectors that define the box. Mask of shape [N_points, P, N_s].
    """
    # Create vectors from -> to
    v = all_points - all_base  # [N_points, P, N_s, 2] base -> points
    v1 = all_cx - all_base     # [P, N_s, 2]           base -> front-right (box's x-axis)
    v2 = all_cy - all_base     # [P, N_s, 2]           base -> rear-left (box's y-axis)

    # Project each point's vector from the base corner to the box's x-axis and y-axis, respectively
    subscripts = "lpni,pni->lpn"  # L: lidar point index. P: particle index. N: sample index. I: vector element index (x and y)
    dot_v1 = np.einsum(subscripts, v, v1)  # [N_points, P, N_s]
    dot_v2 = np.einsum(subscripts, v, v2)  # [N_points, P, N_s]

    subscripts = "pni,pni->pn"  # P: particle index. N: sample index. I: vector element index (x and y)
    norm_v1 = np.einsum(subscripts, v1, v1)  # [P, N_s]
    norm_v2 = np.einsum(subscripts, v2, v2)  # [P, N_s]

    proj_v1 = dot_v1 / norm_v1
    proj_v2 = dot_v2 / norm_v2

    half_infl_x = 0.5 * infl_x
    half_infl_y = 0.5 * infl_y

    mask_points_inside = ((-half_infl_x <= proj_v1) & (proj_v1 <= (1.0 + half_infl_x)) &
                          (-half_infl_y <= proj_v2) & (proj_v2 <= (1.0 + half_infl_y)))  # [N_points, P, N_s]

    return proj_v1, proj_v2, mask_points_inside


class AnnotationCorrection(Problem):

    def __init__(self, z, point_clouds, constant_states, ego_positions, xl, xu, timestamps, annotation_frequency_hz: float):
        """

        :param z: Original annotations. Must be expressed in global frame.
        :param point_clouds: NumPy array with shape (N_samples, N_points, >=4), where the first 3 indexes along the
         third dimension correspond to x, y, and z point cloud coordinates, respectively; and the 4th index corresponds
         to the delta time with respect to the sample reference timestamp. N_points is the maximum number of
         points across all samples. The points must be ego but not target motion compensated. The point clouds must
         be expressed in global frame. The units must be meters and seconds.
        :param constant_states: States that are constant during optimization, such as z, roll, pitch, length, width, and height.
        :param ego_positions: Position of ego (x, y, z) along the sequence in relation to the global frame.
        :param xl: Lower bounds of the search space.
        :param xu: Upper bounds of the search space.
        :param timestamps:
        :param annotation_frequency_hz: At which frequency in Hz the samples are annotated.
        """
        self.total_nr_boxes = int(len(z) / NR_STATES_ANNOTATION)

        super().__init__(n_var=NR_STATES_OPTIMIZATION * self.total_nr_boxes, n_obj=1, xl=xl, xu=xu)
        self.z = z.reshape(-1, NR_STATES_ANNOTATION, self.total_nr_boxes, order='F')  # Reshape for efficient handling in NumPy
        self.dt_seconds = 1e-9 * np.diff(timestamps)
        self.ego_positions = ego_positions.reshape(self.total_nr_boxes, 3)  # [N_sf, 3]

        # Optimization weights
        self.inv_cov_ctra_cost = 10 * annotation_frequency_hz * np.identity(6)
        self.inv_cov_points_ratio_cost = 1e3  # Normalized point ratios of up to 1 will end up at 1000 cost points.
        self.inv_cov_points_dist_cost = 1e3  # Normalized distances of up to 1 will end up at 1000 cost points.
        self.inv_cov_boxes_dist_cost = 1e-4  # Distances of about 200 m will end up at 0.02 cost points.

        self.normalize_angles = np.vectorize(normalize_angle)

        # Constant vectors can be created once to save computations later
        constants_view = constant_states.reshape(6, -1, order='F')
        all_half_l = 0.5 * constants_view[3, :]
        all_half_w = 0.5 * constants_view[4, :]
        self.all_base = np.stack((- all_half_l, - all_half_w, np.ones_like(all_half_l)), axis=0)  # [3, N]
        self.all_cx = np.stack((+ all_half_l, - all_half_w, np.ones_like(all_half_l)), axis=0)  # [3, N]
        self.all_cy = np.stack((- all_half_l, + all_half_w, np.ones_like(all_half_l)), axis=0)  # [3, N]
        self.size_normalization_factor = np.maximum(all_half_l, all_half_w)  # Largest among half L and half W

        # Create a point cloud view with the adequate dimensions. The original point_clouds shape is [N_sf, N_points, 4]
        self.pc_view = np.expand_dims(point_clouds[:, :, :2].transpose((1, 0, 2)), axis=1)  # [N_points, P=1, N_sf, 2]
        self.pc_all_dts = np.expand_dims(point_clouds[:, :, 3].transpose((1, 0)), axis=1)  # [N_points, P=1, N_sf]

    def _evaluate(self, x, out, *args, **kwargs):

        # Reshape the state vector for efficient handling in NumPy
        # P is the number of particles. N_s is the number of samples that have a box for this track.
        x_view = x.reshape(-1, NR_STATES_OPTIMIZATION, self.total_nr_boxes, order='F')  # [P, 6, N_sf]
        out["F"] = self.ctra_cost(x_view) + self.point_cloud_cost(x_view) + self.distance_to_ego_cost(x_view)

    def ctra_cost(self, x):
        """
        For every state vector in x (at index i), compute a CTRA prediction to the timestep immediately after in time
         (at index i+1) and compute the difference between this prediction and the corresponding state vector (at i+1).
        :param x: The aggregated NumPy array containing all boxes' states to be optimized. Must have 3 axes.
        :return:
        """
        # Split them into previous and next states
        x_prev = x[:, :, :-1]  # [P, 6, N_sf-1]
        x_next = x[:, :, 1:]  # [P, 6, N_sf-1]

        # Basic variables
        dt = np.tile(self.dt_seconds, (np.shape(x)[0], 1))  # Repeat the dt row as many times as candidate solutions there are
        th1, s1, w1, a1 = x_prev[:, 2, :], x_prev[:, 3, :], x_prev[:, 4, :], x_prev[:, 5, :]
        th2, s2, w2, a2 = x_next[:, 2, :], x_next[:, 3, :], x_next[:, 4, :], x_next[:, 5, :]
        dth, ds = np.multiply(w1, dt), np.multiply(a1, dt)
        q1x, q1y, q2x, q2y = x_prev[:, 0, :], x_prev[:, 1, :], x_next[:, 0, :], x_next[:, 1, :]
        c_th_z, s_th_z = np.cos(th1 + dth), np.sin(th1 + dth)

        # Calculate error
        appr = w1 < CTRA_LOW_YAW_RATE_THRESHOLD  # Whether to approximate to avoid numerical issues for small yaw rates
        e1, e2 = np.zeros_like(th1), np.zeros_like(th1)  # [P, N_sf-1]
        e1[appr] = -0.5 * (dt[appr] * (ds[appr] + 2 * s1[appr]) * np.sin(th1[appr]) +
                           2 * q1y[appr] - 2 * q2y[appr]) * s_th_z[appr] - 0.5 * (
                    dt[appr] * (ds[appr] + 2 * s1[appr]) * np.cos(th1[appr]) +
                    2 * q1x[appr] - 2 * q2x[appr]) * c_th_z[appr]
        e2[appr] = -0.5 * (dt[appr] * (ds[appr] + 2 * s1[appr]) * np.sin(th1[appr]) + 2 * q1y[appr] -
                           2 * q2y[appr]) * c_th_z[appr] + 0.5 * (
                    dt[appr] * (ds[appr] + 2 * s1[appr]) * np.cos(th1[appr]) + 2 * q1x[appr] -
                    2 * q2x[appr]) * s_th_z[appr]
        e1[~appr] = (a1[~appr] * np.cos(dth[~appr]) / np.power(w1[~appr], 2) - a1[~appr] / np.power(w1[~appr], 2) -
                     q1x[~appr] * c_th_z[~appr] + q2x[~appr] * c_th_z[~appr] - q1y[~appr] * s_th_z[~appr] +
                     q2y[~appr] * s_th_z[~appr] - s1[~appr] * np.sin(dth[~appr]) / w1[~appr])
        e2[~appr] = (ds[~appr] / w1[~appr] - a1[~appr] * np.sin(dth[~appr]) / np.power(w1[~appr], 2) +
                     q1x[~appr] * s_th_z[~appr] - q2x[~appr] * s_th_z[~appr] - q1y[~appr] * c_th_z[~appr] +
                     q2y[~appr] * c_th_z[~appr] - s1[~appr] * np.cos(dth[~appr]) / w1[~appr] + s1[~appr] / w1[~appr])
        new_shape = (-1, self.total_nr_boxes-1)
        error_vector = np.stack((e1.reshape(new_shape),
                                  e2.reshape(new_shape),
                                  self.normalize_angles(th2 - th1 - dth).reshape(new_shape),
                                  (s2 - s1 - ds).reshape(new_shape),
                                  (w2 - w1).reshape(new_shape),
                                  (a2 - a1).reshape(new_shape)),
                                axis=1)  # [P, 6, N_sf-1]

        # Weigh error with covariances and return
        return np.einsum('...ij,jk,...ki->...', error_vector.transpose((0, 2, 1)), self.inv_cov_ctra_cost, error_vector)

    def point_cloud_cost(self, x):
        """
        Calculate point cloud cost, consisting of three parts: ratio of points within boxes, mean distance from points
        that are within boxes to the box centroid, and having at least one point within each box.
        :param x: The aggregated NumPy array containing all boxes' states to be optimized. Must have 3 axes.
        :return:
        """
        p = np.shape(x)[0]

        all_x = x[:, 0, :]  # [P, N_sf]
        all_y = x[:, 1, :]  # [P, N_sf]
        all_yaw = x[:, 2, :]  # [P, N_sf]
        all_speed = x[:, 3, :]  # [P, N_sf]
        all_yaw_rate = x[:, 4, :]  # [P, N_sf]
        all_acceleration = x[:, 5, :]  # [P, N_sf]

        all_cos = np.cos(all_yaw)  # [P, N_sf]
        all_sin = np.sin(all_yaw)  # [P, N_sf]

        transform_matrix = np.array([
            [all_cos, -all_sin, all_x],
            [all_sin, all_cos, all_y],
            [np.zeros_like(all_x), np.zeros_like(all_x), np.ones_like(all_x)]]).transpose((2, 3, 0, 1))  # [3, 3, P, N_sf] -> [P, N_sf, 3, 3]

        # Transform all corners from the box local frame to global frame
        subscripts = "pnij,jn->pni"  # P: particle index. N: sample index. I and J: matrix element indexes.
        all_base = np.einsum(subscripts, transform_matrix, self.all_base)[:, :, :2]  # [P, N_sf, 2]
        all_cx = np.einsum(subscripts, transform_matrix, self.all_cx)[:, :, :2]      # [P, N_sf, 2]
        all_cy = np.einsum(subscripts, transform_matrix, self.all_cy)[:, :, :2]      # [P, N_sf, 2]

        # Association: get which points are inside an inflated box.
        _, _, mask_points_inside_inflated = which_points_in_box(
            self.pc_view, all_base, all_cx, all_cy, 2.0 * speed_dependent_threshold(all_speed), 1.0)  # [N_points, P, N_sf]
        all_dt = np.ma.masked_where(~mask_points_inside_inflated, np.tile(self.pc_all_dts, (1, p, 1)))

        # Do target motion compensation using the CTRA formulas
        appr = (all_yaw_rate < CTRA_LOW_YAW_RATE_THRESHOLD)  # [P, N_sf]
        all_appr = np.tile((all_yaw_rate < CTRA_LOW_YAW_RATE_THRESHOLD), (np.shape(all_dt)[0], 1, 1))  # [N_points, P, N_sf]
        dx, dy = np.zeros_like(all_dt), np.zeros_like(all_dt)  # [N_points, P, N_sf]
        subscripts = "lpn,pn->lpn"
        # Approximated formulas
        masked_dt = np.ma.masked_where(~all_appr, all_dt)
        dd = (np.einsum(subscripts, masked_dt, np.ma.masked_where(~appr, all_speed)) +
              np.einsum(subscripts, np.power(masked_dt, 2.0) / 2.0, np.ma.masked_where(~appr, all_acceleration)))
        dx[all_appr] = np.einsum(subscripts, dd, np.ma.masked_where(~appr, all_cos))[all_appr]
        dy[all_appr] = np.einsum(subscripts, dd, np.ma.masked_where(~appr, all_sin))[all_appr]
        # Nominal formulas
        masked_dt = np.ma.masked_where(all_appr, all_dt)
        m_sin = np.ma.masked_where(appr, all_sin)
        m_cos = np.ma.masked_where(appr, all_cos)
        m_s = np.ma.masked_where(appr, all_speed)
        m_w = np.ma.masked_where(appr, all_yaw_rate)
        m_a = np.ma.masked_where(appr, all_acceleration)
        skplus1 = m_s + np.einsum(subscripts, masked_dt, m_a)  # [N_points, P, N_sf]
        thkplus1 = np.ma.masked_where(appr, all_yaw) + np.einsum(subscripts, masked_dt, m_w)  # [N_points, P, N_sf]
        sw = m_s * m_w
        skplus1w = np.einsum(subscripts, skplus1, m_w)
        w2 = np.power(m_w, 2.0)
        dx[~all_appr] = (((skplus1w * np.sin(thkplus1)
                        + np.einsum(subscripts, np.cos(thkplus1), m_a)
                        - sw * m_sin
                        - m_a * m_cos)
                         / w2))[~all_appr]
        dy[~all_appr] = (((- skplus1w * np.cos(thkplus1)
                        + np.einsum(subscripts, np.sin(thkplus1), m_a)
                        + sw * m_cos
                        - m_a * m_sin)
                         / w2))[~all_appr]
        # Apply target motion compensation while not modifying the original point clouds
        target_mc_pc_x = np.ma.masked_where(~mask_points_inside_inflated, np.tile(self.pc_view[..., 0], (1, p, 1))) - dx  # Note the negative sign
        target_mc_pc_y = np.ma.masked_where(~mask_points_inside_inflated, np.tile(self.pc_view[..., 1], (1, p, 1))) - dy  # Note the negative sign
        target_mc_pc = np.stack((target_mc_pc_x, target_mc_pc_y), axis=-1)

        # Calculate proximity to any face of the box. Aim for 0. Lower than 0 means out of the box
        proj_v1, proj_v2, mask_points_inside = which_points_in_box(target_mc_pc, all_base, all_cx, all_cy)  # [N_points, P, N_sf]
        m_proj_v1 = np.ma.masked_invalid(proj_v1, copy=False)
        m_proj_v2 = np.ma.masked_invalid(proj_v2, copy=False)
        proximity_front_rear = 2.0 * np.minimum(m_proj_v1, 1.0 - m_proj_v1)  # [N_points, P, N_sf]
        proximity_sides = 2.0 * np.minimum(m_proj_v2, 1.0 - m_proj_v2)  # [N_points, P, N_sf]
        proximity = np.power(np.minimum(proximity_front_rear, proximity_sides), 2.0)  # [N_points, P, N_sf]
        mean_proximity_per_sf = np.mean(proximity, axis=0)  # [P, N_sf]
        mean_proximity = np.mean(mean_proximity_per_sf, axis=1).data  # Unmask the result

        # Calculate ratio of points inside the box per sample so that samples with low amounts of points are not
        # under-represented
        mask_associated_points = ~np.isnan(target_mc_pc[:, :, :, 0])  # [N_points, P, N_sf]
        mask_existing_points_inside = mask_associated_points & mask_points_inside  # [N_points, P, N_sf]
        nr_pts_inside_per_sf = np.sum(mask_existing_points_inside, axis=0)  # [P, N_sf]
        total_pts_per_sf = np.sum(mask_associated_points, axis=0)  # [P, N_sf]
        mask_at_least_one_point_per_sf = total_pts_per_sf > 0.5  # [P, N_sf]
        if np.sum(mask_at_least_one_point_per_sf) > 0.5:
            mean_ratio_points_inside = np.mean(
                np.ma.masked_where(~mask_at_least_one_point_per_sf, nr_pts_inside_per_sf) / np.ma.masked_where(
                    ~mask_at_least_one_point_per_sf, total_pts_per_sf), axis=1)  # [P]
        else:
            mean_ratio_points_inside = np.zeros(p)  # [P]
        ratio_points_inside = 1.0 - mean_ratio_points_inside.data  # [P]

        return (self.inv_cov_points_ratio_cost * ratio_points_inside
                + self.inv_cov_points_dist_cost * mean_proximity)

    def distance_to_ego_cost(self, x):
        all_x = x[:, 0, :]  # [P, N_sf]
        all_y = x[:, 1, :]  # [P, N_sf]

        # Distance from boxes to ego
        delta_boxes_to_ego = np.stack((all_x - self.ego_positions[:, 0], all_y - self.ego_positions[:, 1]),
                                      axis=-1)  # [P, N_sf, 2]
        mean_distance_boxes_to_ego = np.mean(np.linalg.norm(delta_boxes_to_ego, axis=-1), axis=-1)  # [P]

        return - self.inv_cov_boxes_dist_cost * mean_distance_boxes_to_ego


def calculate_speed(dl, track_df):
    assert track_df.shape[0] > 1

    # Collect annotations in global frame
    x_global, y_global, z_global = list(), list(), list()
    for i, row in track_df.iterrows():
        box = dl.get_box_from_row(row)
        if dl.annotations_reference_frame != GLOBAL_FRAME_NAME:
            tf = dl.transforms.get_transform_at_time(dl.annotations_reference_frame, GLOBAL_FRAME_NAME, box.timestamp)
            box.transform_annotations(tf)
        x_global.append(box.x)
        y_global.append(box.y)
        z_global.append(box.z)

    # Calculate speed
    p = np.array((x_global, y_global, z_global)).T  # Shape [N, 3]
    t = track_df["timestamp_ns"].to_numpy()  # Shape [N]
    p_diff_m = p[1:, :] - p[:-1, :]
    t_diff_s = (t[1:] - t[:-1]) * 1e-9
    delta_distance_per_box = np.linalg.norm(p_diff_m, axis=1)  # Shape [N]
    speed_per_box = delta_distance_per_box / t_diff_s

    # Update dataframe. Note that the first speed is duplicated so that the dimensions match
    track_df.loc[:, "speed_m_per_s"] = [speed_per_box[0]] + speed_per_box.tolist()


def setup_data(data_loader, track_id: str, slack_states):
    """
    Prepare the necessary data for the optimization problem.
    :param data_loader: Object containing all sensor data, annotations, and transformations from a sequence.
    :param track_id:
    :param slack_states: Iterable containing 6 elements, indicating how much the search space is extended from the
     initial guess, both in lower and upper bounds, for the variables x, y, yaw, speed, yaw rate, and acceleration;
     respectively.
    :return: all_annotations: a one-dimensional NumPy array that has length number_boxes x 3 (x, y, yaw).
     final_point_cloud_array: a three-dimensional NumPy array that has shape (number_samples, max_number_points, 4),
     which is padded with NaN to keep consistent dimensions.
     all_constant_states: a one-dimensional NumPy array that has length number_boxes x 6 (z, roll, pitch, length, width, height).
     all_ego_positions: a one-dimensional NumPy array that has length number_boxes x 3 (x, y, z).
     xl, xu, and x0: one-dimensional NumPy arrays that have length number_boxes x 6 (x, y, yaw, speed, yaw rate, acceleration).
     timestamps: a one-dimensional NumPy array that has length number_boxes.
    """
    all_annotations, all_point_clouds, all_constant_states, all_ego_positions, xl, xu, x0, timestamps = list(), list(), list(), list(), list(), list(), list(), list()

    # Fetch the correct version of the annotations
    annotations = data_loader.annotations[data_loader.annotations["file_suffix"] == ""]

    # Filter annotations from the specified track ID while ensuring they are sorted by sample index
    filtered_annotations = annotations[annotations["track_uuid"] == track_id].sort_values(by="sample_index")

    # Error handling, mostly for the special case when there were fewer than 2 boxes in the optimization and therefore
    # no corrected boxes were created
    if filtered_annotations.empty: raise RuntimeError(f"No annotations found for track {track_id} while setting up data.")

    # Naive pre-calculation of speeds
    if filtered_annotations.shape[0] > 1: calculate_speed(data_loader, filtered_annotations)

    for i, annotation in filtered_annotations.iterrows():

        # Filter lidar point clouds for the specified sample and convert to NumPy array
        conditions = (
                (data_loader.sensor_data["sample_index"] == annotation["sample_index"]) &
                (data_loader.sensor_data["sensor_modality"] == "lidar"))
        point_cloud_df = data_loader.sensor_data[conditions]
        point_cloud = point_cloud_df[["X", "Y", "Z", "deltaT"]].to_numpy()

        # Transform point clouds to global frame if they are not already
        if data_loader.sensor_data_reference_frame != GLOBAL_FRAME_NAME:
            sample_timestamp = point_cloud_df["absolute_timestamp_ns"].iat[0]
            point_cloud_transform = data_loader.transforms.get_transform_at_time(
                data_loader.sensor_data_reference_frame, GLOBAL_FRAME_NAME, sample_timestamp)
            transform_point_cloud(point_cloud, point_cloud_transform)

        roll, pitch, yaw = get_euler_angles(annotation["qw"], annotation["qx"], annotation["qy"], annotation["qz"])
        internal_box = InternalBox(
            annotation["tx_m"], annotation["ty_m"], annotation["tz_m"],
            roll, pitch, yaw,
            annotation["speed_m_per_s"], annotation["yaw_rate_rad_per_s"], annotation["acceleration_m2_per_s"],
            annotation["length_m"], annotation["width_m"], annotation["height_m"],
            annotation["timestamp_ns"], annotation["category"], annotation["track_uuid"])

        # Transform from ego vehicle reference frame to global if the dataset is specified as such
        if data_loader.annotations_reference_frame != GLOBAL_FRAME_NAME:
            annotations_frame_to_global = data_loader.transforms.get_transform_at_time(
                data_loader.annotations_reference_frame, GLOBAL_FRAME_NAME, internal_box.timestamp)
            internal_box.transform_annotations(annotations_frame_to_global)

        # Fetch ego position
        tf = data_loader.transforms.get_transform_at_time(EGO_FRAME_NAME, GLOBAL_FRAME_NAME, internal_box.timestamp)
        translation = tf[:3, 3]
        all_ego_positions += [translation[0], translation[1], translation[2]]

        # Fill lists with the necessary data
        initial_states = [internal_box.x, internal_box.y, internal_box.yaw, internal_box.speed,
                          internal_box.yaw_rate, internal_box.acceleration]
        all_annotations += initial_states[:3]
        all_constant_states += [internal_box.z, internal_box.roll, internal_box.pitch,
                                internal_box.length, internal_box.width, internal_box.height]
        x0 += initial_states
        xl += [initial_states[i] - slack_states[i] for i in range(len(initial_states))]
        xu += [initial_states[i] + slack_states[i] for i in range(len(initial_states))]
        timestamps.append(internal_box.timestamp)

        # Reduce the point clouds for computational efficiency. Deflate the z component to prevent the ground points
        # from interfering
        sdt = speed_dependent_threshold(internal_box.speed)
        mask = internal_box.which_points_in_box(point_cloud, (sdt, sdt), (0.5, 0.5), (-0.2, 0.0))
        all_point_clouds.append(point_cloud[mask, :])

    # Post-process the point clouds. Put them into a NumPy array of which the unavailable points are NaN
    max_nr_points = 0
    for pc in all_point_clouds:
        nr_points = np.shape(pc)[0]
        if nr_points > max_nr_points:
            max_nr_points = nr_points
    final_point_cloud_array = np.full((len(all_point_clouds), max_nr_points, np.shape(all_point_clouds[0])[1]), np.nan)
    for i, pc in enumerate(all_point_clouds):
        nr_points = np.shape(pc)[0]
        final_point_cloud_array[i, :nr_points, :] = pc

    return np.array(all_annotations), final_point_cloud_array, np.array(all_constant_states), np.array(all_ego_positions), np.array(xl), np.array(xu), np.array(x0), np.array(timestamps, dtype=np.int64)


def update_corrected_annotations(x_opt, all_constant_states, timestamps, data_loader, track_id: str):
    """
    Transform the corrected annotations back to the original annotation's reference frame and dump to a *.feather file.
    :param x_opt: Optimal state vector.
    :param data_loader:
    :param track_id:
    :return:
    """
    # Clear previous corrected data of this track
    condition_to_remove = (data_loader.annotations["file_suffix"] == "-corrected") & (data_loader.annotations["track_uuid"] == track_id)
    data_loader.annotations = data_loader.annotations[~condition_to_remove]

    # Aggregate the corrected annotations
    corrected_annotations = list()
    nr_boxes = int(x_opt.size/NR_STATES_OPTIMIZATION)
    for box_counter in range(nr_boxes):

        x = x_opt[box_counter * NR_STATES_OPTIMIZATION + 0]
        y = x_opt[box_counter * NR_STATES_OPTIMIZATION + 1]
        z = all_constant_states[box_counter * 6 + 0]
        roll = all_constant_states[box_counter * 6 + 1]
        pitch = all_constant_states[box_counter * 6 + 2]
        yaw = x_opt[box_counter * NR_STATES_OPTIMIZATION + 2]
        speed = x_opt[box_counter * NR_STATES_OPTIMIZATION + 3]
        yaw_rate = x_opt[box_counter * NR_STATES_OPTIMIZATION + 4]
        acceleration = x_opt[box_counter * NR_STATES_OPTIMIZATION + 5]
        l = all_constant_states[box_counter * 6 + 3]
        w = all_constant_states[box_counter * 6 + 4]
        h = all_constant_states[box_counter * 6 + 5]
        timestamp = timestamps[box_counter]

        condition_original_box = (data_loader.annotations["track_uuid"] == track_id) & (data_loader.annotations["timestamp_ns"] == timestamp)
        original_box = data_loader.annotations[condition_original_box]
        assert original_box.shape[0] == 1  # The box should be unique
        sample_index = original_box.iloc[0]["sample_index"]
        category = original_box.iloc[0]["category"]

        internal_box = InternalBox(x, y, z, roll, pitch, yaw, speed, yaw_rate, acceleration, l, w, h,
                                        timestamp, category, track_id)

        # Transform back to annotations reference frame if the dataset is specified as such
        if data_loader.annotations_reference_frame != GLOBAL_FRAME_NAME:
            global_to_annotations_frame = data_loader.transforms.get_transform_at_time(
                GLOBAL_FRAME_NAME, data_loader.annotations_reference_frame, internal_box.timestamp)
            internal_box.transform_annotations(global_to_annotations_frame)

        quaternion_xyzw = internal_box.get_quaternion()
        corrected_annotations.append({
            "sample_index": sample_index,
            "timestamp_ns": internal_box.timestamp,
            "track_uuid": internal_box.track_id,
            "category": internal_box.category,
            "length_m": internal_box.length,
            "width_m": internal_box.width,
            "height_m": internal_box.height,
            "qw": quaternion_xyzw[3],
            "qx": quaternion_xyzw[0],
            "qy": quaternion_xyzw[1],
            "qz": quaternion_xyzw[2],
            "tx_m": internal_box.x,
            "ty_m": internal_box.y,
            "tz_m": internal_box.z,
            "speed_m_per_s": internal_box.speed,
            "yaw_rate_rad_per_s": internal_box.yaw_rate,
            "acceleration_m2_per_s": internal_box.acceleration,
            "file_suffix": "-corrected"
        })

    if corrected_annotations: data_loader.add_rows_annotation(pd.DataFrame(corrected_annotations))


def solve_for_all_tracks(data_loader, slack_states, termination: DefaultMultiObjectiveTermination, verbose: bool) -> None:
    """
    Solves the optimization problem for all tracks, one track at a time.
    :param data_loader: Object containing all sensor data, annotations, and transformations from a sequence.
    :param slack_states: Iterable containing 6 elements, indicating how much the search space is extended from the
     initial guess, both in lower and upper bounds, for the variables x, y, yaw, speed, yaw rate, and acceleration;
     respectively.
    :param termination: Criteria for termination of the optimization.
    :param verbose: Whether to output detailed information about the optimization progress (True) or not (False).
    :return:
    """
    # Filter out static or very slow objects, as their distortion will be negligible
    dynamic_annotations = data_loader.annotations[~data_loader.annotations["category"].isin(data_loader.categories_no_correction)]

    # Process each track independently
    unique_track_ids = sorted(dynamic_annotations["track_uuid"].unique().tolist())

    for track_id in tqdm(unique_track_ids, desc=f"Optimizing for one track at a time", leave=False):

        try:
            all_annotations, all_point_clouds, all_constant_states, all_ego_positions, xl, xu, x0, timestamps = setup_data(
                data_loader, track_id, slack_states)
        except RuntimeError as e:
            print(e)
            print(f"Skipping processing for track {track_id}.")
            continue

        # At least two boxes are needed for the CTRA motion model
        if not (np.shape(timestamps)[0] > 1):
            print(f"Track {track_id} does not have at least two boxes, skipping.")
            continue

        # Do not optimize if the track speed is low, as lidar distortion will barely happen there. This allows to skip
        # (nearly) static objects such as parked vehicles or pedestrians and save computations
        speed = x0.reshape(NR_STATES_OPTIMIZATION, -1, order='F')[3, :]
        low_speed = np.max(np.abs(speed)) < LOW_SPEED_THRESHOLD
        if low_speed:
            print(f"Track {track_id} has a highest speed lower than {LOW_SPEED_THRESHOLD} m/s, skipping.")
            continue

        problem = AnnotationCorrection(
            all_annotations, all_point_clouds, all_constant_states, all_ego_positions, xl, xu, timestamps, data_loader.annotation_frequency_hz)
        algorithm = PatternSearch(x0=x0)
        # algorithm = PSO(sampling=x0_warm_start)
        try:
            res = minimize(problem, algorithm, termination, verbose=verbose, seed=1)
            # print("Best solution found: \nX = %s\nF = %s" % (res.X.reshape((-1, NR_STATES_OPTIMIZATION)), res.F))

            # Update results in the data loader
            update_corrected_annotations(res.X, all_constant_states, timestamps, data_loader, track_id)
        except Exception as e:
            print(e)
            print(f"Optimization failed for track {track_id}. Continuing with next track without updating the results in the data loader.")


def run_optimization(data_loader, verbose: bool) -> None:
    """
    Prepares files and data and solves the optimization problem. Results are written to JSON files in batch_dir.
    :param data_loader: Object containing all sensor data, annotations, and transformations from a sequence.
    :param verbose: Whether to output detailed information about the optimization progress (True) or not (False).
    :return:
    """
    # Define termination criteria
    termination = DefaultMultiObjectiveTermination(xtol=1e-8, cvtol=1e-6, ftol=0.0025, period=30, n_max_gen=1e5, n_max_evals=1e6)

    # Solve the full optimization problem
    slack_states = [5.0, 5.0, np.pi / 16, 40.0, np.pi / 8, 20.0]
    solve_for_all_tracks(data_loader, slack_states, termination, verbose)

    # Dump feather file with all results
    data_loader.save_corrected_annotations_to_file()
