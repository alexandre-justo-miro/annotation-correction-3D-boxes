from constants import EGO_FRAME_NAME, GLOBAL_FRAME_NAME
from data_loaders.BaseDataLoader import BaseDataLoader
import numpy as np
from nuscenes import NuScenes
import os.path
import pandas as pd
from PIL import Image
from pytransform3d.transform_manager import NumpyTimeseriesTransform, StaticTransform, TemporalTransformManager
from transformation_utils import get_pq, get_transform_from_pq, transform_point_cloud


class NuScenesDataLoader(BaseDataLoader):
    def __init__(
            self,
            path_to_sequence,
            need_transforms: bool = True,
            need_annotations: bool = True,
            need_sensor_data: bool = True,
            verbose: bool = True,
            dataset_version: str = "v1.0-mini",
            only_load_key_frames: bool = True):
        super().__init__(path_to_sequence, need_transforms, need_annotations, need_sensor_data, verbose)

        root_directory = os.path.dirname(self.path_to_sequence)
        scene_name = os.path.basename(self.path_to_sequence)

        self.nu_scenes = NuScenes(dataset_version, root_directory, verbose=verbose)
        self.scene = next(s for s in self.nu_scenes.scene if s["name"] == scene_name)
        self.first_sample_token = self.scene["first_sample_token"]
        self.first_sample = self.nu_scenes.get("sample", self.first_sample_token)

        self.path_to_corrected_annotations = os.path.abspath(os.path.join(
            os.path.pardir, "out", "corrections", self.get_dataset_name(), f"{scene_name}.feather"))

        self.first_lidar_tokens, self.first_radar_tokens, self.first_camera_tokens = self._get_first_sensor_tokens()

        # Some traits of this dataset
        self.annotations_reference_frame = GLOBAL_FRAME_NAME
        self.sensor_data_reference_frame = EGO_FRAME_NAME  # Lidar comes in sensor frame, but we will transform it to ego while loading.
        self.sensor_data_ego_motion_compensated = True  # From the nuScenes paper: "We perform motion compensation using the localization algorithm described below."
        self.annotation_frequency_hz = 2.0
        self.lidar_frequency_hz = 20.0
        self.lidar_scan_period = 1.0 / self.lidar_frequency_hz
        self.lidar_rotates_clockwise = True
        self.lidar_scan_start_angle = - np.pi

        # Whether to load sensor data only from key frames or from all available frames
        self.only_load_key_frames = only_load_key_frames

        # Categories that are static or very slow and will therefore not be corrected. Source: https://www.nuscenes.org/nuscenes#data-annotation
        self.categories_no_correction = (
            "human.pedestrian.adult",
            "human.pedestrian.child",
            "human.pedestrian.construction_worker",
            "human.pedestrian.personal_mobility",
            "human.pedestrian.police_officer",
            "human.pedestrian.stroller",
            "human.pedestrian.wheelchair",
            "movable_object.barrier",
            "movable_object.debris",
            "movable_object.pushable_pullable",
            "movable_object.trafficcone",
            "static_object.bicycle_rack",
        )

        self.load_all()

    def is_nuscenes_dataset(self): return True

    def get_dataset_name(self): return "nuscenes"

    def _get_first_sensor_tokens(self):
        first_lidar_tokens, first_radar_tokens, first_camera_tokens = list(), list(), list()

        # Loop over all 12 sensors (6 cameras, 1 lidar, 5 radar)
        for sample_data_token in self.first_sample["data"].values():
            sample_data = self.nu_scenes.get("sample_data", sample_data_token)
            if sample_data["sensor_modality"] == "lidar":
                first_lidar_tokens.append(sample_data_token)
            elif sample_data["sensor_modality"] == "radar":
                first_radar_tokens.append(sample_data_token)
            elif sample_data["sensor_modality"] == "camera":
                first_camera_tokens.append(sample_data_token)

        return first_lidar_tokens, first_radar_tokens, first_camera_tokens

    def _load_lidar_image_with_timestamps_per_point(
            self,
            bin_path: str,
            W: int = 1080,  # columns (azimuth bins)
            H: int = 32,  # rows (HDL-32E rings)
    ):
        """
        Load a nuScenes *.pcd.bin (x,y,z,intensity,ring), build a range view, and compute per‑point relative timestamps
        t_rel in seconds (<=0, 0 = end-of-sweep) artificially, as the HDL-32E lidar in nuScenes does not provide them.

        Returns:
            xyz           : (N, 3) lidar points in sensor frame of reference
            t_rel         : (N,) float32 relative timestamp in (-T, 0]
            intensity_img : (H, W) float32 sparse intensity image (NaN where empty)
            range_img     : (H, W) float32 sparse range image (NaN where empty)
            time_img      : (H, W) float32 sparse time image (NaN where empty)
        """
        # Load data from nuScenes .bin file
        scan = np.fromfile(bin_path, dtype=np.float32)
        x, y, z, intensity, ring = scan.reshape((-1, 5)).T
        xyz = scan.reshape((-1, 5))[:, :3]  # (N, 3), in sensor frame of reference

        # Calculate range
        rng = np.sqrt(x * x + y * y + z * z)  # [m], [0, +inf)

        # Calculate column indices via azimuth.
        az = np.arctan2(y, x)                        # [rad], [-pi, pi).
        az_wrapped = (az + 2 * np.pi) % (2 * np.pi)  # [rad], [0, 2pi)
        assert np.min(az) >= self.lidar_scan_start_angle
        # We make the first column start where the scan starts, and the last one where the scan ends.
        cols = (W * (az - self.lidar_scan_start_angle) / (2 * np.pi)).astype(np.int32)
        assert (not (np.min(cols) < 0)) and (not (np.max(cols) > W - 1)), \
            f"Column indices out of bounds: {np.min(cols)} to {np.max(cols)} (expected 0 to {W - 1})"

        # Fetch row indices from ring
        rows = ring.astype(np.int32)
        assert not (np.min(rows) < 0) and not (np.max(rows) > H - 1), \
            f"Ring indices out of bounds: {np.min(rows)} to {np.max(rows)} (expected 0 to {H - 1})"

        # We use the fact that the LiDAR in nuScenes scans uniformly in clockwise direction from the BEV perspective
        # (see https://github.com/nutonomy/nuscenes-devkit/issues/239), and that, from the nuScenes paper, "the
        # timestamp of the lidar scan is the time when the full rotation of the current lidar frame is achieved."
        # (Note: the lidar starts on the left side, i.e. - pi w.r.t. to its axis, which points to the right.)
        s = -1.0 if self.lidar_rotates_clockwise else 1.0  # Rotation sign, clockwise vs. CCW
        frac = ((s * (az_wrapped - self.lidar_scan_start_angle)) % (2 * np.pi)) / (2 * np.pi)  # [-], [0, 1)
        t_rel = frac * self.lidar_scan_period - self.lidar_scan_period                         # [s], (-T, 0]

        # Sparse images (by last-writer wins if collisions)
        intensity_img = np.full((H, W), np.nan, dtype=np.float32)
        range_img = np.full((H, W), np.nan, dtype=np.float32)
        time_img = np.full((H, W), np.nan, dtype=np.float32)

        # Cast vectors to matrix
        intensity_img[rows, cols] = intensity
        range_img[rows, cols] = rng
        time_img[rows, cols] = t_rel

        return xyz, t_rel.astype(np.float32), intensity_img, range_img, time_img

    def load_annotations(self, file_suffix: str = ""):

        # Clear previous data with the same file suffix before loading
        if not self.annotations.empty: self.annotations = self.annotations[self.annotations["file_suffix"] != file_suffix]

        # Special case for loading corrected annotations
        if file_suffix == "-corrected":
            self.load_corrected_annotations()
            return

        annotations = list()

        current_sample_token = self.first_sample_token
        sample_index = 0
        while current_sample_token != "":
            sample = self.nu_scenes.get("sample", current_sample_token)

            ann_tokens = sample["anns"]
            for ann_token in ann_tokens:
                ann = self.nu_scenes.get("sample_annotation", ann_token)

                x, y, z = ann["translation"]
                qw, qx, qy, qz = ann["rotation"]
                width, length, height = ann["size"]
                dynamics = ann["dynamics"] if "dynamics" in ann else None

                annotations.append({
                    "sample_index": sample_index,
                    "timestamp_ns": int(sample["timestamp"] * 1e3),
                    "track_uuid": ann["instance_token"],
                    "category": ann["category_name"],
                    "length_m": length,
                    "width_m": width,
                    "height_m": height,
                    "qw": qw,
                    "qx": qx,
                    "qy": qy,
                    "qz": qz,
                    "tx_m": x,
                    "ty_m": y,
                    "tz_m": z,
                    "speed_m_per_s": dynamics["Speed"] if dynamics is not None else 0.0,
                    "yaw_rate_rad_per_s": dynamics["HeadingRate"] if dynamics is not None else 0.0,
                    "acceleration_m2_per_s": dynamics["Acceleration"] if dynamics is not None else 0.0,
                    "file_suffix": file_suffix
                })

            current_sample_token = sample["next"]
            sample_index += 1

        if annotations: self.add_rows_annotation(pd.DataFrame(annotations))

    def load_sensor_data(self):
        sensor_data = self.create_empty_sensor_data_dictionary()
        camera_data = self.create_empty_camera_data_dictionary()

        # Map to keep track of sample indexes
        sample_timestamp_ns_to_sample_index = dict()

        # Loop over lidars
        if self.verbose:
            print(f"INFO: loading sensor data from key frames {'only' if self.only_load_key_frames else 'and sweeps'}.")
        for first_lidar_token in self.first_lidar_tokens:

            # Loop over samples of this lidar
            current_lidar_token = first_lidar_token
            while current_lidar_token != "":
                sample_data = self.nu_scenes.get("sample_data", current_lidar_token)
                sensor_name = sample_data["channel"]

                # Update loop variable
                current_lidar_token = sample_data["next"]

                is_key_frame = bool(sample_data["is_key_frame"])
                if self.only_load_key_frames and not is_key_frame:
                    continue

                # Load point cloud. We use our own function because the one in nuscenes-devkit does not load the ring
                # index and does not reconstruct timestamp per lidar point.
                data_file_path = os.path.join(self.nu_scenes.dataroot, sample_data["filename"])
                points, dt, _, _, _ = self._load_lidar_image_with_timestamps_per_point(str(data_file_path))

                # Transform all points from sensor to ego frame of reference
                transform_point_cloud(points, self.transforms.get_transform(sensor_name, EGO_FRAME_NAME))

                # Get the closest annotation timestamp to do motion compensation to (i.e. sample timestamp)
                # closest_sample = self.nu_scenes.getclosest("sample", sample_data["timestamp"])
                closest_sample_timestamp_ns = int(sample_data["timestamp"] * 1e3)  # TODO: int(closest_sample["timestamp"] * 1e3)

                # Add sample timestamp and index. This assumes that the lidar samples are in chronological order.
                if closest_sample_timestamp_ns not in sample_timestamp_ns_to_sample_index:
                    sample_timestamp_ns_to_sample_index[closest_sample_timestamp_ns] = len(sample_timestamp_ns_to_sample_index)

                # Store data
                x_list = points[:, 0].tolist()
                sensor_data["X"] += x_list
                sensor_data["Y"] += points[:, 1].tolist()
                sensor_data["Z"] += points[:, 2].tolist()
                sensor_data["deltaT"] += dt.tolist()
                number_points_in_sample = len(x_list)
                sensor_data["sensor_modality"] += [sample_data["sensor_modality"]] * number_points_in_sample
                sensor_data["sensor_index"] += [sensor_name] * number_points_in_sample
                sensor_data["sample_index"] += [sample_timestamp_ns_to_sample_index[closest_sample_timestamp_ns]] * number_points_in_sample
                sensor_data["absolute_timestamp_ns"] += [closest_sample_timestamp_ns] * number_points_in_sample

        # Loop over cameras
        for first_camera_token in self.first_camera_tokens:

            continue  # TODO: load camera data. For it we need a smart handling of sample timestamps

            # Loop over samples of this camera
            current_camera_token = first_camera_token
            while current_camera_token != "":
                sample_data = self.nu_scenes.get("sample_data", current_camera_token)
                data_file_path = os.path.join(self.nu_scenes.dataroot, sample_data["filename"])

                # Update loop variable
                current_camera_token = sample_data["next"]

                is_key_frame = bool(sample_data["is_key_frame"])
                if self.only_load_key_frames and not is_key_frame:
                    continue

                # Get the closest annotation timestamp (i.e. sample timestamp)
                # TODO: closest_sample = self.nu_scenes.getclosest("sample", sample_data["timestamp"])
                sample_timestamp_ns = int(sample_data["timestamp"] * 1e3)  # TODO: int(closest_sample["timestamp"] * 1e3)

                camera_data["sensor_index"] += [sample_data["channel"]]
                camera_data["sample_index"] += [sample_timestamp_ns_to_sample_index[sample_timestamp_ns]]
                camera_data["absolute_timestamp_ns"] += [int(sample_data["timestamp"] * 1e3)]
                camera_data["image"] += [Image.open(data_file_path)]

        self.sensor_data = pd.DataFrame(sensor_data)
        self.camera_data = pd.DataFrame(camera_data)

    def load_transforms(self):
        tm = TemporalTransformManager()

        # Add static transforms. Loop over all 12 sensors (6 cameras, 1 lidar, 5 radar)
        for sample_data_token in self.first_sample["data"].values():
            sample_data = self.nu_scenes.get("sample_data", sample_data_token)
            sensor_name = sample_data["channel"]
            calibrated_sensor = self.nu_scenes.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
            p = np.array(calibrated_sensor["translation"])  # xyz
            q = np.array(calibrated_sensor["rotation"])  # wxyz quaternion
            tm.add_transform(sensor_name, EGO_FRAME_NAME, StaticTransform(get_transform_from_pq(p, q)))

        # Add dynamic transforms.
        time = list()
        pqs = list()

        # Order the list nu_scenes.ego_pose by "timestamp" to be safe
        ego_poses = sorted(self.nu_scenes.ego_pose, key=lambda x: x["timestamp"])
        for ego_pose in ego_poses:
            timestamp = ego_pose["timestamp"] * 1e3  # From us to ns for compatibility with other datasets
            if timestamp < self.first_sample["timestamp"] * 1e3 - 0.1e9 or self.first_sample["timestamp"] * 1e3 + 20.1e9 < timestamp:  # A scene in nuScenes lasts 20 seconds.
                continue  # Prevent fetching ego poses from other sequences. TODO: find a smarter way to scan through ego poses?
            if timestamp not in time:  # Do not add the same transform more than once
                p = np.array(ego_pose["translation"])  # xyz
                q = np.array(ego_pose["rotation"])  # wxyz quaternion
                pq = get_pq(p, q)
                time.append(timestamp)
                pqs.append(pq)

        # As there are not enough future transforms after the last sample, but these are needed to interpolate, we
        # duplicate the last transform as a workaround
        idx_max = time.index(max(time))
        last_time = max(time) + 100e6
        last_pq = pqs[idx_max]
        time = time + [last_time]
        pqs = pqs + [last_pq]

        time = np.array(time, np.int64)
        pqs = np.array(pqs, np.float32)

        tm.add_transform(EGO_FRAME_NAME, GLOBAL_FRAME_NAME, NumpyTimeseriesTransform(time, pqs))

        self.transforms = tm
