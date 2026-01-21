from constants import EGO_FRAME_NAME, GLOBAL_FRAME_NAME
from data_loaders.BaseDataLoader import BaseDataLoader
import numpy as np
import os.path
import pandas as pd
from PIL import Image
from pytransform3d.transform_manager import NumpyTimeseriesTransform, StaticTransform, TemporalTransformManager
from tqdm import tqdm
from transformation_utils import get_pq, get_transform_from_pq, transform_point_cloud
from truckscenes import TruckScenes
from truckscenes.utils.data_classes import LidarPointCloud


class MANTruckScenesDataLoader(BaseDataLoader):
    def __init__(
            self,
            path_to_sequence,
            need_transforms: bool = True,
            need_annotations: bool = True,
            need_sensor_data: bool = True,
            verbose: bool = True,
            dataset_version: str = "v1.0-mini",
            apply_ego_motion_compensation: bool = True,
            only_load_key_frames: bool = True):
        super().__init__(path_to_sequence, need_transforms, need_annotations, need_sensor_data, verbose)

        root_directory = os.path.dirname(self.path_to_sequence)
        scene_name = os.path.basename(self.path_to_sequence)

        self.truck_scenes = TruckScenes(dataset_version, root_directory, False)
        self.scene = next(s for s in self.truck_scenes.scene if s["name"] == scene_name)
        self.first_sample_token = self.scene["first_sample_token"]
        self.first_sample = self.truck_scenes.get("sample", self.first_sample_token)

        self.path_to_corrected_annotations = os.path.join(
            self.truck_scenes.dataroot, self.truck_scenes.version, f"sample_annotation-corrected-{scene_name}.feather")

        self.first_lidar_tokens, self.first_radar_tokens, self.first_camera_tokens = self._get_first_sensor_tokens()

        # Some traits of this dataset
        self.annotations_reference_frame = GLOBAL_FRAME_NAME
        self.sensor_data_reference_frame = EGO_FRAME_NAME
        self.sensor_data_ego_motion_compensated = apply_ego_motion_compensation
        self.annotation_frequency_hz = 2.0

        # Whether to load sensor data only from key frames or from all available frames
        self.only_load_key_frames = only_load_key_frames

        # Categories that are static or very slow and will therefore not be corrected
        self.categories_no_correction = (
            "static_object.traffic_sign", "static_object.bicycle_rack",
            "movable_object.debris", "movable_object.trafficcone", "movable_object.pushable_pullable", "movable_object.barrier",
            "vehicle.ego_trailer", "vehicle.trailer", "vehicle.construction",
            "human.pedestrian.adult", "human.pedestrian.construction_worker", "human.pedestrian.child",
            "human.pedestrian.stroller", "human.pedestrian.police_officer",
        )

        self.load_all()

    def is_man_truckscenes_dataset(self): return True

    def get_dataset_name(self): return "man-truckscenes"

    def _get_first_sensor_tokens(self):
        first_lidar_tokens, first_radar_tokens, first_camera_tokens = list(), list(), list()

        # Loop over all 16 sensors (4 cameras, 6 lidar, 6 radar)
        for sample_data_token in self.first_sample["data"].values():
            sample_data = self.truck_scenes.get("sample_data", sample_data_token)
            if sample_data["sensor_modality"] == "lidar":
                first_lidar_tokens.append(sample_data_token)
            elif sample_data["sensor_modality"] == "radar":
                first_radar_tokens.append(sample_data_token)
            elif sample_data["sensor_modality"] == "camera":
                first_camera_tokens.append(sample_data_token)

        return first_lidar_tokens, first_radar_tokens, first_camera_tokens

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
            sample = self.truck_scenes.get("sample", current_sample_token)

            ann_tokens = sample["anns"]
            for ann_token in ann_tokens:
                ann = self.truck_scenes.get("sample_annotation", ann_token)

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

        # Created to not need to query get_transform_at_time multiple times for the same timestamp. This will save time,
        # especially when needing to interpolate
        transform_map = dict()

        # Map to keep track of sample indexes
        sample_timestamp_ns_to_sample_index = dict()

        # Loop over lidars
        if self.verbose:
            print(f"INFO: {'' if self.sensor_data_ego_motion_compensated else 'not '}applying ego motion compensation to the lidar point clouds.")
            print(f"INFO: loading sensor data from key frames {'only' if self.only_load_key_frames else 'and sweeps'}.")
        for first_lidar_token in self.first_lidar_tokens:

            # Loop over samples of this lidar
            current_lidar_token = first_lidar_token
            while current_lidar_token != "":
                sample_data = self.truck_scenes.get("sample_data", current_lidar_token)
                sensor_name = sample_data["channel"]

                # Update loop variable
                current_lidar_token = sample_data["next"]

                is_key_frame = bool(sample_data["is_key_frame"])
                if self.only_load_key_frames and not is_key_frame:
                    continue

                # Load point cloud
                data_file_path = os.path.join(self.truck_scenes.dataroot, sample_data["filename"])
                point_cloud = LidarPointCloud.from_file(str(data_file_path))
                points = point_cloud.points.T  # shape after transposing: [N, 4]

                # Transform all points from sensor to ego frame of reference
                transform_point_cloud(points, self.transforms.get_transform(sensor_name, EGO_FRAME_NAME))

                # Get the closest annotation timestamp to do motion compensation to (i.e. sample timestamp)
                closest_sample = self.truck_scenes.getclosest("sample", sample_data["timestamp"])
                closest_sample_timestamp_ns = int(closest_sample["timestamp"] * 1e3)

                # Add sample timestamp and index. This assumes that the lidar samples are in chronological order.
                if closest_sample_timestamp_ns not in sample_timestamp_ns_to_sample_index:
                    sample_timestamp_ns_to_sample_index[closest_sample_timestamp_ns] = len(sample_timestamp_ns_to_sample_index)

                # Do ego motion compensation if requested
                if self.sensor_data_ego_motion_compensated:
                    ego_to_global_aggregated = list()
                    iterator = point_cloud.timestamps.flatten().tolist()
                    if self.verbose: iterator = tqdm(iterator, desc=f"Applying ego motion compensation for a sample of {sensor_name}...")
                    for t in iterator:
                        point_timestamp_ns = int(t * 1e3)
                        if point_timestamp_ns not in transform_map:
                            transform_map[point_timestamp_ns] = self.transforms.get_transform_at_time(EGO_FRAME_NAME, GLOBAL_FRAME_NAME, point_timestamp_ns)
                        ego_to_global_aggregated.append(transform_map[point_timestamp_ns])
                    ego_to_global_aggregated = np.array(ego_to_global_aggregated)  # [N, 4, 4]
                    points_homogeneous = np.vstack((points[:, :3].T, np.ones((1, np.shape(points)[0]))))  # [4, N]
                    points = np.einsum("ijk,ki->ij", ego_to_global_aggregated, points_homogeneous)[:, :3]  # [N, 3]
                    transform_point_cloud(points, self.transforms.get_transform_at_time(GLOBAL_FRAME_NAME, EGO_FRAME_NAME, closest_sample_timestamp_ns))

                # Store data
                sensor_data["X"] += points[:, 0].tolist()
                sensor_data["Y"] += points[:, 1].tolist()
                sensor_data["Z"] += points[:, 2].tolist()
                delta_t = (point_cloud.timestamps.astype(np.int64) - closest_sample_timestamp_ns * 1e-3) * 1e-6
                delta_t_list = delta_t.flatten().tolist()  # Flatten as the original array is shaped (1, N)
                sensor_data["deltaT"] += delta_t_list
                number_points_in_sample = len(delta_t_list)
                sensor_data["sensor_modality"] += ["lidar"] * number_points_in_sample
                sensor_data["sensor_index"] += [sensor_name] * number_points_in_sample
                sensor_data["sample_index"] += [sample_timestamp_ns_to_sample_index[closest_sample_timestamp_ns]] * number_points_in_sample
                sensor_data["absolute_timestamp_ns"] += [closest_sample_timestamp_ns] * number_points_in_sample

        # Loop over cameras
        for first_camera_token in self.first_camera_tokens:

            # Loop over samples of this camera
            current_camera_token = first_camera_token
            while current_camera_token != "":
                sample_data = self.truck_scenes.get("sample_data", current_camera_token)
                data_file_path = os.path.join(self.truck_scenes.dataroot, sample_data["filename"])

                # Update loop variable
                current_camera_token = sample_data["next"]

                is_key_frame = bool(sample_data["is_key_frame"])
                if self.only_load_key_frames and not is_key_frame:
                    continue

                # Get the closest annotation timestamp (i.e. sample timestamp)
                closest_sample = self.truck_scenes.getclosest("sample", sample_data["timestamp"])
                sample_timestamp_ns = int(closest_sample["timestamp"] * 1e3)

                camera_data["sensor_index"] += [sample_data["channel"]]
                camera_data["sample_index"] += [sample_timestamp_ns_to_sample_index[sample_timestamp_ns]]
                camera_data["absolute_timestamp_ns"] += [int(sample_data["timestamp"] * 1e3)]
                camera_data["image"] += [Image.open(data_file_path)]

        self.sensor_data = pd.DataFrame(sensor_data)
        self.camera_data = pd.DataFrame(camera_data)

    def load_transforms(self):
        tm = TemporalTransformManager()

        # Add static transforms. Loop over all 16 sensors (4 cameras, 6 lidar, 6 radar)
        for sample_data_token in self.first_sample["data"].values():
            sample_data = self.truck_scenes.get("sample_data", sample_data_token)
            sensor_name = sample_data["channel"]
            calibrated_sensor = self.truck_scenes.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
            p = np.array(calibrated_sensor["translation"])  # xyz
            q = np.array(calibrated_sensor["rotation"])  # wxyz quaternion
            tm.add_transform(sensor_name, EGO_FRAME_NAME, StaticTransform(get_transform_from_pq(p, q)))

        # Add dynamic transforms.
        time = list()
        pqs = list()

        # Aim to retrieve 22 seconds (from -1 to +21) at 1000 samples per second (in practice, only about 1/10 of these will exist)
        for t in [-1.0e6 + self.first_sample["timestamp"] + 0.001e6 * i for i in range(22000)]:
            ego_pose = self.truck_scenes.getclosest("ego_pose", t)
            timestamp = ego_pose["timestamp"] * 1e3  # From us to ns for compatibility with other datasets
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
