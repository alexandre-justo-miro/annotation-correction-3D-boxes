from constants import EGO_FRAME_NAME, GLOBAL_FRAME_NAME
from data_loaders.BaseDataLoader import BaseDataLoader
import numpy as np
import os.path
import pandas as pd
from PIL import Image
from pytransform3d.transform_manager import NumpyTimeseriesTransform, TemporalTransformManager
from transformation_utils import get_pq


class Argoverse2DataLoader(BaseDataLoader):
    def __init__(
            self,
            path_to_sequence,
            need_transforms: bool = True,
            need_annotations: bool = True,
            need_sensor_data: bool = True,
            verbose: bool = True):
        super().__init__(path_to_sequence, need_transforms, need_annotations, need_sensor_data, verbose)

        self.path_to_corrected_annotations = os.path.join(path_to_sequence, "annotations-corrected.feather")

        self.load_all()

        # Some traits of this dataset
        self.annotations_reference_frame = EGO_FRAME_NAME
        self.sensor_data_reference_frame = EGO_FRAME_NAME
        self.sensor_data_ego_motion_compensated = True
        self.annotation_frequency_hz = 10.0

        # Categories that are static or very slow and will therefore not be corrected
        self.categories_no_correction = (
            "PEDESTRIAN", "BOLLARD", "CONSTRUCTION_CONE", "CONSTRUCTION_BARREL", "STOP_SIGN", "SIGN", "STROLLER",
            "MOBILE_PEDESTRIAN_SIGN", "OFFICIAL_SIGNALER")

    def is_argoverse2_dataset(self): return True

    def get_dataset_name(self): return "argoverse2"

    def _get_annotations_file_name(self, file_suffix: str = ""):
        """

        :param file_suffix:
        :return:
        """
        absolute_path = os.path.join(self.path_to_sequence, f"annotations{file_suffix}.feather")
        return absolute_path if os.path.exists(absolute_path) else None

    def _get_lidar_file_list(self):
        path_to_lidar_data = os.path.join(self.path_to_sequence, "sensors", "lidar")
        return sorted([os.path.join(path_to_lidar_data, d) for d in os.listdir(path_to_lidar_data)])

    def _get_camera_file_list(self):
        path_to_data = os.path.join(self.path_to_sequence, "sensors", "cameras")
        cameras = [os.path.join(path_to_data, d) for d in os.listdir(path_to_data)]
        image_list = list()
        for camera in cameras:
            for d in os.listdir(camera):
                image_list.append(os.path.join(camera, d))
        return sorted(image_list)

    def load_annotations(self, file_suffix: str = ""):

        # Clear previous data with the same file suffix before loading
        if not self.annotations.empty: self.annotations = self.annotations[self.annotations["file_suffix"] != file_suffix]

        # Special case for loading corrected annotations
        if file_suffix == "-corrected":
            self.load_corrected_annotations()
            return

        annotations_file_name = self._get_annotations_file_name(file_suffix)
        if annotations_file_name is None: return
        annotations = pd.read_feather(annotations_file_name)
        number_of_rows = annotations.shape[0]
        for attribute in ("speed_m_per_s", "yaw_rate_rad_per_s", "acceleration_m2_per_s"):
            if attribute not in annotations.columns:
                annotations[attribute] = [0.0] * number_of_rows

        # Build sample indexes, as they are not provided originally
        sorted_timestamps = sorted(annotations["timestamp_ns"].unique().tolist())
        timestamp_to_sample_index = {t: i for i, t in enumerate(sorted_timestamps)}
        annotations["sample_index"] = annotations["timestamp_ns"].map(timestamp_to_sample_index)

        annotations["file_suffix"] = [file_suffix] * number_of_rows
        self.add_rows_annotation(pd.DataFrame(annotations))

    def load_sensor_data(self):
        sensor_data = self.create_empty_sensor_data_dictionary()
        camera_data = self.create_empty_camera_data_dictionary()

        # Load lidar
        lidar_file_list = self._get_lidar_file_list()
        sample_index = 0
        for lidar_file in lidar_file_list:
            timestamp_ns = int(os.path.basename(lidar_file)[:-8])  # Get file name while removing ".feather" from it
            df = pd.read_feather(lidar_file)

            sensor_data["X"] += df["x"].tolist()
            sensor_data["Y"] += df["y"].tolist()
            sensor_data["Z"] += df["z"].tolist()
            sensor_data["deltaT"] += (df["offset_ns"] * 1e-9).tolist()

            number_points_in_sample = len(df["offset_ns"])

            sensor_data["sensor_modality"] += ["lidar"] * number_points_in_sample
            sensor_data["sensor_index"] += [0] * number_points_in_sample
            sensor_data["sample_index"] += [sample_index] * number_points_in_sample
            sensor_data["absolute_timestamp_ns"] += [timestamp_ns] * number_points_in_sample

            sample_index += 1

        # Load camera
        camera_file_list = self._get_camera_file_list()
        camera_file_list = []  # TODO: find out why PyCharm crashes when loading camera data for Argoverse2 and erase this line of code.
        sample_index = 0
        for camera_file in camera_file_list:
            timestamp_ns = int(os.path.basename(camera_file)[:-4])  # Get file name while removing ".jpg" from it

            camera_data["sensor_index"] += [camera_file.split("/")[-2]]
            camera_data["sample_index"] += [sample_index]
            camera_data["absolute_timestamp_ns"] += [timestamp_ns]
            camera_data["image"] += [Image.open(camera_file)]

            sample_index += 1

        self.sensor_data = pd.DataFrame(sensor_data)
        self.camera_data = pd.DataFrame(camera_data)

    def load_transforms(self):
        time = list()
        pqs = list()

        ego_data = pd.read_feather(os.path.join(self.path_to_sequence, "city_SE3_egovehicle.feather"))
        for index, sample_data in ego_data.iterrows():
            p = np.array([sample_data["tx_m"], sample_data["ty_m"], sample_data["tz_m"]])
            q = np.array([sample_data["qw"], sample_data["qx"], sample_data["qy"], sample_data["qz"]])
            pq = get_pq(p, q)

            time.append(int(sample_data["timestamp_ns"]))
            pqs.append(pq)

        time = np.array(time, np.int64)
        pqs = np.array(pqs, np.float32)

        tm = TemporalTransformManager()
        tm.add_transform(EGO_FRAME_NAME, GLOBAL_FRAME_NAME, NumpyTimeseriesTransform(time, pqs))

        self.transforms = tm
