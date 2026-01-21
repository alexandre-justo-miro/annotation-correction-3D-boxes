from abc import ABC, abstractmethod
from box_types import InternalBox
import os.path
import pandas as pd
from time import time
from transformation_utils import get_euler_angles


class BaseDataLoader(ABC):
    def __init__(
            self,
            path_to_sequence,
            need_transforms: bool,
            need_annotations: bool,
            need_sensor_data: bool,
            verbose: bool):
        self.annotations = pd.DataFrame(columns=[
            "sample_index",
            "timestamp_ns",
            "track_uuid",
            "category",
            "length_m",
            "width_m",
            "height_m",
            "qw",
            "qx",
            "qy",
            "qz",
            "tx_m",
            "ty_m",
            "tz_m",
            "file_suffix"  # original, corrected...
        ])

        self.path_to_sequence = path_to_sequence
        self.need_transforms = need_transforms
        self.need_annotations = need_annotations
        self.need_sensor_data = need_sensor_data
        self.verbose = verbose

        self.sensor_data = None
        self.camera_data = None

        self.transforms = None

        self.path_to_corrected_annotations = None

    def get_sequence_name(self): return os.path.basename(self.path_to_sequence)

    @abstractmethod
    def get_dataset_name(self): pass

    @abstractmethod
    def load_annotations(self, file_suffix: str = ""): pass

    @abstractmethod
    def load_sensor_data(self): pass

    @abstractmethod
    def load_transforms(self): pass

    def get_box_from_row(self, row) -> InternalBox:
        roll, pitch, yaw = get_euler_angles(row["qw"], row["qx"], row["qy"], row["qz"])
        internal_box = InternalBox(
            row["tx_m"], row["ty_m"], row["tz_m"],
            roll, pitch, yaw,
            row["speed_m_per_s"], row["yaw_rate_rad_per_s"], row["acceleration_m2_per_s"],
            row["length_m"], row["width_m"], row["height_m"],
            row["timestamp_ns"], row["category"], row["track_uuid"])
        return internal_box

    def save_corrected_annotations_to_file(self):
        corrected_annotations = self.annotations[self.annotations["file_suffix"] == "-corrected"]
        corrected_annotations.reset_index(drop=True).to_feather(self.path_to_corrected_annotations)

    def add_rows_annotation(self, new_rows):
        if self.annotations.empty:
            self.annotations = new_rows
        else:
            self.annotations = pd.concat((self.annotations, new_rows), ignore_index=True)

    def create_empty_sensor_data_dictionary(self):
        return {
            "sensor_modality": list(),
            "sensor_index": list(),
            "sample_index": list(),
            "absolute_timestamp_ns": list(),
            "X": list(),
            "Y": list(),
            "Z": list(),
            "deltaT": list()
        }

    def create_empty_camera_data_dictionary(self):
        return {
            "sensor_index": list(),
            "sample_index": list(),
            "absolute_timestamp_ns": list(),
            "image": list()
        }

    def load_all(self):
        if self.verbose: print("Loading data...")
        start_time = time()
        if self.need_transforms: self.load_transforms()
        if self.need_annotations:
            self.load_annotations()
            try:
                self.load_annotations("-corrected")
            except FileNotFoundError:
                if self.verbose: print("INFO: No corrected annotations were found")
        if self.need_sensor_data: self.load_sensor_data()
        if self.verbose: print(f"Loaded data in {time() - start_time:.2f} seconds.")

    def load_corrected_annotations(self):
        corrected_annotations = pd.read_feather(self.path_to_corrected_annotations)
        if self.annotations.empty:
            self.annotations = corrected_annotations
        else:
            self.annotations = pd.concat([self.annotations, corrected_annotations], ignore_index=True)

    def is_argoverse2_dataset(self): return False
    def is_man_truckscenes_dataset(self): return False
