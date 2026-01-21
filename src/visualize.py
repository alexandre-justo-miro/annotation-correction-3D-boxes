from constants import EGO_FRAME_NAME, GLOBAL_FRAME_NAME
import numpy as np
import pyarrow as pa
from pytransform3d.rotations import quaternion_from_matrix
import rerun as rr
from target_motion_compensation import target_motion_compensation
from transformation_utils import transform_point_cloud


# Eight distinguishable colors, intended for each sensor to have a different color
PALETTE = (
    (25, 25, 112),   # midnightblue
    (255, 0, 0),     # red
    (255, 255, 0),   # yellow
    (222, 184, 135), # burlywood
    (0, 250, 154),   # mediumspringgreen
    (0, 191, 255),   # deepskyblue
    (0, 0, 255),     # blue
    (255, 105, 180)  # hotpink
)


class CustomDataBatch(rr.ComponentBatchMixin):
    """A batch of dynamics data."""

    def __init__(self, dynamics, descriptor: str, data_type=pa.float32()) -> None:
        self.dynamics = dynamics
        self.descriptor = descriptor
        self.data_type = data_type

    def component_descriptor(self) -> rr.ComponentDescriptor:
        """The descriptor of the custom component."""
        return rr.ComponentDescriptor(self.descriptor)

    def as_arrow_array(self) -> pa.Array:
        """The arrow batch representing the custom component."""
        return pa.array(self.dynamics, type=self.data_type)


class CustomBoxes3D(rr.AsComponents):
    """A custom archetype that extends Rerun's builtin `Boxes3D` archetype with custom components."""

    def __init__(self, sizes=None, centers=None, quaternions=None, class_ids=None, labels=None, colors=None, speed=None,
                 yaw_rate=None, acceleration=None, track_id=None, show_labels=False) -> None:
        self.boxes3d = rr.Boxes3D(
            sizes=sizes,
            centers=centers,
            quaternions=quaternions,
            class_ids=class_ids,
            # fill_mode=rr.components.FillMode.Solid,
            labels=labels,
            colors=colors,
            show_labels=show_labels)
        self.speed = CustomDataBatch(speed, "Speed (m/s)").or_with_descriptor_overrides(
            archetype_name="user.CustomBoxes3D", archetype_field_name="speed"
        )
        self.yaw_rate = CustomDataBatch(yaw_rate, "Yaw Rate (rad/s)").or_with_descriptor_overrides(
            archetype_name="user.CustomBoxes3D", archetype_field_name="yaw_rate"
        )
        self.acceleration = CustomDataBatch(acceleration, "Acceleration (m/s^2)").or_with_descriptor_overrides(
            archetype_name="user.CustomBoxes3D", archetype_field_name="acceleration"
        )
        self.track_id = CustomDataBatch(track_id, "Track ID", pa.string()).or_with_descriptor_overrides(
            archetype_name="user.CustomBoxes3D", archetype_field_name="track_id"
        )

    def as_component_batches(self) -> list[rr.DescribedComponentBatch]:
        return (
                list(self.boxes3d.as_component_batches())  # The components from Points3D
                + [self.speed] + [self.yaw_rate] + [self.acceleration] + [self.track_id]  # Custom data
        )


class CustomPoints3D(rr.AsComponents):
    """A custom archetype that extends Rerun's builtin `Points3D` archetype with custom components."""

    def __init__(self, points, colors=None, labels=None, radii=None, delta_time=None, sensor_id=None) -> None:
        self.points3d = rr.Points3D(points, colors=colors, labels=labels, radii=radii)
        self.delta_time = CustomDataBatch(delta_time, "Delta Time (ms)").or_with_descriptor_overrides(
            archetype_name="user.CustomPoints3D", archetype_field_name="delta_time"
        )
        self.sensor_id = CustomDataBatch(sensor_id, "Sensor ID", pa.int8()).or_with_descriptor_overrides(
            archetype_name="user.CustomPoints3D", archetype_field_name="sensor_id"
        )

    def as_component_batches(self) -> list[rr.DescribedComponentBatch]:
        return (
                list(self.points3d.as_component_batches())  # The components from Points3D
                + [self.delta_time] + [self.sensor_id]  # Custom data
        )


def log_ego_pose(data_loader):

    tm = data_loader.transforms

    # Static transforms (e.g. sensor-to-ego)
    # for sensor_frame in tm.nodes:
    #     if (sensor_frame == EGO_FRAME_NAME) or (sensor_frame == GLOBAL_FRAME_NAME): continue
    #     tf = tm.get_transform(sensor_frame, EGO_FRAME_NAME)
    #     rr.log(
    #         [GLOBAL_FRAME_NAME, EGO_FRAME_NAME, sensor_frame],
    #         rr.Transform3D(
    #             translation=tf[:3, 3],
    #             rotation=rr.Quaternion(xyzw=np.roll(quaternion_from_matrix(tf[:3, :3]), -1)),  # From wxyz to xyzw
    #             relation=rr.TransformRelation.ParentFromChild
    #         ),
    #         static=True
    #     )

    first_ego_translation = None

    timestamps = sorted(tm._transforms[(EGO_FRAME_NAME, GLOBAL_FRAME_NAME)].time.tolist())
    for timestamp in timestamps[1:-1]:  # First and last transforms are extrapolated

        rr.set_time("timestamp", timestamp=np.datetime64(timestamp, "ns"))

        tf = tm.get_transform_at_time(EGO_FRAME_NAME, GLOBAL_FRAME_NAME, timestamp)

        translation = tf[:3, 3]

        # Normalize ego pose with respect to the first ego pose to avoid large numbers that cause rerun to glitch
        if first_ego_translation is None: first_ego_translation = translation

        rr.log(
            [GLOBAL_FRAME_NAME, EGO_FRAME_NAME],
            rr.Transform3D(
                translation=translation - first_ego_translation,
                rotation=rr.Quaternion(xyzw=np.roll(quaternion_from_matrix(tf[:3, :3]), -1)),  # From wxyz to xyzw
                axis_length=10.0,  # The length of the visualized axis.
                relation=rr.TransformRelation.ParentFromChild
            ),
        )

    return first_ego_translation


def log_cameras(data_loader) -> None:
    """Log camera data."""
    all_camera_data = data_loader.camera_data

    # Set up and log sensor names and lookup
    sensor_name_to_index = dict()
    unique_sensor_names = all_camera_data["sensor_index"].unique().tolist()
    for sensor_name in unique_sensor_names:
        sensor_name_to_index[sensor_name] = len(sensor_name_to_index)

    sorted_unique_timestamps = sorted(all_camera_data["absolute_timestamp_ns"].unique().tolist())
    for unique_timestamp in sorted_unique_timestamps:
        rr.set_time("timestamp", timestamp=np.datetime64(unique_timestamp, "ns"))
        camera_data = all_camera_data[all_camera_data["absolute_timestamp_ns"] == unique_timestamp]

        for i, image_row in camera_data.iterrows():
            rr.log([GLOBAL_FRAME_NAME, EGO_FRAME_NAME, image_row["sensor_index"]], rr.Image(image_row["image"]))


def log_lidar(data_loader, target_motion_compensation_file_suffix: str = "") -> None:
    """
    Log lidar data
    :param data_loader:
    :param target_motion_compensation_file_suffix: Optional file suffix for which annotation files that contain dynamics
     estimates shall be used for target motion compensation. When not provided, no such compensation will be applied and
     the original points will be visualized instead.
    :return:
    """
    print(f"Logging lidar data with{'out' if target_motion_compensation_file_suffix == '' else ''} target motion compensation.")

    all_lidar_data = data_loader.sensor_data[data_loader.sensor_data["sensor_modality"] == "lidar"]

    # Set up and log sensor names and lookup
    sensor_name_to_index = dict()
    unique_sensor_names = sorted(all_lidar_data["sensor_index"].unique().tolist())
    for sensor_name in unique_sensor_names:
        sensor_name_to_index[sensor_name] = len(sensor_name_to_index)

    sorted_unique_timestamps = sorted(all_lidar_data["absolute_timestamp_ns"].unique().tolist())
    for unique_timestamp in sorted_unique_timestamps:

        rr.set_time("timestamp", timestamp=np.datetime64(unique_timestamp, "ns"))

        lidar_data = all_lidar_data[all_lidar_data["absolute_timestamp_ns"] == unique_timestamp]

        if target_motion_compensation_file_suffix != "":
            point_cloud = lidar_data[["X", "Y", "Z", "deltaT"]].to_numpy()
            # Transform point clouds to global frame if they are not already
            if data_loader.sensor_data_reference_frame != GLOBAL_FRAME_NAME:
                point_cloud_transform = data_loader.transforms.get_transform_at_time(
                    data_loader.sensor_data_reference_frame, GLOBAL_FRAME_NAME, unique_timestamp)
                transform_point_cloud(point_cloud, point_cloud_transform)
            corrected_annotations_sample = data_loader.annotations[
                (data_loader.annotations["file_suffix"] == target_motion_compensation_file_suffix) &
                (data_loader.annotations["timestamp_ns"] == unique_timestamp)]
            tf_matrix = data_loader.transforms.get_transform_at_time(
                data_loader.annotations_reference_frame, GLOBAL_FRAME_NAME, unique_timestamp)
            list_corrected_annotations_sample = list()
            for _, ann in corrected_annotations_sample.iterrows():
                box = data_loader.get_box_from_row(ann)
                box.transform_annotations(tf_matrix)
                list_corrected_annotations_sample.append(box)
            target_motion_compensation(point_cloud, list_corrected_annotations_sample)
            # Transform point clouds back to their original frame if needed
            if data_loader.sensor_data_reference_frame != GLOBAL_FRAME_NAME:
                point_cloud_transform = data_loader.transforms.get_transform_at_time(
                    GLOBAL_FRAME_NAME, data_loader.sensor_data_reference_frame, unique_timestamp)
                transform_point_cloud(point_cloud, point_cloud_transform)
            lidar_data.loc[:, "X"] = point_cloud[:, 0].tolist()
            lidar_data.loc[:, "Y"] = point_cloud[:, 1].tolist()
            lidar_data.loc[:, "Z"] = point_cloud[:, 2].tolist()

        # Log each lidar separately
        for sensor_name in unique_sensor_names:
            sensor_index_data = lidar_data[lidar_data["sensor_index"] == sensor_name]
            points = sensor_index_data[["X", "Y", "Z"]].values
            delta_time = sensor_index_data[["deltaT"]].values.flatten()  # Flatten from shape (N, 1) to shape (N)
            sensor_index = sensor_name_to_index[sensor_name]
            rr.log([GLOBAL_FRAME_NAME, EGO_FRAME_NAME, sensor_name],
                   CustomPoints3D(points, colors=[PALETTE[sensor_index]], labels=None, radii=[0.05],
                                  delta_time=delta_time * 1e3, sensor_id=[sensor_index])
                   )


def log_annotations(data_loader, first_ego_translation, file_suffix: str = "") -> None:

    annotations = data_loader.annotations[data_loader.annotations["file_suffix"] == file_suffix]
    if annotations.empty:
        print(f"INFO: No annotations found for file suffix {file_suffix}.")
        return

    entity_path = [GLOBAL_FRAME_NAME]
    if data_loader.annotations_reference_frame == EGO_FRAME_NAME: entity_path.append(EGO_FRAME_NAME)
    entity_path.append(f"Annotations{file_suffix}")

    # Normalize ego pose with respect to the first ego pose to avoid large numbers that cause rerun to glitch when the
    # annotations are given in the global frame
    first_ego_translation = first_ego_translation if (data_loader.annotations_reference_frame == GLOBAL_FRAME_NAME) else np.zeros_like(first_ego_translation)

    # Set up and log category data and lookup
    label2id = dict()
    unique_categories = sorted(annotations["category"].unique().tolist())
    for category in unique_categories:
        label2id[category] = len(label2id)
    annotation_context = [(i, label) for label, i in label2id.items()]
    rr.log(entity_path, rr.AnnotationContext(annotation_context), static=True)

    sorted_unique_timestamps = sorted(annotations["timestamp_ns"].unique().tolist())
    for unique_timestamp in sorted_unique_timestamps:

        rr.set_time("timestamp", timestamp=np.datetime64(unique_timestamp, "ns"))

        df = annotations[annotations["timestamp_ns"] == unique_timestamp]
        sizes = list(zip(df["length_m"], df["width_m"], df["height_m"]))
        centers = list(zip(df["tx_m"] - first_ego_translation[0], df["ty_m"] - first_ego_translation[1], df["tz_m"] - first_ego_translation[2]))
        quaternions = list(zip(df["qx"], df["qy"], df["qz"], df["qw"]))
        class_ids = df["category"].map(label2id).tolist()
        speed = df["speed_m_per_s"].tolist()
        yaw_rate = df["yaw_rate_rad_per_s"].tolist()
        acceleration = df["acceleration_m2_per_s"].tolist()
        track_id = df["track_uuid"].tolist()

        # Send to rerun
        rr.log(
            entity_path,
            CustomBoxes3D(
                sizes=sizes,
                centers=centers,
                quaternions=quaternions,
                class_ids=class_ids,
                labels=None,
                colors=[(255, 0, 0) if file_suffix == "" else (0, 255, 0)],  # Corrected boxes in green, original boxes in red
                speed=speed,
                yaw_rate=yaw_rate,
                acceleration=acceleration,
                track_id=track_id,
                show_labels=False,
            ),
        )


def log_sequence(data_loader) -> None:

    sensor_space_views = [
        rr.blueprint.Spatial2DView(
            name=sensor_name,
            origin=f"{GLOBAL_FRAME_NAME}/{EGO_FRAME_NAME}/{sensor_name}",
        )
        for sensor_name in data_loader.camera_data["sensor_index"].unique().tolist()
    ]
    blueprint = rr.blueprint.Vertical(
        rr.blueprint.Horizontal(
            rr.blueprint.Spatial3DView(
                name="3D",
                origin=GLOBAL_FRAME_NAME,
                # Default for `ImagePlaneDistance` so that the pinhole frustum visualizations don't take up too much space.
                # defaults=[rr.components.ImagePlaneDistance(4.0)],
                background=rr.blueprint.archetypes.Background(color=(255, 255, 255)),  # Enforce white background
                line_grid=rr.blueprint.archetypes.LineGrid3D(visible=False),  # Remove grid lines
            ),
            rr.blueprint.TextDocumentView(origin="description", name="Description"),
            column_shares=[5, 1],
        ),
        rr.blueprint.Grid(contents=sensor_space_views, column_shares=[1] * len(sensor_space_views)),
        row_shares=[5, 1]
    )

    rr.init(application_id=f"{data_loader.get_dataset_name()} - Sequence {data_loader.get_sequence_name()}")
    rr.spawn(memory_limit="16GB")
    rr.send_blueprint(blueprint)

    first_ego_translation = log_ego_pose(data_loader)

    for file_suffix in ("", "-corrected"):
        log_annotations(data_loader, first_ego_translation, file_suffix)

    log_lidar(data_loader, "-corrected")
    # TODO: find out why rerun crashes after logging a few camera data
    # log_cameras(data_loader)
