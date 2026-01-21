from constants import EGO_FRAME_NAME, GLOBAL_FRAME_NAME, LOW_SPEED_THRESHOLD
import copy
import csv
import glob
import math
import matplotlib.pyplot as plt
import numpy as np
import os.path
from pandas import concat, cut, read_csv
from pytransform3d.transformations import invert_transform
import re
from target_motion_compensation import target_motion_compensation
from tqdm import tqdm
from transformation_utils import get_transform, normalize_angle, transform_point_cloud


def get_data_per_track_struct():
    data = dict()
    data['NrPointsTargetMC'] = list()
    data['EuclideanDistanceError'] = list()
    data['LongitudinalDistanceError'] = list()
    data['LateralDistanceError'] = list()
    data['LocationX'] = list()
    data['LocationY'] = list()
    data['Heading'] = list()
    data['Speed'] = list()
    data['HeadingRate'] = list()
    data['Acceleration'] = list()
    data['Length'] = list()
    data['Width'] = list()
    data['Time'] = list()
    data['DistanceToEgo'] = list()
    return data


def write_to_csv(data, dataset_name: str, sequence_name: str, label: str, track_id: str):
    if not os.path.exists("results"):
        os.mkdir("results")
    csv_file = os.path.join("results", f'{dataset_name}_{sequence_name}_{label}_{track_id}.csv')
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data.keys())
        rows = zip(*data.values())
        writer.writerows(rows)


def calculate_metrics(data_loader):
    """
    Calculates metrics, such as number of lidar points within boxes or distance to ego vehicle, both for original and
    corrected annotations of a sequence; and writes the results to two csv files (one for original and one for
    corrected).
    :param data_loader:
    :return:
    """

    # Fetch the correct version of the annotations while ensuring they are sorted by sample index
    annotations = data_loader.annotations[data_loader.annotations["file_suffix"] == ""].sort_values(by="sample_index")
    annotations_corrected = data_loader.annotations[data_loader.annotations["file_suffix"] == "-corrected"].sort_values(by="sample_index")

    if annotations_corrected.empty:
        print(f"No corrected annotations were found for sequence {data_loader.get_sequence_name()}. Not possible to calculate metrics.")
        return

    all_original_tracks_data = dict()
    all_corrected_tracks_data = dict()
    global_point_cloud_data = dict()

    for i, annotation_corrected in tqdm(
            annotations_corrected.iterrows(),
            total=annotations_corrected.shape[0],
            desc="Calculating metrics for each pair of original+corrected box in the sequence..."):

        # Common variables
        sample_index = annotation_corrected["sample_index"]
        track_id = annotation_corrected["track_uuid"]

        # Find the unique annotation that corresponds to the corrected annotation
        condition = (annotations["sample_index"] == sample_index) & (annotations["track_uuid"] == track_id)
        annotation_df = annotations[condition]
        assert annotation_df.shape[0] == 1  # The box should be unique
        annotation = annotation_df.iloc[0]

        if sample_index not in global_point_cloud_data:
            # Filter point clouds for the specified sample and convert to NumPy array
            point_cloud_df = data_loader.sensor_data[data_loader.sensor_data["sample_index"] == sample_index]
            point_cloud = point_cloud_df[["X", "Y", "Z", "deltaT"]].to_numpy()

            # Transform point clouds to global frame if they are not already
            if data_loader.sensor_data_reference_frame != GLOBAL_FRAME_NAME:
                sample_timestamp = point_cloud_df["absolute_timestamp_ns"].iat[0]
                point_cloud_transform = data_loader.transforms.get_transform_at_time(
                    data_loader.sensor_data_reference_frame, GLOBAL_FRAME_NAME, sample_timestamp)
                transform_point_cloud(point_cloud, point_cloud_transform)

            # Add point cloud to dictionary
            global_point_cloud_data[sample_index] = point_cloud

        # Fetch point cloud for this sample
        point_cloud = global_point_cloud_data[sample_index]

        # Create internal boxes
        internal_box = data_loader.get_box_from_row(annotation)
        internal_box_corrected = data_loader.get_box_from_row(annotation_corrected)

        # Transform box to global reference frame if it is not already. Note: corrected annotations are assumed to be in
        # global reference frame already and thus do not need to be transformed.
        if data_loader.annotations_reference_frame != GLOBAL_FRAME_NAME:
            annotations_frame_to_global = data_loader.transforms.get_transform_at_time(
                data_loader.annotations_reference_frame, GLOBAL_FRAME_NAME, internal_box.timestamp)
            internal_box.transform_annotations(annotations_frame_to_global)
            internal_box_corrected.transform_annotations(annotations_frame_to_global)

        # Apply target motion compensation to the lidar data
        target_motion_compensation(point_cloud, [internal_box_corrected])

        if track_id not in all_original_tracks_data:
            all_original_tracks_data[track_id] = get_data_per_track_struct()
            all_corrected_tracks_data[track_id] = get_data_per_track_struct()
        data_original = all_original_tracks_data[track_id]
        data_corrected = all_corrected_tracks_data[track_id]

        euclidean_distance_error = math.dist((internal_box.x, internal_box.y), (internal_box_corrected.x, internal_box_corrected.y))

        # Transform original box into corrected box's frame of reference and calculate longitudinal and lateral distance
        box_copy = copy.deepcopy(internal_box)
        tf = get_transform(internal_box_corrected.x, internal_box_corrected.y, internal_box_corrected.z,
                           internal_box_corrected.roll, internal_box_corrected.pitch, internal_box_corrected.yaw)
        box_copy.transform_annotations(invert_transform(tf))

        for data, box in zip((data_original, data_corrected), (internal_box, internal_box_corrected)):

            # Get timestamp for this box
            data["Time"].append(box.timestamp)

            data["EuclideanDistanceError"].append(euclidean_distance_error)
            data["LongitudinalDistanceError"].append(box_copy.x)
            data["LateralDistanceError"].append(box_copy.y)

            # Calculate number of points in the box after target motion compensation
            data["NrPointsTargetMC"].append(box.calculate_points_in_box(point_cloud))

            data["LocationX"].append(box.x)
            data["LocationY"].append(box.y)
            data["Heading"].append(normalize_angle(box.yaw))
            data["Speed"].append(box.speed)
            data["HeadingRate"].append(box.yaw_rate)
            data["Acceleration"].append(box.acceleration)
            data["Length"].append(box.length)
            data["Width"].append(box.width)

            # Calculate distance to ego
            global_to_ego_frame = data_loader.transforms.get_transform_at_time(
                GLOBAL_FRAME_NAME, EGO_FRAME_NAME, box.timestamp)
            box.transform_annotations(global_to_ego_frame)
            euclidean_distance_to_ego = math.dist((box.x, box.y), (0.0, 0.0))
            data["DistanceToEgo"].append(euclidean_distance_to_ego)

    for all_tracks_data, label in zip((all_original_tracks_data, all_corrected_tracks_data), ("original", "corrected")):

        for track_id, data in all_tracks_data.items():
            # Relativize time and convert to seconds
            start_time = data["Time"][0]
            data["Time"] = [(t - start_time) / 1e9 for t in data["Time"]]

            write_to_csv(data, data_loader.get_dataset_name(), data_loader.get_sequence_name(), label, track_id)


def get_draw_position_offset(dataset_name):
    if "argoverse2" == dataset_name: return -0.25
    elif "man-truckscenes" == dataset_name: return 0.0
    else: raise NotImplementedError


def plot_error_grouped_per_interval(df, nr_sequences: int, what_to_group_by: str, unit: str, bins, dataset_name: str):

    # Define intervals
    inf_str = r'$\infty$'
    labels = [f"[{inf_str if np.isinf(a) else a}, {inf_str if np.isinf(b) else b})" for a, b in zip(bins[:-1], bins[1:])]
    draw_positions = 1.0 + np.arange(len(labels)) + get_draw_position_offset(dataset_name)
    map_interval_to_draw_positions = {label : draw_positions[i] for i, label in enumerate(labels)}
    df['Interval'] = cut(df[what_to_group_by], bins=bins, labels=labels, right=False, include_lowest=True)

    # Calculate statistics for each interval
    grouped = df.groupby('Interval', observed=True)['EuclideanDistanceError']
    stats = grouped.agg([
        "min",
        lambda x: np.percentile(x, 5) if len(x) > 0 else np.nan,
        lambda x: np.percentile(x, 25) if len(x) > 0 else np.nan,
        "median",
        lambda x: np.percentile(x, 75) if len(x) > 0 else np.nan,
        lambda x: np.percentile(x, 95) if len(x) > 0 else np.nan,
        "max"
    ])
    stats.columns = ['Min', '5th', '25th', 'Median', '75th', '95th', 'Max']

    # Add sample count and interval
    stats['Count'] = grouped.size()
    stats["Interval"] = stats.index
    stats["DrawPosition"] = stats["Interval"].map(map_interval_to_draw_positions)

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))

    # Define bar width
    bar_width = 0.75
    index = np.arange(len(stats))

    # Plot bars from Min to Max
    ax.bar(index, stats['Max'] - stats['Min'], bar_width, bottom=stats['Min'],
           color='lightblue', edgecolor='black')

    # Add markers for percentiles and median
    for i in range(len(stats)):
        ax.plot(index[i], stats['5th'].iloc[i], 'yo', label='5th Percentile' if i == 0 else "")
        ax.plot(index[i], stats['25th'].iloc[i], 'ro', label='25th Percentile' if i == 0 else "")
        ax.plot(index[i], stats['Median'].iloc[i], 'go', label='Median' if i == 0 else "")
        ax.plot(index[i], stats['75th'].iloc[i], 'bo', label='75th Percentile' if i == 0 else "")
        ax.plot(index[i], stats['95th'].iloc[i], 'mo', label='95th Percentile' if i == 0 else "")

        # Add sample count label
        ax.text(index[i], 2, f'n={stats["Count"].iloc[i]}', ha='center', va='bottom')

    # Set x-axis labels and title
    ax.set_xlabel(f'{what_to_group_by} Interval ({unit})')
    ax.set_ylabel('Euclidean Distance Error (m)')
    ax.set_title(f'Number of Sequences: {nr_sequences}')
    ax.set_xticks(index)
    ax.set_xticklabels(stats.index, rotation=45)
    ax.legend()

    plt.tight_layout()
    plt.show()

    stats.to_csv(os.path.join("results", f"{dataset_name}_aggregated_{nr_sequences}_sequences_stats_grouped_by_{what_to_group_by}.csv"), sep=";", index=False)


def plot_distributions_and_bar_plots(sequence_name_list, dataset_name: str):

    nr_sequences = len(sequence_name_list)

    # Aggregate all files to be loaded
    files = list()
    for sequence_dir in sequence_name_list:
        pattern = f"{dataset_name}_{os.path.basename(sequence_dir)}_corrected_.*.csv"
        for file in os.listdir("results"):
            if re.match(pattern, file): files.append(os.path.join("results", file))

    # Put all data in a common dataframe
    df = None
    for file in files:
        data = read_csv(file)
        df = data if df is None else concat((df, data), ignore_index=True)

    # Filter out objects that are static or very slow, as they have not been corrected
    df = df[df["Speed"] > LOW_SPEED_THRESHOLD]

    # Plot distributions
    for label, unit in zip(("EuclideanDistanceError", "Speed", "LongitudinalDistanceError", "LateralDistanceError"), ("m", "m/s", "m", "m")):
        # Calculate spread of distributions
        std_dev = df[label].std()
        print(f"\t3 times Standard Deviation of {label} is {3*std_dev:.2f} {unit}.")

        # Dump histograms to plot in LaTeX
        hist, bin_edges = np.histogram(df[label], bins=200, density=True)
        # Write histogram data to CSV
        with open(os.path.join("results", f"histogram_{dataset_name}_{label}.csv"), mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['BinStart', 'BinEnd', 'Count'])
            for i in range(len(hist)):
                writer.writerow([bin_edges[i], bin_edges[i + 1], hist[i]])
        # Plot in Python
        df[label].plot(kind="hist", bins=200, title=f"Number of Sequences: {nr_sequences}")
        plt.xlabel(f"{label} ({unit})")
        plt.ylabel("Frequency")
        plt.show()

    bins_speed = (3, 10, 15, 20, 25, np.inf)
    bins_distance = (0, 50, 100, 150, np.inf)
    plot_error_grouped_per_interval(df, nr_sequences, "Speed", "m/s", bins_speed, dataset_name)
    plot_error_grouped_per_interval(df, nr_sequences, "DistanceToEgo", "m", bins_distance, dataset_name)

    # Dump to a common file for plotting in LaTeX
    df.to_csv(os.path.join("results", f"{dataset_name}_aggregated_{len(sequence_name_list)}_sequences.csv"), sep=";", index=False)


def plot_for_track(dataset_name: str, sequence_name: str, track_id: str):
    """
    Plots the number of points within boxes and the 6 optimization states before and after correction for the specified
    sequence and track, using the data in the saved CSV files.
    :param dataset_name: The name of the dataset being analyzed.
    :param sequence_name:
    :param track_id:
    :return:
    """

    # Plot number of points within box over time
    fig_nr_points, axes_nr_points = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
    x_label, y_label = "Time", "NrPointsTargetMC"
    for label in ("original", "corrected"):
        path_to_csv = os.path.join("results", f'{dataset_name}_{sequence_name}_{label}_{track_id}.csv')
        df = read_csv(path_to_csv)
        df.plot(ax=axes_nr_points, x=x_label, y=y_label, label=label)
        # total_points_ego_mc = np.sum(data['NrPointsEgoMC'])
        total_points_target_mc = df["NrPointsTargetMC"].sum()
        print(f"Number of ego-and-target motion compensated points inside {label} box for track {track_id}:",
              total_points_target_mc)
    axes_nr_points.set_xlabel(x_label)
    axes_nr_points.set_ylabel(y_label)
    plt.show()

    # Plot pose
    fig_pose, axes_pose = plt.subplots(nrows=3, ncols=1, figsize=(12, 6))
    x_label = "Time"
    for label in ("original", "corrected"):
        path_to_csv = os.path.join("results", f'{dataset_name}_{sequence_name}_{label}_{track_id}.csv')
        df = read_csv(path_to_csv)
        for idx, y_label in enumerate(("LocationX", "LocationY", "Heading")):
            df.plot(ax=axes_pose[idx], x=x_label, y=y_label, label=label)
            axes_pose[idx].set_xlabel(x_label)
            axes_pose[idx].set_ylabel(y_label)
    plt.show()

    # Plot dynamics
    fig_dynamics, axes_dynamics = plt.subplots(nrows=3, ncols=1, figsize=(12, 6))
    x_label = "Time"
    label = "corrected"
    path_to_csv = os.path.join("results", f'{dataset_name}_{sequence_name}_{label}_{track_id}.csv')
    df = read_csv(path_to_csv)
    for idx, y_label in enumerate(("Speed", "HeadingRate", "Acceleration")):
        df.plot(ax=axes_dynamics[idx], x=x_label, y=y_label, label=label)
        axes_dynamics[idx].set_xlabel(x_label)
        axes_dynamics[idx].set_ylabel(y_label)
    plt.show()


def display_metrics(sequence_name_list, print_info_per_track: bool, dataset_name: str):
    """
    Prints some key metrics to the terminal for the specified sequences, using the data in the saved CSV files.
    :param sequence_name_list:
    :param print_info_per_track:
    :param dataset_name:
    :return:
    """
    dataset_wide_points_original = 0
    dataset_wide_points_corrected = 0
    dataset_wide_nr_boxes = 0
    dataset_wide_speed = 0

    for sequence_path in sequence_name_list:

        sequence_name = os.path.basename(sequence_path)

        all_csv_files_original = sorted(glob.glob(f"{dataset_name}_{sequence_name}_original_*.csv", root_dir="results"))
        all_csv_files_corrected = sorted(glob.glob(f"{dataset_name}_{sequence_name}_corrected_*.csv", root_dir="results"))

        sequence_wide_points_original = 0
        sequence_wide_points_corrected = 0
        sequence_wide_nr_boxes = 0
        sequence_wide_speed = 0.0

        for csv_file_original, csv_file_corrected in zip(all_csv_files_original, all_csv_files_corrected):

            track_id = csv_file_original[-40:-4]

            path_to_csv_original = os.path.join("results", csv_file_original)
            path_to_csv_corrected = os.path.join("results", csv_file_corrected)

            df_original = read_csv(path_to_csv_original)
            df_corrected = read_csv(path_to_csv_corrected)

            # Do not take into account the boxes that were not corrected
            if df_corrected["Speed"].max() < LOW_SPEED_THRESHOLD: continue

            total_points_original = int(df_original["NrPointsTargetMC"].sum())
            total_points_corrected = int(df_corrected["NrPointsTargetMC"].sum())
            percentage_improvement = 100.0 * (total_points_corrected - total_points_original)/total_points_original if total_points_original != 0 else 0.0

            sequence_wide_points_original += total_points_original
            sequence_wide_points_corrected += total_points_corrected
            sequence_wide_nr_boxes += df_original.shape[0]
            sequence_wide_speed += df_corrected["Speed"].sum()

            if print_info_per_track:
                print(f"Track {track_id}:")
                print(f"\tNr. points:     Original: {total_points_original}. Corrected: {total_points_corrected}. Change: {percentage_improvement:.2f}%.")

        dataset_wide_points_original += sequence_wide_points_original
        dataset_wide_points_corrected += sequence_wide_points_corrected
        percentage_improvement = 100.0 * (
                    sequence_wide_points_corrected - sequence_wide_points_original) / sequence_wide_points_original if sequence_wide_points_original != 0 else 0.0

        dataset_wide_nr_boxes += sequence_wide_nr_boxes
        dataset_wide_speed += sequence_wide_speed
        average_box_speed = sequence_wide_speed / sequence_wide_nr_boxes if sequence_wide_nr_boxes != 0 else 0.0

        print(f"Sequence {sequence_name}:")
        print(f"\tNum. boxes: {sequence_wide_nr_boxes}. Avg. box speed: {average_box_speed:.2f} m/s. Original: {sequence_wide_points_original}. Corrected: {sequence_wide_points_corrected}. Change: {percentage_improvement:.2f}%")

    percentage_improvement = 100.0 * (
                dataset_wide_points_corrected - dataset_wide_points_original) / dataset_wide_points_original if dataset_wide_points_original != 0 else 0.0
    average_box_speed = dataset_wide_speed / dataset_wide_nr_boxes if dataset_wide_nr_boxes != 0 else 0.0
    print(f"Dataset {dataset_name} ({len(sequence_name_list)} sequences):")
    print(f"\tNum. boxes: {dataset_wide_nr_boxes}. Avg. box speed: {average_box_speed:.2f} m/s. Original: {dataset_wide_points_original}. Corrected: {dataset_wide_points_corrected}. Change: {percentage_improvement:.2f}%")
