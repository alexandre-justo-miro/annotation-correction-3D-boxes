#!/usr/bin/env python3

import argparse
from data_loaders.Argoverse2DataLoader import Argoverse2DataLoader as Argoverse2
from data_loaders.MANTruckScenesDataLoader import MANTruckScenesDataLoader as MANTruckScenes
from os.path import join
from pathlib import Path
from tqdm import tqdm

_argoverse2_sequences = (
    join('train', 'c556f8e0-a001-3586-b2cf-d3256685c39f'),
    join('train', 'fd5c6932-2ee2-3cfb-9bdc-0b30bfb33a91'),
    join('train', '58d01358-5927-36fa-9e11-d18d1dc1f4f0'),
    join('train', '81d2b40a-c579-3e9c-b520-bee26cda947d'),
    join('train', '3ca11a5e-50b2-3cc3-af7a-ce7ab02b9954'),
    join('train', '6b14d7c0-20f9-390b-af38-507a5de5998c'),
    join('train', 'f648b945-6c70-3105-bd23-9502894e37d4'),
    join('train', 'c780d53a-2d37-3cd8-9e89-530966aef53e'),
    join('val', 'd5d6f11c-3026-3e0e-9d67-c111233e22de'),
    join('val', '9a448a80-0e9a-3bf0-90f3-21750dfef55a'),
)

_man_truckscenes_sequences = (
    "scene-0044384af3d8494e913fb8b14915239e-3",
    "scene-13f4b71b1bd04a9e88747ad8f58a3f67-4",
    "scene-23160c4bff134cdd86b96ee822b4aca9-14",
    "scene-3195e3bfd1de4f9ba82f874790c7856c-15",
    "scene-3f542f89ec5241b6a4e30ca743adcf34-29",
    "scene-6be74229febb4aad9e7ae1a37217b231-4",
    "scene-a6a87db5125846bda72ccfc9931ee153-11",
    "scene-b13542304b244af585a646082c57a079-1",
    "scene-b201f6e737e34c2d87830fc1e2911516-6",
    "scene-fb64d203d417452f830ca73efed80c41-11"
)


def main():
    parser = argparse.ArgumentParser(description="annotation-correction-3D-boxes command-line interface")

    parser.add_argument("-d", "--dataset",
                        type=str,
                        help="Required: Which dataset to use for the experiments. Currently supported options: "
                        "'argoverse2', 'man-truckscenes'.",
                        required=True)
    parser.add_argument("-s", "--sequences",
                        type=str,
                        help="Optional: List of sequences to process. If not provided, all sequences in the dataset "
                             "will be processed. The sequences must be comma-separated without any whitespaces. "
                             "See a list of possible sequence names in run.py")
    tasks = parser.add_mutually_exclusive_group(required=True)
    tasks.add_argument("-v", "--visualize",
                        help="Visualize a sequence, including the corrected 3D bounding box annotations if available. "
                             "A Rerun visualization window will appear, and the sensor data and original box "
                             "annotations for the whole sequence will be shown. If the corrected boxes are available, "
                             "they will be shown too.",
                        action="store_true")
    tasks.add_argument("-c", "--correct",
                        help="Trigger the optimization process to correct 3D bounding box annotations. All selected "
                             "sequences of the selected dataset will undergo optimization, one by one. For each "
                             "sequence, the resulting annotations will then be stored at "
                             "<datasets_directory>/argoverse2/<split_name>/<sequence_name>/annotations-corrected."
                             "feather"
                             " or "
                             "<datasets_directory>/man-truckscenes/v1.0-mini/sample_annotation-corrected-<sequence_name"
                             ">.feather"
                             ", depending on the dataset.",
                        action="store_true")
    tasks.add_argument("-e", "--evaluate",
                        help="Evaluate the quality of the corrected 3D bounding box annotations. All selected "
                             "sequences of the selected dataset will undergo calculation of metrics, one by one. Some "
                             "intermediate files will be stored at results/, and the metrics will be displayed in the "
                             "form of text in the terminal and plots.",
                        action="store_true")
    parser.add_argument("-q", "--quiet",
                        help="Optional: Whether to turn off verbose mode.",
                        action="store_true")
    parser.add_argument("-b", "--base-directory",
                        type=str,
                        help="Optional: Base directory where the datasets are. Defaults to '~/datasets/'.")

    args = parser.parse_args()

    verbose = False if args.quiet else True
    base_dir = args.base_directory if args.base_directory else join(str(Path.home()), "datasets")

    data_loader = Argoverse2 if args.dataset == "argoverse2" else MANTruckScenes if args.dataset == "man-truckscenes" else None
    if data_loader is None: raise ValueError(f"Unsupported dataset {args.dataset}. Supported datasets are 'argoverse2' and 'man-truckscenes'.")

    _seqs = args.sequences.split(",") if args.sequences else _argoverse2_sequences if args.dataset == 'argoverse2' else _man_truckscenes_sequences if args.dataset == 'man-truckscenes' else None
    sequences = [join(base_dir, args.dataset, seq) for seq in _seqs]

    if args.visualize:
        if len(sequences) != 1: print("WARNING: Visualization can be done for only one sequence at a time, but multiple sequences were sent. Defaulting to the first sequence.")
        from visualize import log_sequence
        log_sequence(data_loader(path_to_sequence=sequences[0], verbose=verbose))

    if args.correct:
        from correct import run_optimization
        for seq in tqdm(sequences, desc="Optimizing for one sequence at a time..."): run_optimization(data_loader(path_to_sequence=seq, verbose=verbose), verbose=verbose)

    if args.evaluate:
        from evaluate import calculate_metrics, display_metrics, plot_distributions_and_bar_plots
        for seq in sequences: calculate_metrics(data_loader(path_to_sequence=seq, verbose=verbose))
        display_metrics(sequences, False, args.dataset)
        plot_distributions_and_bar_plots(sequences, args.dataset)


if __name__ == '__main__':
    main()
