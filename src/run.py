#!/usr/bin/env python3

from os.path import abspath, join, pardir
from pathlib import Path


def get_optimal_worker_count():
    # 1. Try to get physical core count (excludes hyper-threading)
    # psutil is the best way, but might need installation: pip install psutil
    import psutil
    physical_cores = psutil.cpu_count(logical=False)
    if physical_cores:
         # Leave 1 core free for system responsiveness
        usable_cores = max(1, physical_cores - 1)
        print(f"Using {usable_cores} physical cores (out of {physical_cores})")
        return usable_cores

    # 2. Fallback: Standard Library (returns logical CPUs, e.g., 16)
    # If we only have logical count, divide by 2 as a safe heuristic for physical cores
    from os import cpu_count
    logical_cores = cpu_count() or 1
    usable_cores = max(1, logical_cores // 2 - 1)  # Leave 1 core free for system responsiveness
    print(f"Falling back to Standard Library logical CPUs calculation and diving by 2 heuristically. Using {usable_cores} physical cores")
    return usable_cores


def set_environment_variables_for_single_threading():
    from os import environ
    environ["OMP_NUM_THREADS"] = "1"
    environ["MKL_NUM_THREADS"] = "1"
    environ["OPENBLAS_NUM_THREADS"] = "1"
    environ["VECLIB_MAXIMUM_THREADS"] = "1"
    environ["NUMEXPR_NUM_THREADS"] = "1"


def get_argoverse2_sequences():
    argoverse2_sequences = (
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
    return argoverse2_sequences


def get_man_truckscenes_sequences():
    man_truckscenes_sequences_file = abspath(join(pardir, "dat", "man-truckscenes-trainval.txt"))
    with open(man_truckscenes_sequences_file) as file:
        man_truckscenes_sequences = tuple(line.strip() for line in file if line.strip())
        return man_truckscenes_sequences


def _correct_worker(args):
    set_environment_variables_for_single_threading()  # Set common env vars to stop NumPy/Scipy from exploding with threads
    data_loader_cls, seq, verbose, log_dir = args
    log_path = join(log_dir, f"correct-{Path(seq).name}.log")
    from contextlib import redirect_stderr, redirect_stdout
    with open(log_path, "w") as file, redirect_stdout(file), redirect_stderr(file):
        try:
            from correct import run_optimization
            run_optimization(data_loader_cls(path_to_sequence=seq, verbose=verbose), verbose=verbose)
        except Exception as e: print(f"Error occurred with sequence {seq}: {e}")


def _evaluate_worker(args):
    set_environment_variables_for_single_threading()  # Set common env vars to stop NumPy/Scipy from exploding with threads
    data_loader_cls, seq, verbose, log_dir = args
    log_path = join(log_dir, f"evaluate-{Path(seq).name}.log")
    from contextlib import redirect_stderr, redirect_stdout
    with open(log_path, "w") as file, redirect_stdout(file), redirect_stderr(file):
        try:
            from evaluate import calculate_metrics
            calculate_metrics(data_loader_cls(path_to_sequence=seq, verbose=verbose))
        except Exception as e: print(f"Error occurred with sequence {seq}: {e}")


def _run_in_parallel(worker_func, worker_args, num_workers, desc):
    from multiprocessing import Pool
    with Pool(processes=num_workers) as pool:
        from tqdm import tqdm
        with tqdm(total=len(worker_args), desc=desc) as pbar:
            for _ in pool.imap(worker_func, worker_args): pbar.update(1)


def _parse_arguments():
    import argparse

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
                            "out/corrections/<dataset-name>/<sequence-name>.feather.",
                       action="store_true")
    tasks.add_argument("-e", "--evaluate",
                       help="Evaluate the quality of the corrected 3D bounding box annotations. All selected "
                            "sequences of the selected dataset will undergo calculation of metrics, one by one. Some "
                            "intermediate files will be stored at out/evaluations/, and the metrics will be displayed "
                            "in the form of text in the terminal and plots.",
                       action="store_true")
    parser.add_argument("-q", "--quiet",
                        help="Optional: Whether to turn off verbose mode.",
                        action="store_true")
    parser.add_argument("-b", "--base-directory",
                        type=str,
                        help="Optional: Base directory where the datasets are. Defaults to '~/datasets/'.")

    return parser.parse_args()


def main():
    # Set environment variables to ensure single-threaded execution of NumPy/Scipy in each worker process, which is crucial
    # for avoiding performance degradation due to thread contention when running multiple processes in parallel. This must
    # be done before importing any libraries that use these variables (like NumPy, SciPy, or the data loaders), and it must
    # be done in the main process as well as in each worker process.
    set_environment_variables_for_single_threading()

    args = _parse_arguments()

    verbose = False if args.quiet else True
    base_dir = abspath(args.base_directory if args.base_directory else join(str(Path.home()), "datasets"))
    if verbose: print(f"Using base directory {base_dir}")

    from data_loaders.Argoverse2DataLoader import Argoverse2DataLoader as Argoverse2
    from data_loaders.MANTruckScenesDataLoader import MANTruckScenesDataLoader as MANTruckScenes
    data_loader = Argoverse2 if args.dataset == "argoverse2" else MANTruckScenes if args.dataset == "man-truckscenes" else None
    if data_loader is None: raise ValueError(f"Unsupported dataset {args.dataset}. Supported datasets are 'argoverse2' and 'man-truckscenes'.")

    _seqs = args.sequences.split(",") if args.sequences else get_argoverse2_sequences() if args.dataset == 'argoverse2' else get_man_truckscenes_sequences() if args.dataset == 'man-truckscenes' else None
    sequences = [join(base_dir, args.dataset, seq) for seq in _seqs]

    if args.visualize:
        if len(sequences) != 1: print("WARNING: Visualization can be done for only one sequence at a time, but multiple sequences were sent. Defaulting to the first sequence.")
        from visualize import log_sequence
        log_sequence(data_loader(path_to_sequence=sequences[0], verbose=verbose))
        return

    log_dir = abspath(join(pardir, "out", "logs"))
    from os import makedirs; makedirs(log_dir, exist_ok=True)
    if verbose: print(f"Writing stdout and stderr logs for each parallel sequence to {log_dir}")

    worker_args = [(data_loader, seq, verbose, log_dir) for seq in sequences]
    num_workers = get_optimal_worker_count()

    if args.correct:
        if verbose: print(f"Storing annotation corrections for each parallel sequence to {abspath(join(pardir, 'out', 'corrections', args.dataset))}")
        try: _run_in_parallel(_correct_worker, worker_args, num_workers, "Optimizing sequences in parallel...")
        except Exception as e: print(f"Error occurred: {e}"); raise

    if args.evaluate:
        if verbose: print(f"Storing evaluations for each parallel sequence to {abspath(join(pardir, 'out', 'evaluations', args.dataset))}")
        try: _run_in_parallel(_evaluate_worker, worker_args, num_workers, "Evaluating sequences in parallel...")
        except Exception as e: print(f"Error occurred: {e}"); raise
        from evaluate import display_metrics, plot_distributions_and_bar_plots
        display_metrics(sequences, False, args.dataset)
        plot_distributions_and_bar_plots(sequences, args.dataset)


if __name__ == '__main__':
    main()
