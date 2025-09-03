import argparse
import logging
from typing import List
import edgeiq


def main(
    ground_truth_path: str,
    labels: List[str],
    results_path: str,
    output_dir: str
):
    # Load ground truth
    ground_truth = edgeiq.parse_mot_annotations(
        path=ground_truth_path,
        labels=labels
    )
    perf_analyzer = edgeiq.TrackerPerformanceAnalyzer(
        ground_truth=ground_truth,
        max_distance=100
    )

    # Load tracker output
    per_frame_results: List[edgeiq.TrackingResults] = edgeiq.load_analytics_results(
        filepath=results_path,
        packet_types=['TrackingResults']
    )

    # Limit frames analyzed by the shorter of the two lists (they should ideally
    # be the same if collected on the same video)
    num_frames = min(len(ground_truth), len(per_frame_results))

    for frame_idx in range(num_frames):
        tracking_results = per_frame_results[frame_idx]
        perf_analyzer.update(frame_idx, tracking_results)

    id_changes, id_swaps = perf_analyzer.generate_report()
    id_changes.write_to_file(output_dir)
    id_swaps.write_to_file(output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze Tracker IDs')
    parser.add_argument(
        '--ground-truth-path',
        type=str,
        required=True,
        help='The path to the ground truth MOT file'
    )
    parser.add_argument(
        '--labels',
        nargs='+',
        required=True,
        help='The list of labels in the same order as the model class IDs'
    )
    parser.add_argument(
        '--results-path',
        type=str,
        required=True,
        help='The path to the results analytics file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='.',
        help='The path to the output directory'
    )
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()
    main(
        ground_truth_path=args.ground_truth_path,
        labels=args.labels,
        results_path=args.results_path,
        output_dir=args.output_dir
    )
