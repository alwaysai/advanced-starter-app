import argparse
import logging
from typing import List
import edgeiq


def main(
    ground_truth_path: str,
    results_path: str,
    metrics: List[str],
    output_path: str
):
    evaluator = edgeiq.MOTEvaluator(
        gt_files=[ground_truth_path],
        tracker_files=[results_path],
        metrics_list=metrics,
        output_pth=output_path
    )
    evaluator.evaluate_all()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark Tracker Performance')
    parser.add_argument(
        '--ground-truth-path',
        type=str,
        required=True,
        help='The path to the ground truth MOT file'
    )
    parser.add_argument(
        '--metrics',
        nargs='+',
        required=True,
        help='The list of metrics to calculate. One or more of [HOTA, CLEAR, Identity, VACE]'
    )
    parser.add_argument(
        '--results-path',
        type=str,
        required=True,
        help='The path to the results MOT file'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        default='tracker-benchmark.json',
        help='The path to the output directory'
    )
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()
    main(
        ground_truth_path=args.ground_truth_path,
        metrics=args.metrics,
        results_path=args.results_path,
        output_path=args.output_path
    )
