import argparse
import logging
from typing import List
import edgeiq


def write_tracking_results_to_mot(
    tracking_results: List[edgeiq.TrackingResults[edgeiq.TrackablePrediction[edgeiq.ObjectDetectionPrediction]]],
    output_path: str
):
    with open(output_path, "w") as f:
        for frame_idx, frame_res in enumerate(tracking_results):
            for res in frame_res.values():
                line = (
                    f"{frame_idx},"
                    f"{res.tid},"
                    f"{res.prediction.box.start_x},"
                    f"{res.prediction.box.start_y},"
                    f"{res.prediction.box.width},"
                    f"{res.prediction.box.height},"
                    f"{res.confidence:.6f},"
                    f"{res.prediction.index},"
                    f"1.0,"  # visibility
                    "-1\n"  # placeholder
                )
                f.write(line)


def main(
    results_path: str,
    output_path: str
):
    # Load tracker output
    per_frame_results: List[edgeiq.TrackingResults] = edgeiq.load_analytics_results(
        filepath=results_path,
        packet_types=['TrackingResults']
    )

    write_tracking_results_to_mot(
        tracking_results=per_frame_results,
        output_path=output_path
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Tracking Results to MOT')
    parser.add_argument(
        '--results-path',
        type=str,
        required=True,
        help='The path to the results analytics file'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        default='tracking-results.txt',
        help='The path to the output directory'
    )
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()
    main(
        results_path=args.results_path,
        output_path=args.output_path
    )
