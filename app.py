import json
import time
from typing import List, Optional
import edgeiq
import numpy as np

from config import InferenceMode, load_config, VideoMode


def object_enters(object_id, prediction):
    print("{}: {} enters".format(object_id, prediction.label))


def object_exits(object_id, prediction):
    print("{} exits".format(prediction.label))


def get_video_stream(mode, arg: str | int) -> edgeiq.VideoStream:
    if mode == VideoMode.FILE:
        return edgeiq.FileVideoStream(arg)
    elif mode == VideoMode.USB:
        return edgeiq.WebcamVideoStream(arg)
    elif mode == VideoMode.IP:
        return edgeiq.IPVideoStream(arg)
    else:
        raise ValueError(f'Unsupported mode {mode}!')


def get_inference(
    mode: InferenceMode,
    model_id: str,
    annotations_file_paths: Optional[List[str]]
) -> edgeiq.ObjectDetection:
    if mode == InferenceMode.INFERENCE:
        obj_detect = edgeiq.ObjectDetection(model_id=model_id)
        if (edgeiq.is_jetson() or edgeiq.find_nvidia_gpu()) \
                and obj_detect.model_config.tensor_rt_support:
            engine = edgeiq.Engine.TENSOR_RT
        elif obj_detect.model_config.dnn_support:
            engine = edgeiq.Engine.DNN
        else:
            engine = edgeiq.Engine.ONNX_RT
        obj_detect.load(engine)
        return obj_detect

    elif mode == InferenceMode.AAI_ANNOTATIONS:
        annotation_results = [
            edgeiq.load_analytics_results(file_path) for file_path
            in annotations_file_paths
        ]
        return edgeiq.ObjectDetectionAnalytics(
            annotations=annotation_results,
            model_id=model_id
        )
    elif mode == InferenceMode.COCO_ANNOTATIONS:
        annotation_results = [
            # End frame was determined manually based on video properties
            edgeiq.parse_coco_annotations(file_path) for file_path
            in annotations_file_paths
        ]
        with open('loaded-annotations.json', 'w') as f:
            json.dump([edgeiq.to_json_serializable(o) for o in annotation_results], f, indent=2)

        return edgeiq.ObjectDetectionAnalytics(
            annotations=annotation_results, model_id=model_id)
    else:
        raise ValueError(f'Unsupported mode {mode}!')


class NoVideoWriter(edgeiq.VideoWriter):
    def __init__(self):
        pass

    def write_frame(self, frame: np.ndarray):
        pass

    def close(self):
        pass


def get_video_writer(enable: bool, *args, **kwargs) -> edgeiq.VideoWriter:
    if enable:
        return edgeiq.VideoWriter(*args, **kwargs)
    else:
        return NoVideoWriter()


def main():
    cfg = load_config()
    print(f'Configuration:\n{cfg}')

    # Select the last model in the app configuration models list
    # Currently supports Tensor RT and DNN
    model_id_list = edgeiq.AppConfig().model_id_list
    if len(model_id_list) == 0:
        raise RuntimeError('No models in model ID list!')

    model_id = model_id_list[-1]
    obj_detect = get_inference(
        mode=cfg.inference.mode,
        model_id=model_id,
        annotations_file_paths=cfg.inference.annotations_file_paths
    )

    print(f'Engine: {obj_detect.engine}')
    print(f'Accelerator: {obj_detect.accelerator}\n')
    print(f'Model:\n{obj_detect.model_id}\n')
    print(f'Labels:\n{obj_detect.labels}\n')

    video_streams = [
        get_video_stream(
            mode=video_cfg.mode,
            arg=video_cfg.arg
        ) for video_cfg in cfg.video_streams.video_streams
    ]

    trackers = []
    # NOTE: you may want to disable the tracker in the future
    # currently there is one tracker per stream, but this will
    # track all labels passed to it, so it may not work if the model
    # is different than the annotations
    for i in range(0, len(video_streams)):
        trackers.append(edgeiq.KalmanTracker(
            max_distance=cfg.tracker.max_distance,
            deregister_frames=cfg.tracker.deregister_frames,
            min_inertia=cfg.tracker.min_inertia,
            enter_cb=object_enters,
            exit_cb=object_exits
        ))

    video_writer = get_video_writer(
        enable=cfg.video_writer.enable,
        output_path=cfg.video_writer.output_path,
        fps=cfg.video_writer.fps,
        codec=cfg.video_writer.codec,
        chunk_duration_s=cfg.video_writer.chunk_duration_s
    )

    fps = edgeiq.FPS()

    try:
        with edgeiq.Streamer() as streamer:
            for video_stream in video_streams:
                video_stream.start()
            # Allow camera stream to warm up
            time.sleep(2.0)
            fps.start()

            while True:
                frames = []
                for i, video_stream in enumerate(video_streams):
                    frame = video_stream.read()

                    results = obj_detect.detect_objects(
                        frame,
                        confidence_level=cfg.inference.confidence,
                        overlap_threshold=cfg.inference.overlap_threshold
                    )
                    predictions = edgeiq.filter_predictions_by_label(
                        predictions=results.predictions,
                        label_list=cfg.inference.labels
                    ) if cfg.inference.labels else results.predictions

                    # Generate text to display on streamer
                    text = [f'Model: {obj_detect.model_id}']
                    text.append(f'Loaded to {obj_detect.engine}:{obj_detect.accelerator}')
                    text.append('Inference time: {:1.3f} s'.format(results.duration))
                    text.append('Objects:')

                    objects = trackers[i].update(predictions)

                    # Update the label to reflect the object ID
                    tracked_predictions = []
                    for object_id, prediction in objects.items():
                        # Use the original class label instead of the prediction
                        # label to avoid iteratively adding the ID to the label
                        class_label = obj_detect.labels[prediction.index]
                        prediction.label = f'{object_id}: {class_label}'
                        text.append(f'{prediction.label}')
                        tracked_predictions.append(prediction)

                    frame = edgeiq.markup_image(
                        frame,
                        tracked_predictions,
                        show_labels=True,
                        show_confidences=False,
                        colors=obj_detect.colors
                    )
                    frames.append(frame)
                # then after we iterate, combine the frames and mark up the frame count
                frame = edgeiq.safe_hstack(frames)
                frame, _, _ = edgeiq.draw_text_with_background(
                    image=frame,
                    text=f'Frame idx: {fps._num_frames}',
                    start_x=5,
                    start_y=5,
                    font_size=1,
                    font_thickness=2,
                    color=(0, 0, 0)
                )
                streamer.send_data(frame, text)
                video_writer.write_frame(frame)
                fps.update()

                if streamer.check_exit():
                    break

    finally:
        fps.stop()
        for video_stream in video_streams:
            video_stream.stop()
        video_writer.close()
        print('elapsed time: {:.2f}'.format(fps.get_elapsed_seconds()))
        print('approx. FPS: {:.2f}'.format(fps.compute_fps()))

        print('Program Ending')


if __name__ == '__main__':
    main()
