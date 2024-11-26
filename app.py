import time
import edgeiq
from datetime import datetime

from config import load_config, VideoMode


def object_exits(object_id, prediction):
    enter_timestamp = prediction._enter_timestamp
    exit_timestamp = edgeiq.generate_event_timestamp()
    enter_dt = datetime.strptime(enter_timestamp, "%Y-%m-%dT%H:%M:%SZ")
    exit_dt = datetime.strptime(exit_timestamp, "%Y-%m-%dT%H:%M:%SZ")
    dwell_time = (exit_dt - enter_dt).total_seconds()
    dwell_event = edgeiq.ValueEvent(
        event_label='person_dwell_seconds',
        object_label=object_id,
        value=dwell_time
    )
    dwell_event.publish_event()
    print("{}: {} exits".format(object_id, prediction.label))


def object_enters(object_id, prediction):
    prediction._enter_timestamp = edgeiq.generate_event_timestamp()
    print("{} enters".format(prediction.label))


def main():
    cfg = load_config()

    print(cfg.video_stream)
    video_stream_cls = edgeiq.IPVideoStream if cfg.video_stream.mode == VideoMode.IP \
        else edgeiq.WebcamVideoStream

    # Select the last model in the app configuration models list
    # Currently supports Tensor RT and DNN
    model_id_list = edgeiq.AppConfig().model_id_list
    if len(model_id_list) == 0:
        raise RuntimeError('No models in model ID list!')
    model_id = model_id_list[-1]
    obj_detect = edgeiq.ObjectDetection(model_id)
    if edgeiq.is_jetson() and obj_detect.model_config.tensor_rt_support:
        engine = edgeiq.Engine.TENSOR_RT
    elif obj_detect.model_config.dnn_support:
        engine = edgeiq.Engine.DNN
    else:
        raise ValueError(f'Model {obj_detect.model_id} not supported on this device!')
    obj_detect.load(engine)

    print(f'Engine: {obj_detect.engine}')
    print(f'Accelerator: {obj_detect.accelerator}\n')
    print(f'Model:\n{obj_detect.model_id}\n')
    print(f'Labels:\n{obj_detect.labels}\n')

    tracker = edgeiq.KalmanTracker(
        max_distance=cfg.tracker.max_distance,
        deregister_frames=cfg.tracker.deregister_frames,
        min_inertia=cfg.tracker.min_inertia,
        enter_cb=object_enters,
        exit_cb=object_exits
    )

    # TODO: Load zones
    zone_list = edgeiq.ZoneList.from_config_file('zone-config.json')
    zone_list = edgeiq.ZoneList(
        zones=zone_list.zones,
        image_width=cfg.video_stream.app_frame_size[0],
        image_height=cfg.video_stream.app_frame_size[1],
    )
    zone_list._colors = [(0, 255, 0), (0, 0, 255)]


    # TODO: Convert zone to image size and update colors

    fps = edgeiq.FPS()

    last_publish = None

    try:
        with video_stream_cls(cfg.video_stream.arg) as video_stream, \
                edgeiq.Streamer() as streamer:
            # Allow Webcam to warm up
            time.sleep(2.0)
            fps.start()

            while True:
                frame = video_stream.read()
                # TODO: This will keep the aspect ratio, but not add additional
                # black bars to make the image match the desired size. It will
                # crash and the user can then update the config to match.
                frame = edgeiq.resize(
                    image=frame,
                    width=cfg.video_stream.app_frame_size[0],
                    height=cfg.video_stream.app_frame_size[1],
                    keep_scale=True
                )

                results = obj_detect.detect_objects(frame, confidence_level=.5)
                predictions = edgeiq.filter_predictions_by_label(
                    predictions=results.predictions,
                    label_list=['person']
                )

                # Generate text to display on streamer
                text = [f'Model: {obj_detect.model_id}']
                text.append(f'Loaded to {obj_detect.engine}:{obj_detect.accelerator}')
                text.append('Inference time: {:1.3f} s'.format(results.duration))
                text.append('Objects:')

                objects = tracker.update(predictions)

                # Update the label to reflect the object ID
                tracked_predictions = []
                for object_id, prediction in objects.items():
                    # Use the original class label instead of the prediction
                    # label to avoid iteratively adding the ID to the label
                    class_label = obj_detect.labels[prediction.index]
                    prediction.label = f'{object_id}: {class_label}'
                    text.append(f'{prediction.label}')
                    tracked_predictions.append(prediction)

                # Periodically send occupancy events
                current_time = datetime.now()
                if last_publish is None or (
                        (current_time - last_publish).total_seconds() >= 2):
                    # Get person occupancy counts
                    total_detected = max(objects.keys()) if len(objects) > 0 \
                        else 0
                    current_detected = len(tracked_predictions)
                    total_detected_event = edgeiq.ValueEvent(
                        timestamp=edgeiq.generate_event_timestamp(),
                        event_label='total_detected',
                        object_label='person',
                        value=total_detected
                    )
                    total_detected_event.publish_event()
                    current_detected_event = edgeiq.ValueEvent(
                        timestamp=edgeiq.generate_event_timestamp(),
                        event_label='current_detected',
                        object_label='person',
                        value=current_detected
                    )
                    current_detected_event.publish_event()

                    # Get zone capacity
                    zone_capacity = {name: 0 for name in zone_list.zone_names}
                    for zone in zone_list.zones:
                        for prediction in objects.values():
                            if zone.check_prediction_within_zone(prediction):
                                zone_capacity[zone.name] += 1

                    for zone_name, capacity in zone_capacity.items():
                        event = edgeiq.ValueEvent(
                            timestamp=edgeiq.generate_event_timestamp(),
                            event_label='zone_capacity',
                            object_label='person',
                            zone_label=zone_name,
                            value=capacity
                        )
                        event.publish_event()

                    last_publish = datetime.now()

                frame = zone_list.markup_image_with_zones(
                    image=frame,
                    show_labels=True,
                    show_boundaries=True,
                    fill_zones=True
                )

                frame = edgeiq.markup_image(
                    frame,
                    tracked_predictions,
                    show_labels=True,
                    show_confidences=False,
                    colors=obj_detect.colors
                )
                streamer.send_data(frame, text)
                fps.update()

                if streamer.check_exit():
                    break

    finally:
        fps.stop()
        print('elapsed time: {:.2f}'.format(fps.get_elapsed_seconds()))
        print('approx. FPS: {:.2f}'.format(fps.compute_fps()))

        print('Program Ending')


if __name__ == '__main__':
    main()
