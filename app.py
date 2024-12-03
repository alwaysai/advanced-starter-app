from datetime import datetime
import time
import edgeiq

from config import load_config, VideoMode

ZONES = None


def object_exits(object_id, prediction):
    end_event = edgeiq.EndTimedEvent(
        event_label='exit_event',
        event_id=prediction._event_id,
        object_label='person',
        object_id=object_id
    )
    end_event.publish_event()
    print("{}: {} exits".format(object_id, prediction.label))


def object_enters(object_id, prediction):
    zone = ZONES.get_zone_for_prediction(prediction)
    zone_name = zone.name if zone is not None else 'None'
    occurrence_event = edgeiq.OccurrenceEvent(
        event_label='zone_entry',
        zone_label=zone_name,
        object_id=object_id,
        object_label='person'
    )
    occurrence_event.publish_event()
    start_event = edgeiq.StartTimedEvent(
        event_label='entry_event',
        object_label='person',
        object_id=object_id
    )
    start_event.publish_event()
    prediction._event_id = start_event.event_id
    prediction._last_zone = zone_name
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
    global ZONES
    ZONES = zone_list

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
                text = [f'Model: {obj_detect.model_id}', f'Labels:\n{obj_detect.labels}\n']
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

                    # Get zone entry events
                    for zone in zone_list.zones:
                        for key, prediction in objects.items():
                            if zone.check_prediction_within_zone(prediction):
                                if zone.name != prediction._last_zone:
                                    prediction._last_zone = zone.name
                                    zone_change_event = edgeiq.OccurrenceEvent(
                                        object_id=key,
                                        event_label='zone_entry',
                                        zone_label=zone.name,
                                        object_label='person'
                                    )
                                    zone_change_event.publish_event()

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
