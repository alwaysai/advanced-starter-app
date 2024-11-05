import time
import edgeiq


def object_enters(object_id, prediction):
    print("{}: {} enters".format(object_id, prediction.label))


def object_exits(object_id, prediction):
    print("{} exits".format(prediction.label))


def main():
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

    print("Engine: {}".format(obj_detect.engine))
    print("Accelerator: {}\n".format(obj_detect.accelerator))
    print("Model:\n{}\n".format(obj_detect.model_id))
    print("Labels:\n{}\n".format(obj_detect.labels))

    tracker = edgeiq.CentroidTracker(
            deregister_frames=30,
            enter_cb=object_enters,
            exit_cb=object_exits)
    fps = edgeiq.FPS()

    try:
        with edgeiq.WebcamVideoStream(cam=0) as video_stream, \
                edgeiq.Streamer() as streamer:
            # Allow Webcam to warm up
            time.sleep(2.0)
            fps.start()

            while True:
                frame = video_stream.read()
                results = obj_detect.detect_objects(frame, confidence_level=.5)
                predictions = results.predictions

                # Generate text to display on streamer
                text = ["Model: {}".format(obj_detect.model_id)]
                text.append(
                        "Inference time: {:1.3f} s".format(
                            results.duration))
                text.append("Objects:")

                objects = tracker.update(predictions)

                # Update the label to reflect the object ID
                tracked_predictions = []
                for (object_id, prediction) in objects.items():
                    # Use the original class label instead of the prediction
                    # label to avoid iteratively adding the ID to the label
                    class_label = obj_detect.labels[prediction.index]
                    prediction.label = "{}: {}".format(object_id, class_label)
                    text.append("{}".format(prediction.label))
                    tracked_predictions.append(prediction)

                frame = edgeiq.markup_image(
                        frame, tracked_predictions, show_labels=True,
                        show_confidences=False, colors=obj_detect.colors)
                streamer.send_data(frame, text)
                fps.update()

                if streamer.check_exit():
                    break

    finally:
        fps.stop()
        print("elapsed time: {:.2f}".format(fps.get_elapsed_seconds()))
        print("approx. FPS: {:.2f}".format(fps.compute_fps()))

        print("Program Ending")


if __name__ == "__main__":
    main()
