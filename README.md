# Advanced Starter App
This starter app enables a configurable workflow with the following datapath:
1. Video stream input from file, IP stream, or USB camera.
2. [Object Detection](https://alwaysai.co/docs/application_development/core_computer_vision_services.html#object-detection) inference using a model, or loading from annotations.
3. [Object Tracking](https://alwaysai.co/docs/application_development/core_computer_vision_services.html#object-tracking).
4. Business logic: Add your own custom business logic based on the results
5. Save to an output video file if desired.

The app enables the following workflows:
1. Demo video markup: Load a video, draw annotations, and save the marked up video.
2. Off-target testing: Load a video, draw annotations or perform inference with a model, save output data for analysis.
3. On-target production operation: Connect to an IP stream, perform inference using a model, publish the results to the alwaysAI cloud.

## Requirements
* [alwaysAI account](https://alwaysai.co/auth?register=true)
* [alwaysAI Development Tools](https://alwaysai.co/docs/get_started/development_computer_setup.html)

## Usage
Once the alwaysAI tools are installed on your development machine (or edge device if developing directly on it) you can install and run the app with the following CLI commands:

To perform initial configuration of the app:
```
aai app configure
```

To prepare the runtime environment and install app dependencies:
```
aai app install
```

To start the app:
```
aai app start
```

### Configuration
| Field            | Subfield               | Allowed Values                                                                                                                       |
|------------------|------------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| video_streams    |                        | List[object]: The per-stream configurations                                                                                          |
| video_streams[n] | mode                   | String: The video capture mode: `FILE`, `IP`, or `USB`                                                                               |
| video_streams[n] | arg                    | String or int: The arg for the given `mode`: `IP`->string of URL, `USB`->int of camera index                                         |
| inference        | mode                   | String: The inference mode: `INFERENCE`, `COCO_ANNOTATIONS`, `AAI_ANNOTATIONS`                                                       |
| inference        | confidence             | Float: The confidence threshold to accept a prediction from the inference engine                                                     |
| inference        | overlap_threshold      | Float: The IoU threshold to use for NMS (if supported by the model)                                                                  |
| inference        | labels                 | Optional[List[string]]: The list of labels to accept. When the field is not present, all labels will be passed through to next steps |
| inference        | annotations_file_paths | List[string]: The per-stream paths to annotations files in the format matching the inference mode                                    |
| tracker          | max_distance           | Int: The maximum number of distance in pixels for a new detection to be matched to a tracked object                                  |
| tracker          | deregister_frames      | Int: The number of frames before dropping a lost tracked object                                                                      |
| tracker          | min_inertia            | Int: The threshold of matched frames before an object is initialized as a tracked object                                             |
| video_writer     | enable                 | Bool: Enable/disable output video capture                                                                                            |
| video_writer     | output_path            | String: The path to save the output file(s)                                                                                          |
| video_writer     | fps                    | Int: The FPS to save the output video at                                                                                             |
| video_writer     | codec                  | String: The encoding to use for the video. Popular options include "avc1" (H264), "mp4v" (MPEG-4), "MJPG" (AVI)                      |
| video_writer     | chunk_duration_s       | Int: The Duration in secs of video chunks, or disabled if `null`.                                                                    |

## Support
* [Documentation](https://alwaysai.co/docs/)
* [Community Discord](https://discord.gg/alwaysai)
* Email: support@alwaysai.co

