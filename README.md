# Detector Tracker
Use [Object Detection](https://alwaysai.co/docs/application_development/core_computer_vision_services.html#object-detection) and [Object Tracking](https://alwaysai.co/docs/application_development/core_computer_vision_services.html#object-tracking) to follow unique objects as they move across the frame.

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
| Field        | Subfield          | Allowed Values                                                                                 |
|--------------|-------------------|------------------------------------------------------------------------------------------------|
| video_stream | app_frame_size    | List of two ints: image width and height                                                       |
| video_stream | mode              | String: The video capture mode: `IP` or `USB`                                                          |
| video_stream | arg               | String or int: The arg for the given `mode`: `IP`->string of URL, `USB`->int of camera index                  |
| tracker      | max_distance      | Int: The maximum number of distance in pixels for a new detection to be matched to a tracked object |
| tracker      | deregister_frames | Int: The number of frames before dropping a lost tracked object                                     |
| tracker      | min_inertia       | Int: The threshold of matched frames before an object is initialized as a tracked object            |

## Support
* [Documentation](https://alwaysai.co/docs/)
* [Community Discord](https://discord.gg/alwaysai)
* Email: support@alwaysai.co

