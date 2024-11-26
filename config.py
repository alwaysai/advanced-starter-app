from dataclasses import dataclass
from enum import Enum
from typing import List

import edgeiq


def convert_to_enum(val, enum, name):
    if val is None:
        raise ValueError('{} not found in config!'.format(name))
    try:
        enum_val = enum[val.upper()]
    except KeyError:
        raise ValueError('Invalid {}! Got {} expected {}'.format(
            name, val, enum))
    return enum_val


class VideoMode(str, Enum):
    IP = 'IP'
    USB = 'USB'


@dataclass
class VideoStreamConfig:
    app_frame_size: List[int]
    mode: VideoMode
    arg: str | int

    @classmethod
    def from_dict(cls, cfg: dict):
        cfg['mode'] = convert_to_enum(cfg['mode'], VideoMode, 'Video Mode')
        return cls(**cfg)


@dataclass
class InferenceConfig:
    confidence: float
    overlap_threshold: float
    labels: List[str]

    @classmethod
    def from_dict(cls, cfg: dict):
        return cls(**cfg)


@dataclass
class TrackerConfig:
    max_distance: int
    deregister_frames: int
    min_inertia: int

    @classmethod
    def from_dict(cls, cfg: dict):
        return cls(**cfg)


@dataclass
class Config:
    video_stream: VideoStreamConfig
    inference: InferenceConfig
    tracker: TrackerConfig

    @classmethod
    def from_dict(cls, cfg: dict):
        cfg['video_stream'] = VideoStreamConfig.from_dict(cfg['video_stream'])
        cfg['inference'] = InferenceConfig.from_dict(cfg['inference'])
        cfg['tracker'] = TrackerConfig.from_dict(cfg['tracker'])
        return cls(**cfg)


def load_config() -> Config:
    cfg: dict = edgeiq.AppConfig()._app_file._contents["app_configurations"]
    out = Config.from_dict(cfg)
    return out
