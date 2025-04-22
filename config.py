from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import edgeiq


def convert_to_enum(val: str, enum: type[Enum]):
    if val is None:
        raise ValueError(f'{val} not found in config!')
    try:
        enum_val = enum[val.upper()]
    except KeyError:
        raise ValueError(
            f'Invalid input! Got {val} expected one of {[member.value for member in enum]}'
        )
    return enum_val


class VideoMode(Enum):
    FILE = 'FILE'
    IP = 'IP'
    USB = 'USB'


@dataclass
class VideoStreamConfig:
    mode: VideoMode
    arg: str | int

    @classmethod
    def from_dict(cls, cfg: dict):
        cfg['mode'] = convert_to_enum(cfg['mode'], VideoMode)
        return cls(**cfg)


class InferenceMode(Enum):
    INFERENCE = 'INFERENCE'
    ANNOTATIONS = 'ANNOTATIONS'


@dataclass
class InferenceConfig:
    mode: InferenceMode
    confidence: float
    overlap_threshold: float
    labels: List[str]
    annotations_file_paths: Optional[List[str]] = None

    @classmethod
    def from_dict(cls, cfg: dict):
        cfg['mode'] = convert_to_enum(cfg['mode'], InferenceMode)
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
class VideoWriterConfig:
    enable: bool
    output_path: str
    fps: int
    codec: str
    chunk_duration_s: Optional[int] = None

    @classmethod
    def from_dict(cls, cfg: dict):
        return cls(**cfg)


@dataclass
class Config:
    video_stream: VideoStreamConfig
    inference: InferenceConfig
    tracker: TrackerConfig
    video_writer: VideoWriterConfig

    @classmethod
    def from_dict(cls, cfg: dict):
        cfg['video_stream'] = VideoStreamConfig.from_dict(cfg['video_stream'])
        cfg['inference'] = InferenceConfig.from_dict(cfg['inference'])
        cfg['tracker'] = TrackerConfig.from_dict(cfg['tracker'])
        cfg['video_writer'] = VideoWriterConfig.from_dict(cfg['video_writer'])
        return cls(**cfg)


def load_config() -> Config:
    app_file = edgeiq.AppConfig()._app_file.contents
    if app_file is None:
        raise RuntimeError('alwaysai.app.json not found!')
    cfg = app_file['app_configurations']
    out = Config.from_dict(cfg)
    return out
