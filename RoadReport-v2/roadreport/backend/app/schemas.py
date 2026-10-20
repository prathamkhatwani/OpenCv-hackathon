"""Strict, versioned sidecar contracts. Times are seconds in the video clock."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
import numpy as np

class Strict(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)

class Sample(Strict):
    t: float = Field(ge=0, le=65)
    az: float | None = Field(default=None, ge=-100, le=100)
    gx: float | None = Field(default=None, ge=-50, le=50)
    gy: float | None = Field(default=None, ge=-50, le=50)
    gz: float | None = Field(default=None, ge=-50, le=50)
    lat: float | None = Field(default=None, ge=-90, le=90)
    lon: float | None = Field(default=None, ge=-180, le=180)
    speed_mps: float | None = Field(default=None, ge=0, le=80)

    @model_validator(mode='after')
    def paired_gps(self):
        if (self.lat is None) != (self.lon is None):
            raise ValueError('lat and lon must be supplied together')
        return self

class Telemetry(Strict):
    schema_version: Literal[1] = 1
    acceleration_frame: Literal['vehicle_vertical_gravity_removed'] = 'vehicle_vertical_gravity_removed'
    units: Literal['SI'] = 'SI'
    time_offset_s: float = Field(default=0, ge=-5, le=5)
    synchronization_uncertainty_s: float = Field(default=0.1, ge=0, le=2)
    samples: list[Sample] = Field(min_length=2, max_length=12000)

    @model_validator(mode='after')
    def ordered(self):
        if any(b.t <= a.t for a,b in zip(self.samples,self.samples[1:])):
            raise ValueError('Sample times must be strictly increasing')
        return self

class Calibration(Strict):
    schema_version: Literal[1] = 1
    # Normalized undistorted image points -> metres: x right, y ahead of front axle.
    homography: list[list[float]]
    track_width_m: float = Field(default=1.5, ge=0.8, le=3)
    tyre_width_m: float = Field(default=0.22, ge=0.1, le=0.6)
    wheelbase_m: float = Field(default=2.6, ge=1, le=6)
    lateral_uncertainty_m: float = Field(default=0.15, ge=0.01, le=1)
    timing_uncertainty_s: float = Field(default=0.25, ge=0.05, le=2)
    straight_path_valid: bool = False
    camera_id: str = Field(min_length=1, max_length=80)
    calibration_note: str = Field(min_length=5, max_length=500)

    @model_validator(mode='after')
    def valid_matrix(self):
        h=np.asarray(self.homography,dtype=float)
        if h.shape != (3,3) or not np.isfinite(h).all() or abs(np.linalg.det(h)) < 1e-9:
            raise ValueError('homography must be a finite invertible 3x3 matrix')
        if np.linalg.cond(h)>1e8:
            raise ValueError('Unstable homography')
        return self

class Options(Strict):
    telemetry: Telemetry | None = None
    calibration: Calibration | None = None
    source_kind: Literal['real','synthetic'] = 'real'
    source_note: str = Field(default='', max_length=500)

class Review(Strict):
    decision: Literal['approve','reject']
    reviewer: str = Field(default='team reviewer', min_length=1, max_length=80)
