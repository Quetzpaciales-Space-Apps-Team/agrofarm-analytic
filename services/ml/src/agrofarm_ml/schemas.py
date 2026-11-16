"""Request models. They mirror docs/contract/openapi.yaml and validate at the boundary."""

from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field

EnsoPhase = Literal["el_nino", "la_nina", "neutral"]
CropId = Annotated[str, Field(min_length=1, max_length=60)]
Weight = Annotated[float, Field(ge=0, le=1)]


class Scenario(BaseModel):
    rainfall_change_pct: float = Field(ge=-80, le=100)
    temperature_change_c: float = Field(ge=-5, le=8)
    enso_phase: EnsoPhase


class _FarmRequest(BaseModel):
    farm_id: UUID
    centroid_lat: float = Field(ge=-90, le=90)
    centroid_lon: float = Field(ge=-180, le=180)
    scenario: Scenario


class VegetationRequest(_FarmRequest):
    pass


class SimulateRequest(_FarmRequest):
    crop: CropId


class Priorities(BaseModel):
    weights: dict[str, Weight]
    constraints: dict[str, Any]


class RotationScoreRequest(_FarmRequest):
    current_rotation: list[CropId] = Field(max_length=12)
    priorities: Priorities
