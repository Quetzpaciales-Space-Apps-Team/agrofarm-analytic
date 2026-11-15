"""Terrain layers from the Copernicus GLO-30 DEM: slope, aspect, D8 flow accumulation and
provisional erosion and waterlogging zones."""

from agrofarm_data.terrain.pipeline import DATASET, write_clean

__all__ = ["DATASET", "write_clean"]
