"""Small trusted operations used to verify registry integration."""

from __future__ import annotations

import geopandas as gpd

from llm_geo.operations import code


@code
def identity_features(features: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return the supplied geospatial features unchanged.

    Args:
        features: Input GeoDataFrame to pass through unchanged.

    Returns:
        The same GeoDataFrame instance without any modification.
    """
    return features