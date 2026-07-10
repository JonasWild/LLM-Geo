"""Shared GIS instructions used by specialized agents."""

GIS_RULES = """
- Never invent data columns, coordinate reference systems, paths, or results.
- Use only the system-retrieved local GeoJSON source paths supplied in the workflow.
- Never issue a new HTTP request or query an external data API from generated code.
- Treat retrieved GeoJSON files as read-only inputs. Never write maps, charts,
	reports, manifests, or derived data beside an input source.
- The program runs with the run's results directory as its working directory. Write
	every generated artifact using a relative path in that directory.
- Inspect data before planning and preserve observed field names.
- Reproject spatial layers to compatible projected CRSs before distance/area work.
- Preserve identifier semantics, especially leading zeros in FIPS/GEOID fields.
- Handle nulls, invalid geometries, join cardinality, and duplicate spatial joins.
- Every operation must have explicit inputs, outputs, and validation criteria.
- Maps and charts must include meaningful titles, units, legends/colorbars, and be saved.
- Use current Pandas, GeoPandas, Shapely, Rasterio, Matplotlib, SciPy, and Statsmodels APIs.
""".strip()
