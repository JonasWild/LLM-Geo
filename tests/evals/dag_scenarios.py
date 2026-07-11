"""Medium-to-high complexity agentic GIS evaluation prompts."""

from __future__ import annotations


DAG_TASKS = (
    ("hospitals_near_units", "Retrieve friendly units from SitaWare, find nearby hospitals with Overpass, determine the nearest hospital for each unit, and create a tile-backed map and concise summary."),
    ("bridge_chokepoints", "Retrieve a SitaWare route, query bridges and waterways with Overpass, identify route crossings, rank potential chokepoints, and visualize them."),
    ("tactical_symbol_map", "Retrieve SitaWare entities, resolve suitable symbols through SIDC Search, and render a tile-backed tactical overview."),
    ("critical_infrastructure", "Geocode an operational area with Nominatim, retrieve power, water, medical, and communications infrastructure with Overpass, classify it, and create a map and summary."),
    ("units_outside_boundary", "Retrieve units from SitaWare, resolve a named administrative boundary through Nominatim, identify units outside it, and produce GeoJSON and a map."),
    ("medical_support_ranking", "Retrieve units from SitaWare and medical facilities from Overpass, calculate distances, rank facilities by accessibility, and produce a per-unit recommendation."),
    ("route_fuel_accessibility", "Retrieve a route from SitaWare, query nearby fuel stations with Overpass, calculate route detours, and map coverage gaps."),
    ("incident_clustering", "Retrieve recent incidents from SitaWare, cluster them spatially, calculate cluster summaries, and render a cluster or heat map."),
    ("urban_unit_distribution", "Retrieve units from SitaWare, retrieve urban land-use polygons with Overpass, classify units as urban or non-urban, and summarize affiliation distributions."),
    ("multi_location_comparison", "Geocode three named locations, retrieve equivalent infrastructure categories for each with Overpass, compare counts and density, and generate comparative maps."),
    ("road_network_summary", "Resolve an area of interest with Nominatim, retrieve its road network with Overpass, classify road types, calculate lengths, and produce a map and report."),
    ("water_crossing_risk", "Retrieve a planned SitaWare route, obtain rivers and bridges with Overpass, find unsupported water crossings, and produce a risk map."),
    ("friendly_force_dispersion", "Retrieve friendly units, resolve their SIDCs, calculate nearest-neighbor distances and clusters, and visualize potential concentration areas."),
    ("restricted_zone_violations", "Retrieve units and restricted areas from SitaWare, calculate spatial violations, classify them by unit type using SIDC Search, and create an exception report."),
    ("emergency_service_coverage", "Resolve a city boundary, retrieve fire stations, police stations, and hospitals with Overpass, calculate approximate coverage areas, and map underserved regions."),
    ("entity_reconciliation", "Retrieve SitaWare facilities and comparable OpenStreetMap facilities, spatially match likely duplicates, identify conflicts, and produce a reconciliation table."),
    ("operational_picture", "Retrieve SitaWare units, incidents, and routes, resolve unit symbols through SIDC Search, and render all layers over the tile server with a legend."),
    ("supply_route_vulnerability", "Retrieve a supply route from SitaWare, query bridges, tunnels, major intersections, and fuel stations with Overpass, rank vulnerable segments, and create a map."),
    ("named_place_context", "Geocode a named location, retrieve surrounding terrain-relevant OpenStreetMap features, retrieve nearby SitaWare entities, resolve their symbols, and create a contextual map."),
    ("facility_proximity_matrix", "Retrieve multiple SitaWare units and several Overpass facility categories, calculate a unit-to-facility distance matrix, and summarize the best options for each unit."),
    ("responsibility_overlap", "Retrieve unit areas of responsibility from SitaWare, calculate overlaps and gaps, associate each area with its unit SIDC, and produce a diagnostic map."),
    ("historical_movement", "Retrieve SitaWare unit positions for two time windows, calculate displacement and direction, identify unusual movement, and render movement vectors."),
    ("communication_site_suitability", "Resolve an area of interest, retrieve high points or relevant terrain proxies and access roads from public sources, combine them with SitaWare constraints, rank candidate communication sites, and map them."),
    ("evacuation_support", "Retrieve an incident location from SitaWare, obtain hospitals and road infrastructure with Overpass, calculate candidate evacuation destinations, and produce ranked recommendations."),
    ("border_activity_overview", "Resolve a named border region, retrieve border-related OpenStreetMap features, combine them with SitaWare incidents and units, resolve SIDCs, and create a layered overview."),
    ("infrastructure_dependencies", "Retrieve selected SitaWare facilities and surrounding power, road, water, and communications features with Overpass, derive dependencies, and visualize them."),
    ("unit_accessibility", "Retrieve several units and their area of interest, obtain the road network and key facilities, calculate comparative accessibility indicators, and create a ranked report."),
    ("operational_change_detection", "Compare two SitaWare snapshots, detect created, removed, moved, or changed entities, resolve symbols for changed units, and render the differences."),
    ("multi_aoi_picture", "Resolve several named operational areas, retrieve relevant OpenStreetMap features for each, combine them with SitaWare entities, resolve SIDCs, and generate one consistent multi-layer map."),
    ("mission_support_assessment", "Retrieve a SitaWare mission area, friendly units, routes, and incidents; enrich it with Overpass medical, fuel, bridge, and road data; resolve tactical symbols; calculate risks and support gaps; and produce a tile-backed map plus an analytical summary."),
)

