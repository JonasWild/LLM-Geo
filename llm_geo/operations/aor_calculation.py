import os
from pathlib import Path

from typing import List, Dict, Any, Optional, Tuple
from xml.etree import ElementTree as ET
import math

from llm_geo import code


@code
async def calculate_aor(name_of_xml_file: str, unit_id: str) -> Dict[str, Any]:
    """
    Calculates Area of Responsibility (AOR) for a specific unit from OPLAN XML. Use this code block when you are looking for the boundaries of an aor.

    Args:
        name_of_xml_file: Name of the layer XML file (e.g., 'my layer.xml')
        unit_id: ID of the unit object in XML (e.g., '9#0316ed81-f646-47ea-91d3-f54c47cec9b8')
    
    Returns:
        .geosjon with one Point feature (the unit of the aor) and multiple LineString features that represent the boundary lines with the following structure:
        {
            'type': 'FeatureCollection',
            'features': [
                {
                    'type': 'Feature',
                    'geometry':
                    {
                        'type': 'Point',
                        'coordinates': [round(unit_position[0], 6), round(unit_position[1], 6)]
                    },
                    'properties': {
                        'sidc': obj_sidc,
                        'name': unit_name,
                        'sw_id': unit_id
                    }
                },
                {
                    'type': 'Feature',
                    'geometry':
                    {
                        'type': 'LineString',
                        'coordinates': <<List of coordinates of boundary lines>>
                    },
                    'properties': {
                        'sidc': <<sidc of the boundary line>>,
                        'name': <<name of the boundary line>>,
                        'sw_id': <<sitaware id of this element>>
                    }
                },
                {
                    <<next boundary line>>
                }
            ],
            'message':<< If no aea can be calculated, this message will be filled. Otherwise empty>>
        }

    Raises:
        FileNotFoundError: If XML file doesn't exist
        ValueError: If unit not found or not a UNIT type

    Examples:
        result_dict = calculate_aor_geojson(
            name_of_xml_file="my_layer.xml",
            unit_id="9#0316ed81-f646-47ea-91d3-f54c47cec9b8"
        )

        if result_dict.get("message"):
            raise RuntimeError(f"AOR konnte nicht berechnet werden: {result_dict['message']}")


        # 1a. Unit-Point extraction (Index 0 is the unit point)
        unit_feature = result_dict["features"][0]
        gdf_unit = gpd.GeoDataFrame.from_features([unit_feature], crs="EPSG:4326")
        # gdf_unit hat jetzt 1 Zeile, Spalten: geometry (Point), sidc, name, sw_id

        # 1b. Boundary line extraction (Index 1 bis Ende sind LineStrings)
        boundary_features = result_dict["features"][1:]
        gdf_boundaries = gpd.GeoDataFrame.from_features(boundary_features, crs="EPSG:4326")
        # gdf_boundaries hat N Zeilen, Spalten: geometry (LineString), sidc, name, sw_id


    """
    # Injected from environment
    root_dir = os.environ["INJECTED_ROOT_DIR"]
    chat_id = os.environ["INJECTED_CHAT_ID"]
    
    # Construct path to XML file using chat context
    xml_path = os.path.join(root_dir, chat_id, name_of_xml_file)
    
    if not Path(xml_path).exists():
        raise FileNotFoundError(f"XML file not found: {name_of_xml_file}")
    
    # Parse XML
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Find unit by ID
    unit_element = root.find(f".//object[@id='{unit_id}']")
    if unit_element is None:
        raise ValueError(f"Unit with ID '{unit_id}' not found in XML")
    
    # Validate it's a UNIT
    unit_type_elem = unit_element.find(".//c2-attributes//type")
    if unit_type_elem is None or unit_type_elem.text != "UNIT":
        raise ValueError(f"Object '{unit_id}' is not a UNIT (type: {unit_type_elem.text if unit_type_elem else 'unknown'})")
    
    # Extract unit coordinates
    unit_coords = _extract_coordinates(unit_element)
    if len(unit_coords) == 0:
        raise ValueError(f"Unit '{unit_id}' has no coordinates")
    
    unit_position = unit_coords[0]
    
    # Extract unit name
    unit_name_elem = unit_element.find(".//c2-attributes//name")
    unit_name = unit_name_elem.text if unit_name_elem is not None else ""
    
    # Extract unit affiliation from SIDC
    sidc_elem = unit_element.find("symbolcode")
    if sidc_elem is None:
        raise ValueError(f"Unit '{unit_id}' has no symbolcode")
    
    sidc = sidc_elem.text.replace("2525b:", "")
    is_friendly = _is_friendly_sidc(sidc)
    is_hostile = _is_hostile_sidc(sidc)
    
    if not is_friendly and not is_hostile:
        raise ValueError(f"Unit '{unit_id}' has unknown affiliation (SIDC: {sidc})")
    
    affiliation = "friendly" if is_friendly else "hostile"
    
    # Find all Phase Lines and Boundaries
    # Regex: .{4}GLP.{9} (Phase Line) or .{4}GLB.{9} (Boundary)
    lines = []
    for obj in root.findall(".//object"):
        obj_type_elem = obj.find(".//c2-attributes//type")
        if obj_type_elem is None:
            continue
        
        obj_type = obj_type_elem.text
        if obj_type not in ["TACTICAL_GRAPHIC", "ORGANISATIONAL_BOUNDARY"]:
            continue
        
        obj_sidc_elem = obj.find("symbolcode")
        if obj_sidc_elem is None:
            continue
        
        obj_sidc = obj_sidc_elem.text.replace("2525b:", "")
        
        # Extract line name
        obj_name_elem = obj.find(".//c2-attributes//name")
        obj_name = obj_name_elem.text if obj_name_elem is not None else ""
        
        # Check if it's a Phase Line or Boundary
        if not (_is_phase_line_or_boundary(obj_sidc)):
            continue
        
        # Check affiliation match
        obj_is_friendly = _is_friendly_sidc(obj_sidc)
        obj_is_hostile = _is_hostile_sidc(obj_sidc)
        
        # Only consider lines with matching affiliation
        if is_friendly and obj_is_friendly:
            coords = _extract_coordinates(obj)
            if len(coords) >= 2:
                lines.append({
                    "id": obj.get("id"),
                    "sidc": obj_sidc,
                    "name": obj_name,
                    "coordinates": coords,
                    "affiliation": "friendly"
                })
        elif is_hostile and obj_is_hostile:
            coords = _extract_coordinates(obj)
            if len(coords) >= 2:
                lines.append({
                    "id": obj.get("id"),
                    "sidc": obj_sidc,
                    "name": obj_name,
                    "coordinates": coords,
                    "affiliation": "hostile"
                })
    
    # Calculate AOR using ray-casting algorithm
    # Pass lines with full info to _find_nearby_lines
    aor_result = _find_nearby_lines_with_info(unit_position, lines)

    geojson_content = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [round(unit_position[0], 6), round(unit_position[1], 6)]
                },
                "properties": {
                    "sidc": obj_sidc,
                    "name": unit_name,
                    "sw_id": unit_id
                }
             }
        ],
        "message":""
    }

    if len(aor_result["boundary_lines"]) == 0:
        geojson_content["message"] = "No AOR could be calculated"
        

    for boundary_line in aor_result["boundary_lines"]:
        coords = []
        for coordinate in boundary_line["coordinates"]:
            coords.append([round(coordinate[0]["lat"], 6),(round(coordinate[0]["lon"], 6))])

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": coords
            },
            "properties": {
                "sidc": boundary_line["sidc"],
                "name": boundary_line["name"],
                "sw_id": ""
            }
        }
        geojson_content["features"].append(feature)

    return geojson_content

def _extract_coordinates(element: ET.Element) -> List[Tuple[float, float]]:
    """Extract lat/lon coordinates from XML geometry element."""
    coords = []
    geometry_elem = element.find("geometry")
    if geometry_elem is None:
        return coords
    
    # Find all geo-position elements
    for pos_elem in geometry_elem.findall(".//geo-position"):
        lat_elem = pos_elem.find("latitude")
        lon_elem = pos_elem.find("longitude")
        
        if lat_elem is not None and lon_elem is not None:
            try:
                lat = float(lat_elem.text)
                lon = float(lon_elem.text)
                coords.append((lat, lon))
            except (ValueError, TypeError):
                continue
    
    return coords

def _is_friendly_sidc(sidc: str) -> bool:
    """Check if SIDC indicates friendly affiliation (position 2, index 1 = 'F')."""
    if len(sidc) < 15:
        return False
    return sidc[1] == 'F'

def _is_hostile_sidc(sidc: str) -> bool:
    """Check if SIDC indicates hostile affiliation (position 2, index 1 = 'H')."""
    if len(sidc) < 15:
        return False
    return sidc[1] == 'H'

def _is_phase_line_or_boundary(sidc: str) -> bool:
    """Check if SIDC matches Phase Line (.{4}GLP.{9}) or Boundary (.{4}GLB.{9})."""
    if len(sidc) < 15:
        return False
    # Check positions 5-7 (index 4-6) for GLP or GLB
    return sidc[4:7] in ["GLP", "GLB"]

def _find_nearby_lines_with_info(position: Tuple[float, float], lines: List[Dict]) -> Dict[str, Any]:
    """
    Find AOR boundary using ray-casting algorithm.
    
    Strategy:
    1. Cast 72 rays (every 5°) from unit position
    2. For each ray: find all intersections with line segments
    3. Keep only nearest intersection per ray
    4. Add visible corner points from lines
    5. Group points by their original line
    6. Return boundary lines with intersection points and corner points
    
    Returns:
        Dictionary with:
        - 'boundary_lines': List of dicts with sidc, name, and points (intersections + visible corners)
    """
    # Track points by line ID
    # Store: line_id -> {sidc, name, points: list of tuples, original_line: dict with original coordinates}
    boundary_lines_dict = {}
    
    # Store original lines for reference
    original_lines = {line["id"]: line for line in lines}
    
    # Step 1 & 2: Cast 72 rays (every 5°)
    max_distance_deg = 1.0  # ~111 km range
    
    for angle_deg in range(0, 360, 5):
        angle_rad = math.radians(angle_deg)
        
        # Calculate ray direction
        dx = math.sin(angle_rad)
        dy = math.cos(angle_rad)
        
        # Calculate ray endpoint
        ray_end = (
            position[0] + dy * max_distance_deg,
            position[1] + dx * max_distance_deg
        )
        
        nearest_intersection = None
        nearest_distance = float('inf')
        nearest_line_id = None
        
        # Check all lines for intersections
        for line in lines:
            line_coords = line["coordinates"]
            if len(line_coords) < 2:
                continue
            
            # Check each segment of the line
            for i in range(1, len(line_coords)):
                segment_start = line_coords[i - 1]
                segment_end = line_coords[i]
                
                # Calculate intersection
                intersection = _calculate_segment_intersection(
                    position, ray_end, segment_start, segment_end
                )
                
                if intersection is not None:
                    distance = _calculate_distance(position, intersection)
                    
                    # Keep only nearest intersection (>1m to avoid self-intersection)
                    if distance > 0.001 and distance < nearest_distance:
                        nearest_distance = distance
                        nearest_intersection = intersection
                        nearest_line_id = line["id"]
        
        # Store the nearest intersection point with its line info
        if nearest_intersection is not None and nearest_line_id is not None:
            if nearest_line_id not in boundary_lines_dict:
                line_info = next((l for l in lines if l["id"] == nearest_line_id), None)
                if line_info:
                    boundary_lines_dict[nearest_line_id] = {
                        "sidc": line_info["sidc"],
                        "name": line_info["name"],
                        "points": [],  # Use list to preserve order
                        "original_line": original_lines.get(nearest_line_id)  # Store original line for reference
                    }
            
            if nearest_line_id in boundary_lines_dict:
                rounded_coord = (round(nearest_intersection[0], 6), round(nearest_intersection[1], 6))
                # Only add if not already in list
                if rounded_coord not in boundary_lines_dict[nearest_line_id]["points"]:
                    boundary_lines_dict[nearest_line_id]["points"].append(rounded_coord)
    
    # Step 3: Add visible corner points
    for line in lines:
        line_coords = line["coordinates"]
        if len(line_coords) < 2:
            continue
        
        for coord_idx, coord in enumerate(line_coords):
            # Check if this corner is visible from unit position
            is_visible = True
            
            for other_line in lines:
                other_coords = other_line["coordinates"]
                if len(other_coords) < 2:
                    continue
                
                # Check if any segment blocks the view
                for i in range(1, len(other_coords)):
                    seg_start = other_coords[i - 1]
                    seg_end = other_coords[i]
                    
                    # Skip if corner is endpoint of blocking segment
                    if coord == seg_start or coord == seg_end:
                        continue
                    
                    if _do_segments_intersect(position, coord, seg_start, seg_end):
                        is_visible = False
                        break
                
                if not is_visible:
                    break
            
            if is_visible:
                # Add this visible corner to its line
                if line["id"] not in boundary_lines_dict:
                    boundary_lines_dict[line["id"]] = {
                        "sidc": line["sidc"],
                        "name": line["name"],
                        "points": [],  # Use list to preserve order
                        "original_line": line  # Store original line for reference
                    }
                
                # Round coordinates and add to list (avoid duplicates while preserving order)
                rounded_coord = (round(coord[0], 6), round(coord[1], 6))
                if rounded_coord not in boundary_lines_dict[line["id"]]["points"]:
                    boundary_lines_dict[line["id"]]["points"].append(rounded_coord)
    
    # Build boundary_lines list
    # Sort points based on their position along the original line
    boundary_lines = []
    for line_id, line_data in boundary_lines_dict.items():
        line_points = line_data["points"]
        original_line = line_data.get("original_line")
        
        if len(line_points) > 0:
            if original_line and len(original_line["coordinates"]) >= 2:
                # Sort points based on their position along the original line
                sorted_line_points = _sort_points_by_original_line(
                    line_points, 
                    original_line["coordinates"]
                )
            else:
                # Fallback to nearest-neighbor if no original line available
                sorted_line_points = _sort_points_along_line(line_points)
            
            boundary_lines.append({
                "sidc": line_data["sidc"],
                "name": line_data["name"],
                "coordinates": [{"lat": lat, "lon": lon} for lat, lon in sorted_line_points]
            })
    
    # Sort lines into a continuous chain first
    sorted_boundary_lines = _sort_lines_into_chain(boundary_lines, position)
    
    # Then add intersection points to close gaps between consecutive lines
    sorted_boundary_lines = _add_intersection_points_for_gaps(sorted_boundary_lines, lines)
    
    return {
        "boundary_lines": sorted_boundary_lines
    }

def _add_intersection_points_for_gaps(
    boundary_lines: List[Dict],
    original_lines: List[Dict]
) -> List[Dict]:
    """
    Add intersection points between consecutive boundary lines to close gaps.
    
    Uses TWO passes:
    1. First pass: Add intersection points at the END of each line
    2. Second pass: Add intersection points at the START of each line
    
    Args:
        boundary_lines: List of boundary lines (sorted into chain)
        original_lines: List of original lines from XML
    
    Returns:
        Boundary lines with intersection points added to close gaps
    """
    if len(boundary_lines) < 2:
        return boundary_lines
    
    # === FIRST PASS: Add intersection points at the END of each line ===
    result = []
    
    for i in range(len(boundary_lines)):
        current_bl = boundary_lines[i]
        next_bl = boundary_lines[(i + 1) % len(boundary_lines)]
        
        # Add current boundary line coordinates
        new_coords = list(current_bl["coordinates"])
        
        if not current_bl["coordinates"] or not next_bl["coordinates"]:
            result.append({
                "sidc": current_bl["sidc"],
                "name": current_bl["name"],
                "coordinates": new_coords
            })
            continue
        
        gap_point1 = current_bl["coordinates"][-1]
        gap_point2 = next_bl["coordinates"][0]
        
        gap_center = (
            (gap_point1["lat"] + gap_point2["lat"]) / 2,
            (gap_point1["lon"] + gap_point2["lon"]) / 2
        )
        
        # Find ALL pairs of original lines and check if their intersection is near this gap
        best_intersection = None
        best_distance = float('inf')
        
        for ol1 in original_lines:
            for ol2 in original_lines:
                if ol1 is ol2:
                    continue
                
                # Find intersection
                intersection = _find_intersection_of_original_lines(
                    ol1["coordinates"],
                    ol2["coordinates"],
                    gap_point1,
                    gap_point2
                )
                
                if intersection:
                    # Check distance to gap center
                    distance = _calculate_distance(gap_center, intersection)
                    
                    if distance < best_distance and distance < 5.0:  # Within 5km of gap
                        best_distance = distance
                        best_intersection = intersection
        
        # Add best intersection point if found
        if best_intersection:
            rounded = {
                "lat": round(best_intersection[0], 6),
                "lon": round(best_intersection[1], 6)
            }
            
            # Check if not already in line
            is_already_included = any(
                c["lat"] == rounded["lat"] and c["lon"] == rounded["lon"]
                for c in new_coords
            )
            
            if not is_already_included:
                new_coords.append(rounded)
        
        result.append({
            "sidc": current_bl["sidc"],
            "name": current_bl["name"],
            "coordinates": new_coords
        })
    
    # === SECOND PASS: Add intersection points at the START of each line ===
    # This ensures both lines sharing an intersection point have it
    final_result = []
    
    for i in range(len(result)):
        current_bl = result[i]
        prev_bl = result[(i - 1) % len(result)]  # Previous line (wrapping around)
        
        # Start with current line's coordinates
        new_coords = list(current_bl["coordinates"])
        
        if not current_bl["coordinates"] or not prev_bl["coordinates"]:
            final_result.append({
                "sidc": current_bl["sidc"],
                "name": current_bl["name"],
                "coordinates": new_coords
            })
            continue
        
        # Get the gap between previous line's end and current line's start
        gap_point1 = prev_bl["coordinates"][-1]  # End of previous line
        gap_point2 = current_bl["coordinates"][0]  # Start of current line
        
        # Check if previous line already has an intersection point at its end
        # that should also be at our start
        prev_has_intersection_at_end = False
        intersection_to_add = None
        
        if len(prev_bl["coordinates"]) >= 2:
            # Check if the last point of previous line is an intersection point
            # (i.e., it's close to our gap but not the original end point)
            prev_original_end = gap_point2  # This is where we expect the gap
            prev_actual_end = prev_bl["coordinates"][-1]
            
            # Calculate distance between previous line's end and our start
            gap_distance = _calculate_distance(
                (gap_point1["lat"], gap_point1["lon"]),
                (gap_point2["lat"], gap_point2["lon"])
            )
            
            # If gap is significant (>100m), check if we should add intersection at start
            if gap_distance > 0.1:
                # Find intersection point for this gap
                gap_center = (
                    (gap_point1["lat"] + gap_point2["lat"]) / 2,
                    (gap_point1["lon"] + gap_point2["lon"]) / 2
                )
                
                best_intersection = None
                best_distance = float('inf')
                
                for ol1 in original_lines:
                    for ol2 in original_lines:
                        if ol1 is ol2:
                            continue
                        
                        intersection = _find_intersection_of_original_lines(
                            ol1["coordinates"],
                            ol2["coordinates"],
                            gap_point1,
                            gap_point2
                        )
                        
                        if intersection:
                            distance = _calculate_distance(gap_center, intersection)
                            if distance < best_distance and distance < 5.0:
                                best_distance = distance
                                best_intersection = intersection
                
                # If we found an intersection, add it at the START of current line
                if best_intersection:
                    rounded = {
                        "lat": round(best_intersection[0], 6),
                        "lon": round(best_intersection[1], 6)
                    }
                    
                    # Check if not already at the start
                    is_already_at_start = (
                        len(new_coords) > 0 and
                        new_coords[0]["lat"] == rounded["lat"] and
                        new_coords[0]["lon"] == rounded["lon"]
                    )
                    
                    if not is_already_at_start:
                        # Add at the beginning
                        new_coords.insert(0, rounded)
        
        final_result.append({
            "sidc": current_bl["sidc"],
            "name": current_bl["name"],
            "coordinates": new_coords
        })
    
    return final_result

def _add_line_intersections(
    boundary_lines: List[Dict],
    boundary_lines_dict: Dict
) -> List[Dict]:
    """
    Find intersection points between adjacent lines and add them to close gaps.
    
    For each pair of consecutive lines, find where their original geometries intersect
    and add that point to both lines.
    
    Args:
        boundary_lines: List of boundary lines (sorted into chain)
        boundary_lines_dict: Dictionary with original line data
    
    Returns:
        List of boundary lines with intersection points added
    """
    if len(boundary_lines) < 2:
        return boundary_lines
    
    result_lines = []
    
    for i in range(len(boundary_lines)):
        current_line = boundary_lines[i]
        next_line = boundary_lines[(i + 1) % len(boundary_lines)]
        
        # Find original lines - use the one stored in boundary_lines_dict
        # The original_line is already stored there from _find_nearby_lines_with_info
        current_original = None
        next_original = None
        
        # Try to find by matching first/last points to identify which original line was used
        current_coords = current_line["coordinates"]
        next_coords = next_line["coordinates"]
        
        for line_id, line_data in boundary_lines_dict.items():
            orig_line = line_data.get("original_line")
            if not orig_line:
                continue
            
            # Check if this original line matches the current boundary line
            # by comparing if the boundary points are a subset of the original points
            if current_original is None:
                orig_points = set((round(c["lat"], 4) if isinstance(c, dict) else round(c[0], 4), 
                                   round(c["lon"], 4) if isinstance(c, dict) else round(c[1], 4)) 
                                  for c in orig_line["coordinates"])
                
                boundary_points = set((round(c["lat"], 4), round(c["lon"], 4)) for c in current_coords)
                
                # If most boundary points are in original points, this is likely the right line
                if len(boundary_points & orig_points) >= len(boundary_points) * 0.5:
                    current_original = orig_line
            
            if next_original is None:
                orig_points = set((round(c["lat"], 4) if isinstance(c, dict) else round(c[0], 4), 
                                   round(c["lon"], 4) if isinstance(c, dict) else round(c[1], 4)) 
                                  for c in orig_line["coordinates"])
                
                boundary_points = set((round(c["lat"], 4), round(c["lon"], 4)) for c in next_coords)
                
                if len(boundary_points & orig_points) >= len(boundary_points) * 0.5:
                    next_original = orig_line
        
        # Get current line coordinates
        current_coords = list(current_line["coordinates"])
        
        # Try to find intersection between original lines
        intersection_point = None
        
        if current_original and next_original:
            # Find intersection of the two original lines
            intersection_point = _find_intersection_of_original_lines(
                current_original["coordinates"],
                next_original["coordinates"],
                current_coords[-1] if current_coords else None,
                next_line["coordinates"][0] if next_line["coordinates"] else None
            )
        
        # Add intersection point if found
        if intersection_point:
            rounded_intersection = {
                "lat": round(intersection_point[0], 6),
                "lon": round(intersection_point[1], 6)
            }
            
            # Check if not already in line
            is_already_included = any(
                c["lat"] == rounded_intersection["lat"] and c["lon"] == rounded_intersection["lon"]
                for c in current_coords
            )
            
            if not is_already_included:
                current_coords.append(rounded_intersection)
        
        result_lines.append({
            "sidc": current_line["sidc"],
            "name": current_line["name"],
            "coordinates": current_coords
        })
    
    return result_lines

def _find_intersection_of_original_lines(
    line1_coords,
    line2_coords,
    gap_point1=None,
    gap_point2=None
) -> Optional[Tuple[float, float]]:
    """
    Find intersection point between two original lines.
    
    Searches all segment pairs to find an intersection.
    If gap points are provided, prefers intersections close to the gap.
    
    Args:
        line1_coords: Coordinates of first original line
        line2_coords: Coordinates of second original line
        gap_point1: End point of first boundary line (optional)
        gap_point2: Start point of second boundary line (optional)
    
    Returns:
        Intersection point as (lat, lon) tuple, or None if not found
    """
    if len(line1_coords) < 2 or len(line2_coords) < 2:
        return None
    
    # Calculate gap center if gap points provided
    gap_center = None
    if gap_point1 and gap_point2:
        # Handle both dict and tuple formats
        if isinstance(gap_point1, dict):
            lat1, lon1 = float(gap_point1["lat"]), float(gap_point1["lon"])
        else:
            lat1, lon1 = float(gap_point1[0]), float(gap_point1[1])
        
        if isinstance(gap_point2, dict):
            lat2, lon2 = float(gap_point2["lat"]), float(gap_point2["lon"])
        else:
            lat2, lon2 = float(gap_point2[0]), float(gap_point2[1])
        
        gap_center = (
            (lat1 + lat2) / 2,
            (lon1 + lon2) / 2
        )
    
    best_intersection = None
    best_distance = float('inf')
    
    # Try all segment pairs
    for i in range(len(line1_coords) - 1):
        for j in range(len(line2_coords) - 1):
            # Get segment 1
            seg1_start = line1_coords[i]
            seg1_end = line1_coords[i + 1]
            
            if isinstance(seg1_start, dict):
                seg1_start = (float(seg1_start["lat"]), float(seg1_start["lon"]))
                seg1_end = (float(seg1_end["lat"]), float(seg1_end["lon"]))
            else:
                seg1_start = (float(seg1_start[0]), float(seg1_start[1]))
                seg1_end = (float(seg1_end[0]), float(seg1_end[1]))
            
            # Get segment 2
            seg2_start = line2_coords[j]
            seg2_end = line2_coords[j + 1]
            
            if isinstance(seg2_start, dict):
                seg2_start = (float(seg2_start["lat"]), float(seg2_start["lon"]))
                seg2_end = (float(seg2_end["lat"]), float(seg2_end["lon"]))
            else:
                seg2_start = (float(seg2_start[0]), float(seg2_start[1]))
                seg2_end = (float(seg2_end[0]), float(seg2_end[1]))
            
            # Find intersection
            intersection = _calculate_segment_intersection(seg1_start, seg1_end, seg2_start, seg2_end)
            
            if intersection:
                # Check distance to gap center
                if gap_center:
                    distance = _calculate_distance(gap_center, intersection)
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_intersection = intersection
                else:
                    # No gap center, just return first intersection found
                    return intersection
    
    return best_intersection

def _calculate_segment_intersection(
    p1: Tuple[float, float], p2: Tuple[float, float],
    p3: Tuple[float, float], p4: Tuple[float, float]
) -> Optional[Tuple[float, float]]:
    """
    Calculate intersection point of line segments P1-P2 and P3-P4.
    Returns None if no intersection or segments are parallel.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    
    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    
    if abs(denom) < 1e-10:
        return None  # Parallel lines
    
    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
    ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom
    
    if 0 <= ua <= 1 and 0 <= ub <= 1:
        # Intersection within both segments
        x = x1 + ua * (x2 - x1)
        y = y1 + ua * (y2 - y1)
        return (x, y)
    
    return None  # No intersection within segments

def _do_segments_intersect(
    a: Tuple[float, float], b: Tuple[float, float],
    c: Tuple[float, float], d: Tuple[float, float],
    tolerance: float = 1e-6
) -> bool:
    """
    Check if line segments AB and CD truly intersect (not just touch).
    Touching at endpoints doesn't count as intersection.
    """
    def orientation(o: Tuple[float, float], p: Tuple[float, float], q: Tuple[float, float]) -> float:
        return (p[0] - o[0]) * (q[1] - o[1]) - (p[1] - o[1]) * (q[0] - o[0])
    
    o1 = orientation(a, b, c)
    o2 = orientation(a, b, d)
    o3 = orientation(c, d, a)
    o4 = orientation(c, d, b)
    
    eps = 1e-10
    
    # General case: true intersection (segments cross)
    if o1 * o2 < 0 and o3 * o4 < 0:
        return True
    
    # Special cases: touching at endpoints → ignore (not a true intersection)
    if abs(o1) < eps and _point_on_segment(a, b, c, tolerance):
        return False
    if abs(o2) < eps and _point_on_segment(a, b, d, tolerance):
        return False
    if abs(o3) < eps and _point_on_segment(c, d, a, tolerance):
        return False
    if abs(o4) < eps and _point_on_segment(c, d, b, tolerance):
        return False
    
    return False

def _point_on_segment(
    a: Tuple[float, float], b: Tuple[float, float],
    c: Tuple[float, float], tolerance: float
) -> bool:
    """Check if point C lies on segment AB (within tolerance)."""
    def dist(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    
    return abs(dist(a, c) + dist(c, b) - dist(a, b)) < tolerance

def _calculate_distance(p1, p2) -> float:
    """
    Calculate distance between two coordinates in km (Haversine formula).
    
    Args:
        p1: First point (can be tuple or dict)
        p2: Second point (can be tuple or dict)
    """
    R = 6371.0  # Earth radius in km
    
    # Handle both tuple and dict formats
    if isinstance(p1, dict):
        lat1, lon1 = math.radians(float(p1["lat"])), math.radians(float(p1["lon"]))
    else:
        lat1, lon1 = math.radians(float(p1[0])), math.radians(float(p1[1]))
    
    if isinstance(p2, dict):
        lat2, lon2 = math.radians(float(p2["lat"])), math.radians(float(p2["lon"]))
    else:
        lat2, lon2 = math.radians(float(p2[0])), math.radians(float(p2[1]))
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c

def _sort_points_clockwise(points: List[Tuple[float, float]], center: Tuple[float, float]) -> List[Tuple[float, float]]:
    """
    Sort points in clockwise order around a center point.
    
    Args:
        points: List of (lat, lon) tuples
        center: Center point as (lat, lon)
    
    Returns:
        List of points sorted clockwise around center
    """
    if len(points) <= 2:
        return points
    
    # Remove duplicates
    unique_points = list(set(points))
    
    if len(unique_points) <= 2:
        return unique_points
    
    # Calculate angle from center for each point
    def angle_from_center(point):
        # math.atan2 returns angle in radians from -π to π
        # We want clockwise sorting, so negate the angle
        lat_diff = point[0] - center[0]
        lon_diff = point[1] - center[1]
        return -math.atan2(lon_diff, lat_diff)
    
    # Sort by angle
    sorted_points = sorted(unique_points, key=angle_from_center)
    
    return sorted_points

def _sort_points_along_line(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Sort points along a line to form a continuous path.
    Tests all possible start points and returns the ordering with the shortest total path length.
    """
    if len(points) <= 2:
        return points
    
    best_sorted = None
    best_total_length = float('inf')
    
    # Try starting from each point
    for start_idx in range(len(points)):
        # Nearest-neighbor sort starting from this point
        sorted_points = _nearest_neighbor_from_index(points, start_idx)
        
        # Calculate total path length
        total_length = sum(
            _calculate_distance(sorted_points[i], sorted_points[i+1])
            for i in range(len(sorted_points) - 1)
        )
        
        # Keep the best (shortest) ordering
        if total_length < best_total_length:
            best_total_length = total_length
            best_sorted = sorted_points
    
    return best_sorted if best_sorted is not None else points

def _nearest_neighbor_from_index(points: List[Tuple[float, float]], start_idx: int) -> List[Tuple[float, float]]:
    """
    Sort points using nearest-neighbor algorithm starting from given index.
    Always moves forward to the nearest unvisited point.
    """
    if len(points) <= 2:
        return points
    
    sorted_points = [points[start_idx]]
    remaining = points[:start_idx] + points[start_idx+1:]
    
    while remaining:
        last_point = sorted_points[-1]
        # Find nearest point to last_point
        nearest = min(remaining, key=lambda p: _calculate_distance(last_point, p))
        sorted_points.append(nearest)
        remaining.remove(nearest)
    
    return sorted_points

def _sort_points_by_original_line(
    points: List[Tuple[float, float]], 
    original_coords: List[Tuple[float, float]]
) -> List[Tuple[float, float]]:
    """
    Sort points based on their position along the original line.
    
    For each point, find the closest point on the original line segments.
    Use the distance along the line (cumulative segment length) as sorting key.
    
    Args:
        points: List of (lat, lon) tuples to sort
        original_coords: List of (lat, lon) tuples from original line
    
    Returns:
        Points sorted along the original line
    """
    if len(points) < 2 or len(original_coords) < 2:
        return points
    
    # Precompute cumulative distances along original line
    cumulative_distances = [0.0]
    for i in range(1, len(original_coords)):
        dist = _calculate_distance(original_coords[i-1], original_coords[i])
        cumulative_distances.append(cumulative_distances[-1] + dist)
    
    total_line_length = cumulative_distances[-1]
    
    def distance_along_line(point: Tuple[float, float]) -> float:
        """
        Find the distance from line start to the closest point on the line.
        Uses perpendicular projection onto line segments.
        """
        min_distance = float('inf')
        closest_distance_along = 0.0
        
        for i in range(1, len(original_coords)):
            seg_start = original_coords[i-1]
            seg_end = original_coords[i]
            
            # Project point onto line segment
            projected = _project_point_on_segment(point, seg_start, seg_end)
            
            # Calculate distance from point to projection
            dist_to_line = _calculate_distance(point, projected)
            
            # Calculate distance along line to projection
            if i == 1:
                dist_along_seg = _calculate_distance(seg_start, projected)
            else:
                dist_along_seg = cumulative_distances[i-1] + _calculate_distance(seg_start, projected)
            
            # Use combined metric: prefer points close to the line
            # If multiple projections are equally close, use the first one
            if dist_to_line < min_distance:
                min_distance = dist_to_line
                closest_distance_along = dist_along_seg
        
        return closest_distance_along
    
    # Sort points by their distance along the original line
    sorted_points = sorted(points, key=distance_along_line)
    
    return sorted_points

def _project_point_on_segment(
    point,
    seg_start,
    seg_end
) -> Tuple[float, float]:
    """
    Project a point onto a line segment.
    Returns the closest point on the segment to the given point.
    
    Args:
        point: Point to project (can be tuple or dict)
        seg_start: Segment start (can be tuple or dict)
        seg_end: Segment end (can be tuple or dict)
    """
    # Handle both tuple and dict formats for point
    if isinstance(point, dict):
        x0 = float(point["lat"])
        y0 = float(point["lon"])
    else:
        x0 = float(point[0])
        y0 = float(point[1])
    
    # Handle both tuple and dict formats for seg_start/seg_end
    if isinstance(seg_start, dict):
        x1 = float(seg_start["lat"])
        y1 = float(seg_start["lon"])
    else:
        x1 = float(seg_start[0])
        y1 = float(seg_start[1])
    
    if isinstance(seg_end, dict):
        x2 = float(seg_end["lat"])
        y2 = float(seg_end["lon"])
    else:
        x2 = float(seg_end[0])
        y2 = float(seg_end[1])
    
    # Vector from start to end
    dx = x2 - x1
    dy = y2 - y1
    
    # Handle degenerate case (segment is a point)
    if dx == 0 and dy == 0:
        return seg_start if isinstance(seg_start, tuple) else (x1, y1)
    
    # Calculate projection parameter t
    # t = ((P - A) · (B - A)) / |B - A|²
    t = ((x0 - x1) * dx + (y0 - y1) * dy) / (dx * dx + dy * dy)
    
    # Clamp t to [0, 1] to stay on segment
    t = max(0, min(1, t))
    
    # Return projected point
    return (x1 + t * dx, y1 + t * dy)

def _sort_lines_into_chain(boundary_lines: List[Dict], position: Tuple[float, float]) -> List[Dict]:
    """
    Sort boundary lines into a continuous chain by connecting endpoints.
    
    Strategy:
    1. Clean up lines (remove duplicates, ensure consistent format)
    2. Intelligently select start line (southwest-most midpoint)
    3. Use greedy algorithm to connect lines (with gap handling)
    4. Ensure all lines have consistent orientation (clockwise)
    5. Check if polygon is closed (optional, for validation)
    
    Args:
        boundary_lines: List of line dicts with 'coordinates' key
        position: Unit position (used to determine line direction)
    
    Returns:
        List of lines sorted into a continuous chain, all oriented clockwise
    """
    if len(boundary_lines) == 0:
        return boundary_lines
    
    # Step 1: Clean up lines - remove duplicates and lines with < 2 unique points
    cleaned_lines = []
    for line in boundary_lines:
        coords = line["coordinates"]
        if len(coords) < 2:
            continue
        
        # Remove consecutive duplicate points
        unique_coords = []
        for coord in coords:
            point = (coord["lat"], coord["lon"])
            if not unique_coords or (unique_coords[-1]["lat"], unique_coords[-1]["lon"]) != point:
                unique_coords.append(coord)
        
        # Check if first and last point are the same (closed polygon)
        if len(unique_coords) >= 3:
            first_point = (unique_coords[0]["lat"], unique_coords[0]["lon"])
            last_point = (unique_coords[-1]["lat"], unique_coords[-1]["lon"])
            if first_point == last_point:
                # Remove the last point to avoid closed loop
                unique_coords = unique_coords[:-1]
        
        # Only keep lines with at least 2 unique points
        if len(unique_coords) >= 2:
            cleaned_lines.append({
                "sidc": line["sidc"],
                "name": line["name"],
                "coordinates": unique_coords
            })
    
    if len(cleaned_lines) <= 1:
        return cleaned_lines
    
    # Step 2: Intelligently select start line (southwest-most midpoint)
    def get_midpoint_angle(line_idx):
        """Calculate angle of line's midpoint from unit position (southwest = smallest angle)"""
        line = cleaned_lines[line_idx]
        coords = line["coordinates"]
        mid_lat = sum(c["lat"] for c in coords) / len(coords)
        mid_lon = sum(c["lon"] for c in coords) / len(coords)
        
        # Calculate angle from unit position (0 = North, increasing clockwise)
        lat_diff = mid_lat - position[0]
        lon_diff = mid_lon - position[1]
        angle = math.atan2(lon_diff, lat_diff)
        
        # Normalize to [0, 2π]
        if angle < 0:
            angle += 2 * math.pi
        
        return angle
    
    # Find line with southwest-most midpoint (smallest angle after normalization)
    all_indices = list(range(len(cleaned_lines)))
    all_indices.sort(key=get_midpoint_angle)
    start_idx = all_indices[0]
    
    # Step 3: Sort lines into chain using greedy algorithm
    remaining = set(range(len(cleaned_lines)))
    remaining.remove(start_idx)
    sorted_indices = [start_idx]
    has_gap = False
    
    while remaining:
        last_line_idx = sorted_indices[-1]
        last_line = cleaned_lines[last_line_idx]
        last_line_coords = last_line["coordinates"]
        
        if not last_line_coords:
            break
        
        # Get the end point of the last line
        end_point = (last_line_coords[-1]["lat"], last_line_coords[-1]["lon"])
        
        # Find the line whose start or end point is closest to this end point
        best_next_idx = None
        best_distance = float('inf')
        reverse_line = False
        
        for line_idx in remaining:
            line = cleaned_lines[line_idx]
            line_coords = line["coordinates"]
            
            if not line_coords:
                continue
            
            # Check both start and end points (line might need to be reversed)
            start_point = (line_coords[0]["lat"], line_coords[0]["lon"])
            end_point_candidate = (line_coords[-1]["lat"], line_coords[-1]["lon"])
            
            # Distance from last line's end to this line's start
            dist_start = _calculate_distance(end_point, start_point)
            # Distance from last line's end to this line's end (would require reversal)
            dist_end = _calculate_distance(end_point, end_point_candidate)
            
            if dist_start < best_distance:
                best_distance = dist_start
                best_next_idx = line_idx
                reverse_line = False
            if dist_end < best_distance:
                best_distance = dist_end
                best_next_idx = line_idx
                reverse_line = True
        
        # Handle gaps: if no close line found, pick any remaining line
        if best_next_idx is None:
            has_gap = True
            best_next_idx = remaining.pop()
            reverse_line = False
        elif best_distance > 1.0:  # Gap > 1km, warn about it
            has_gap = True
        
        # If line needs to be reversed, reverse its coordinates
        if reverse_line and best_next_idx in remaining:
            cleaned_lines[best_next_idx]["coordinates"] = list(
                reversed(cleaned_lines[best_next_idx]["coordinates"])
            )
        
        sorted_indices.append(best_next_idx)
        if best_next_idx in remaining:
            remaining.remove(best_next_idx)
    
    # Build sorted result
    sorted_lines = [cleaned_lines[idx] for idx in sorted_indices]
    
    # Step 4: Ensure all lines have consistent clockwise orientation
    sorted_lines = _ensure_clockwise_orientation(sorted_lines, position)
    
    # Step 5: Check if polygon is closed (for validation/logging)
    if len(sorted_lines) >= 2:
        first_point = (sorted_lines[0]["coordinates"][0]["lat"], sorted_lines[0]["coordinates"][0]["lon"])
        last_point = (sorted_lines[-1]["coordinates"][-1]["lat"], sorted_lines[-1]["coordinates"][-1]["lon"])
        closure_distance = _calculate_distance(first_point, last_point)
        
        if closure_distance > 0.1:  # > 100m gap
            # Polygon is not closed, but we still return the lines
            # This is okay - the agent can handle open polygons
            pass
    
    # Step 6: Simplify lines using Ramer-Douglas-Peucker algorithm
    simplified_lines = _simplify_all_lines(sorted_lines)
    
    return simplified_lines

def _ensure_clockwise_orientation(boundary_lines: List[Dict], position: Tuple[float, float]) -> List[Dict]:
    """
    Ensure all boundary lines have consistent clockwise orientation around the unit position.
    
    Strategy:
    1. Calculate the signed area of the polygon formed by all line endpoints
    2. If area is negative (counter-clockwise), reverse all lines
    3. Also ensure each individual line flows in the general clockwise direction
    
    Args:
        boundary_lines: List of line dicts with 'coordinates' key
        position: Unit position (center point)
    
    Returns:
        List of lines with consistent clockwise orientation
    """
    if len(boundary_lines) == 0:
        return boundary_lines
    
    # Collect all points in order
    all_points = []
    for line in boundary_lines:
        for coord in line["coordinates"]:
            all_points.append((coord["lat"], coord["lon"]))
    
    if len(all_points) < 3:
        return boundary_lines
    
    # Calculate signed area using Shoelace formula
    # Positive area = clockwise, Negative area = counter-clockwise
    signed_area = 0.0
    n = len(all_points)
    
    for i in range(n):
        lat1, lon1 = all_points[i]
        lat2, lon2 = all_points[(i + 1) % n]
        signed_area += (lon2 - lon1) * (lat2 + lat1)
    
    # If signed area is negative, the polygon is counter-clockwise
    # We need to reverse all lines to make it clockwise
    if signed_area < 0:
        # Reverse the order of lines AND reverse each line's coordinates
        boundary_lines.reverse()
        for line in boundary_lines:
            line["coordinates"] = list(reversed(line["coordinates"]))
    
    # Additional check: ensure each line flows in clockwise direction
    # A line flows clockwise if its midpoint angle increases along the line
    for line in boundary_lines:
        coords = line["coordinates"]
        if len(coords) < 2:
            continue
        
        # Calculate angles of first and last point from unit
        first_angle = math.atan2(coords[0]["lon"] - position[1], coords[0]["lat"] - position[0])
        last_angle = math.atan2(coords[-1]["lon"] - position[1], coords[-1]["lat"] - position[0])
        
        # Normalize to [0, 2π]
        if first_angle < 0:
            first_angle += 2 * math.pi
        if last_angle < 0:
            last_angle += 2 * math.pi
        
        # For clockwise orientation, angles should generally increase
        # But we need to handle the wrap-around at 2π
        angle_diff = last_angle - first_angle
        
        # If angle差 is significantly negative (and not just wrap-around), reverse the line
        if angle_diff < -math.pi:
            # This is likely a wrap-around case, don't reverse
            continue
        elif angle_diff < 0 and abs(angle_diff) > 0.5:  # More than ~30 degrees backward
            # Check if reversing would be better
            reversed_first_angle = math.atan2(coords[-1]["lon"] - position[1], coords[-1]["lat"] - position[0])
            reversed_last_angle = math.atan2(coords[0]["lon"] - position[1], coords[0]["lat"] - position[0])
            
            if reversed_first_angle < 0:
                reversed_first_angle += 2 * math.pi
            if reversed_last_angle < 0:
                reversed_last_angle += 2 * math.pi
            
            reversed_diff = reversed_last_angle - reversed_first_angle
            
            if reversed_diff > angle_diff:
                line["coordinates"] = list(reversed(line["coordinates"]))
    
    return boundary_lines

def _simplify_all_lines(boundary_lines: List[Dict]) -> List[Dict]:
    """
    Simplify all boundary lines using Ramer-Douglas-Peucker algorithm.
    Tolerance is calculated as 0.5% of the maximum AOR dimension.
    
    Args:
        boundary_lines: List of line dicts with 'coordinates' key
    
    Returns:
        List of lines with simplified coordinates
    """
    if len(boundary_lines) == 0:
        return boundary_lines
    
    # Collect all points to calculate AOR dimension
    all_points = []
    for line in boundary_lines:
        for coord in line["coordinates"]:
            all_points.append((coord["lat"], coord["lon"]))
    
    if len(all_points) < 2:
        return boundary_lines
    
    # Calculate maximum dimension of AOR
    max_dimension = 0.0
    min_lat = min(p[0] for p in all_points)
    max_lat = max(p[0] for p in all_points)
    min_lon = min(p[1] for p in all_points)
    max_lon = max(p[1] for p in all_points)
    
    # Use diagonal of bounding box as max dimension
    max_dimension = _calculate_distance(
        (min_lat, min_lon),
        (max_lat, max_lon)
    )
    
    # Tolerance: 0.5% of max dimension (in km)
    tolerance_km = max_dimension * 0.005
    
    # Convert tolerance from km to degrees (approximate)
    # 1 degree latitude ≈ 111 km
    tolerance_deg = tolerance_km / 111.0
    
    # Simplify each line
    simplified_lines = []
    for line in boundary_lines:
        coords = line["coordinates"]
        if len(coords) <= 2:
            simplified_lines.append(line)
            continue
        
        # Convert to tuple format for RDP algorithm
        points = [(c["lat"], c["lon"]) for c in coords]
        
        # Apply Ramer-Douglas-Peucker
        simplified_points = _ramer_douglas_peucker(points, tolerance_deg)
        
        # Convert back to dict format
        simplified_coords = [{"lat": lat, "lon": lon} for lat, lon in simplified_points]
        
        # Only keep lines with at least 2 points
        if len(simplified_coords) >= 2:
            simplified_lines.append({
                "sidc": line["sidc"],
                "name": line["name"],
                "coordinates": simplified_coords
            })
    
    return simplified_lines

def _ramer_douglas_peucker(points: List[Tuple[float, float]], epsilon: float) -> List[Tuple[float, float]]:
    """
    Ramer-Douglas-Peucker algorithm for line simplification.
    
    Args:
        points: List of (lat, lon) tuples
        epsilon: Maximum perpendicular distance (in degrees) for a point to be kept
    
    Returns:
        Simplified list of points
    """
    if len(points) <= 2:
        return points
    
    # Find the point with the maximum distance from the line segment
    max_distance = 0.0
    max_index = 0
    
    start_point = points[0]
    end_point = points[-1]
    
    for i in range(1, len(points) - 1):
        distance = _perpendicular_distance(points[i], start_point, end_point)
        if distance > max_distance:
            max_distance = distance
            max_index = i
    
    # If max distance is greater than epsilon, recursively simplify
    if max_distance > epsilon:
        # Split at max_index
        left_points = _ramer_douglas_peucker(points[:max_index + 1], epsilon)
        right_points = _ramer_douglas_peucker(points[max_index:], epsilon)
        
        # Combine results (avoid duplicate at max_index)
        return left_points[:-1] + right_points
    else:
        # Return only start and end points
        return [start_point, end_point]

def _perpendicular_distance(
    point: Tuple[float, float],
    line_start: Tuple[float, float],
    line_end: Tuple[float, float]
) -> float:
    """
    Calculate perpendicular distance from a point to a line segment.
    Uses Haversine formula for accurate distance calculation.
    
    Args:
        point: Point to measure distance from
        line_start: Start of line segment
        line_end: End of line segment
    
    Returns:
        Perpendicular distance in km
    """
    x0, y0 = point
    x1, y1 = line_start
    x2, y2 = line_end
    
    # Handle degenerate case (line segment is a point)
    if x1 == x2 and y1 == y2:
        return _calculate_distance(point, line_start)
    
    # Calculate projection parameter t
    # t = ((P - A) · (B - A)) / |B - A|²
    dx = x2 - x1
    dy = y2 - y1
    
    t = ((x0 - x1) * dx + (y0 - y1) * dy) / (dx * dx + dy * dy)
    
    # Clamp t to [0, 1] to stay on line segment
    t = max(0, min(1, t))
    
    # Find closest point on line segment
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    
    # Calculate distance from point to closest point
    return _calculate_distance(point, (closest_x, closest_y))
