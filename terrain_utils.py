import requests
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from typing import List, Tuple, Dict
import streamlit as st
import concurrent.futures

# Open-Meteo Elevation API (Free, no key required)
OPEN_METEO_ELEVATION_URL = "https://api.open-meteo.com/v1/elevation"

# Create a session for connection pooling to improve performance
session = requests.Session()

import time

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_batch_cached(lats_str: str, lons_str: str) -> List[float]:
    """
    Cached helper to fetch a batch of elevations. 
    Arguments are strings to make them hashable for st.cache_data.
    Uses requests.Session for connection pooling.
    Includes Retry Logic for stability.
    """
    params = {
        "latitude": lats_str,
        "longitude": lons_str
    }
    
    max_retries = 3
    base_delay = 1.0 # seconds
    
    for attempt in range(max_retries):
        try:
            # Use the global session for keep-alive
            response = session.get(OPEN_METEO_ELEVATION_URL, params=params, timeout=10)
            
            # Rate Limit Handling (429)
            if response.status_code == 429:
                wait = float(response.headers.get('Retry-After', 2.0))
                time.sleep(wait)
                continue # Retry
                
            if response.status_code != 200:
                print(f"API returned {response.status_code}: {response.text}")
                response.raise_for_status()
                
            data = response.json()
            return data.get('elevation', [])
            
        except (requests.exceptions.RequestException, ConnectionError) as e:
            print(f"Batch Attempt {attempt+1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(base_delay * (2 ** attempt)) # Exponential Backoff
            else:
                 # Return None so the caller knows it failed after all retries
                return None
    return None

def get_elevations(locations: List[Tuple[float, float]], batch_size: int = 100) -> List[float]:
    """
    Fetches elevations from Open-Meteo API using parallel requests and caching.
    
    Optimizations:
    - ThreadPoolExecutor for parallel processing
    - requests.Session for connection reuse
    - st.cache_data to avoid re-fetching known coordinates
    """
    elevations = [None] * len(locations)
    batches = []
    
    # Prepare batches
    for i in range(0, len(locations), batch_size):
        batch = locations[i:i + batch_size]
        batches.append((i, batch))

    # Parallel Fetch
    # Set to 1 worker (Sequential) to guarantee order and strict rate limiting
    # This might be slower, but it stops the "Unstable Connection" errors.
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future_to_batch = {}
        for i, batch in batches:
            lats = ",".join([str(p[0]) for p in batch])
            lons = ",".join([str(p[1]) for p in batch])
            future = executor.submit(fetch_batch_cached, lats, lons)
            future_to_batch[future] = i
            
            # Gentle pacing between submissions
            time.sleep(0.2) 
        
        for future in concurrent.futures.as_completed(future_to_batch):
            start_idx = future_to_batch[future]
            try:
                result = future.result()
                if result:
                    elevations[start_idx : start_idx + len(result)] = result
            except Exception as e:
                print(f"Thread Error: {e}")

    return elevations

def calculate_earth_curvature_drop(dist_m: float, total_dist_m: float) -> float:
    """
    Calculates the apparent drop in height due to Earth's curvature and refraction.
    Using standard K-factor 1.33 (Effective Earth Radius).
    Drop h = d1 * d2 / (2 * k * R_earth)
    d1 = distance from start
    d2 = distance to end (total - d1)
    """
    if total_dist_m == 0: return 0
    
    R_earth = 6371000 # meters
    k_factor = 1.33 # Standard refraction
    
    # Calculate geometric drop offset relative to the chord
    drop = (dist_m * (total_dist_m - dist_m)) / (2 * k_factor * R_earth)
    return drop

@st.cache_data(ttl=600, show_spinner=False)
def analyze_terrain_profile_v3(lat1: float, lon1: float, lat2: float, lon2: float, 
                h_start_agl: float = 10.0, h_end_agl: float = 10.0,
                name_a: str = "Point A", name_b: str = "Point B", 
                force_interval: int = None,
                frequency_mhz: float = 3500,
                k_factor: float = 1.33,
                antenna_tilt_deg: float = 0.0,
                clutter_height_m: float = 0.0,
                _cache_version: int = 3) -> Dict:
    """
    Advanced RF Line of Sight Analysis with Fresnel Zone, K-Factor, and Tilt support.
    """
    # 1. Generate Path Points
    total_distance_m = geodesic((lat1, lon1), (lat2, lon2)).meters
    if total_distance_m == 0:
         return {"status": "Error", "message": "Start and End points are same."}
    
    # Adaptive Sampling Logic
    # UPDATED: Higher resolution to resolve "jagged" charts
    if force_interval:
        interval = force_interval
    elif total_distance_m < 5000:
        interval = 25 # Very fine for short distances
    elif total_distance_m < 20000:
        interval = 50 
    else:
        interval = 100 # Cap at 100m for long distances
        
    num_points = int(total_distance_m / interval) + 2
    
    # Ensure minimum point count for smooth rendering
    if num_points < 50: num_points = 50
    if num_points > 500: num_points = 500 # Cap for API checks
    
    lats = np.linspace(lat1, lat2, num_points)
    lons = np.linspace(lon1, lon2, num_points)
    points_data = list(zip(lats, lons))
    
    distances = np.linspace(0, total_distance_m / 1000.0, num_points) # km
    dist_m_array = np.linspace(0, total_distance_m, num_points) # meters
    
    # 2. Get Elevations
    elevations = get_elevations(points_data)
    
    # Check for failures (None values)
    if any(e is None for e in elevations):
        # FAIL HARD rather than fake it
        return {
            "status": "Error",
            "message": "Failed to fetch Elevation Data from API (Check Connection/Rate Limits).",
            "blocked": True, # Fail safe
            "max_obstruction_height": 0,
            "obstruction_location": None,
             # Return empty DF so UI doesn't crash but shows nothing
            "dataframe": pd.DataFrame() 
        }
        
    elevs_np = np.array(elevations)
    
    # Apply Clutter Height Offset (Simplified Land Use)
    elevs_with_clutter = elevs_np + clutter_height_m
    
    # 3. LoS Calculation with Custom K-Factor
    start_elev = elevs_np[0] + h_start_agl
    end_elev = elevs_np[-1] + h_end_agl
    
    # Apply Antenna Tilt (affects endpoint height)
    # Positive tilt = downtilt = lower endpoint relative to start
    import math
    tilt_rad = math.radians(antenna_tilt_deg)
    tilt_drop = total_distance_m * math.tan(tilt_rad)
    end_elev_with_tilt = end_elev - tilt_drop
    
    # Linear LoS (Flat Earth reference with Tilt)
    los_linear = np.linspace(start_elev, end_elev_with_tilt, num_points)
    
    # Earth Curvature "Sag" with Custom K-Factor
    def calc_curvature_drop_custom_k(dist_m, total_dist_m, k):
        if total_dist_m == 0: return 0
        R_earth = 6371000
        drop = (dist_m * (total_dist_m - dist_m)) / (2 * k * R_earth)
        return drop
    
    curvature_sag = np.array([calc_curvature_drop_custom_k(d, total_distance_m, k_factor) for d in dist_m_array])
    los_curved = los_linear - curvature_sag
    
    # 4. Fresnel Zone Calculation
    # F1 Radius = 17.32 * sqrt(d1 * d2 / (f * D))
    # where d1, d2 in km, f in GHz, D in km
    frequency_ghz = frequency_mhz / 1000.0
    total_distance_km = total_distance_m / 1000.0
    
    fresnel_radius = []
    for i, d_km in enumerate(distances):
        d1 = d_km
        d2 = total_distance_km - d_km
        if d1 <= 0 or d2 <= 0:
            fresnel_radius.append(0)
        else:
            f1_radius = 17.32 * np.sqrt((d1 * d2) / (frequency_ghz * total_distance_km))
            fresnel_radius.append(f1_radius)
    
    fresnel_radius = np.array(fresnel_radius)
    
    # Fresnel Zone boundaries
    fresnel_upper = los_curved + fresnel_radius
    fresnel_lower = los_curved - fresnel_radius
    
    # 5. Clearance Analysis (considering clutter)
    clearance = los_curved - elevs_with_clutter
    fresnel_clearance = fresnel_lower - elevs_with_clutter
    
    is_obstructing = clearance < 0
    is_fresnel_blocked = fresnel_clearance < 0
    blocked = np.any(is_obstructing)
    fresnel_violated = np.any(is_fresnel_blocked)
    
    # 6. Obstruction Classification (based on terrain gradient)
    # High gradient = Terrain/Hill, Low gradient = Clutter/Building
    terrain_gradient = np.gradient(elevs_np, distances * 1000)  # m per m
    
    max_obs_height = 0.0
    obs_loc = None
    obs_type = "None"
    required_height_increase = 0.0
    
    if blocked:
        min_clearance = np.min(clearance)
        max_obs_idx = np.argmin(clearance)
        max_obs_height = -min_clearance
        obs_loc = points_data[max_obs_idx]
        
        # Classify obstruction
        gradient_at_obs = abs(terrain_gradient[max_obs_idx])
        if gradient_at_obs > 0.1:  # >10% slope
            obs_type = "Terrain/Hill"
        elif clutter_height_m > 0:
            obs_type = "Clutter/Building"
        else:
            obs_type = "Vegetation/Forest"
        
        # 7. Reverse Analysis: Required Height Increase
        # To clear the obstruction, we need to raise the LoS by max_obs_height + margin
        # This can be achieved by raising the start antenna
        margin = 5.0  # 5m safety margin
        required_height_increase = max_obs_height + margin
    
    return {
        "status": "Success",
        "blocked": blocked,
        "fresnel_violated": fresnel_violated,
        "max_obstruction_height": max_obs_height,
        "obstruction_location": obs_loc,
        "obstruction_type": obs_type,
        "required_height_increase": required_height_increase,
        "frequency_mhz": frequency_mhz,
        "k_factor": k_factor,
        "dataframe": pd.DataFrame({
            'lat': lats,
            'lon': lons,
            'distance_km': distances,
            'elevation': elevs_np,
            'elevation_with_clutter': elevs_with_clutter,
            'los_elevation': los_curved,
            'fresnel_upper': fresnel_upper,
            'fresnel_lower': fresnel_lower,
            'fresnel_radius': fresnel_radius,
            'clearance': clearance,
            'is_obstructing': is_obstructing,
            'is_fresnel_blocked': is_fresnel_blocked
        }),
        "start_point": (lat1, lon1, start_elev),
        "end_point": (lat2, lon2, end_elev),
        "names": (name_a, name_b),
        "message": "Analysis Complete"
    }
