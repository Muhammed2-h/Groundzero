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
    # Reduced to 2 workers to prevent Rate Limiting / Instability on free tier
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_to_batch = {}
        for i, batch in batches:
            # Join lats/lons into strings for the API (and for caching key)
            lats = ",".join([str(p[0]) for p in batch])
            lons = ",".join([str(p[1]) for p in batch])
            
            # Submit the cached wrapper
            future = executor.submit(fetch_batch_cached, lats, lons)
            future_to_batch[future] = i
        
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
def analyze_terrain_profile(lat1: float, lon1: float, lat2: float, lon2: float, 
                h_start_agl: float = 10.0, h_end_agl: float = 10.0,
                name_a: str = "Point A", name_b: str = "Point B", 
                force_interval: int = None) -> Dict:
    """
    Analyzes Line of Sight between two points including terrain and earth curvature.
    Features Adaptive Sampling for performance.
    """
    # 1. Generate Path Points
    total_distance_m = geodesic((lat1, lon1), (lat2, lon2)).meters
    if total_distance_m == 0:
         return {"status": "Error", "message": "Start and End points are same."}
    
    # Adaptive Sampling Logic
    if force_interval:
        interval = force_interval
    elif total_distance_m < 10000:
        interval = 50
    elif total_distance_m < 50000:
        interval = 100
    else:
        interval = 250
        
    num_points = int(total_distance_m / interval) + 2
    
    # Vectorized point generation 
    lats = np.linspace(lat1, lat2, num_points)
    lons = np.linspace(lon1, lon2, num_points)
    points_data = list(zip(lats, lons))
    
    distances = np.linspace(0, total_distance_m / 1000.0, num_points) # km
    dist_m_array = np.linspace(0, total_distance_m, num_points) # meters
    
    # 2. Get Elevations
    elevations = get_elevations(points_data)
    
    # Check if we have valid data
    valid_elevs = [e for e in elevations if e is not None]
    
    # Fallback to Simulation if API Fails logic
    if len(valid_elevs) < len(elevations) * 0.5:
        st.warning("⚠️ API connection unstable. Switching to Simulated Terrain for visualization.")
        elevations = []
        for i, d in enumerate(distances):
             sim_h = 10.0 + 100.0 * np.sin(np.pi * i / len(distances))
             elevations.append(sim_h)
    else:
        # Interpolate small gaps
        df_temp = pd.DataFrame({'elev': elevations})
        if df_temp['elev'].isnull().any():
            df_temp['elev'] = df_temp['elev'].interpolate(method='linear').fillna(0)
        elevations = df_temp['elev'].tolist()
    
    elevs_np = np.array(elevations)
    
    # 3. LoS Calculation with Earth Curvature
    # Using dynamic user-provided heights
    
    start_elev = elevs_np[0] + h_start_agl
    end_elev = elevs_np[-1] + h_end_agl
    
    # Linear LoS (Flat Earth reference)
    los_linear = np.linspace(start_elev, end_elev, num_points)
    
    # Earth Curvature "Sag"
    curvature_sag = np.array([calculate_earth_curvature_drop(d, total_distance_m) for d in dist_m_array])
    los_curved = los_linear - curvature_sag
    
    # Clearance Analysis
    clearance = los_curved - elevs_np
    is_obstructing = clearance < 0
    blocked = np.any(is_obstructing)
    
    max_obs_height = 0.0
    obs_loc = None
    if blocked:
        # Find max obstruction
        min_clearance = np.min(clearance)
        max_obs_idx = np.argmin(clearance)
        max_obs_height = -min_clearance
        obs_loc = points_data[max_obs_idx]
    
    return {
        "status": "Success",
        "blocked": blocked,
        "max_obstruction_height": max_obs_height,
        "obstruction_location": obs_loc,
        "dataframe": pd.DataFrame({
            'lat': lats,
            'lon': lons,
            'distance_km': distances,
            'elevation': elevs_np,
            'los_elevation_flat': los_linear,
            'los_elevation': los_curved,
            'earth_curve_drop': curvature_sag,
            'is_obstructing': is_obstructing
        }),
        "start_point": (lat1, lon1, start_elev),
        "end_point": (lat2, lon2, end_elev),
        "names": (name_a, name_b),
        "message": "Analysis Complete"
    }
