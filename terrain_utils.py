import requests
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from typing import List, Tuple, Dict
import streamlit as st
import time

# === MULTIPLE ELEVATION API SOURCES ===
# Primary: Open-Meteo (Fast, reliable)
# Fallback 1: Open-Elevation (Self-hosted, no rate limits)
# Fallback 2: OpenTopoData (SRTM data)

ELEVATION_APIS = [
    {
        "name": "Open-Meteo",
        "url": "https://api.open-meteo.com/v1/elevation",
        "format": "query",  # Uses query params
        "lat_param": "latitude",
        "lon_param": "longitude",
        "result_key": "elevation"
    },
    {
        "name": "Open-Elevation",
        "url": "https://api.open-elevation.com/api/v1/lookup",
        "format": "json",  # Uses POST with JSON body
        "result_key": "results"
    },
    {
        "name": "OpenTopoData",
        "url": "https://api.opentopodata.org/v1/srtm30m",
        "format": "query",
        "lat_param": "locations",  # Special format: "lat,lon|lat,lon"
        "result_key": "results"
    }
]

# Create a session for connection pooling
session = requests.Session()
session.headers.update({'User-Agent': 'TerrainAnalyzer/1.0'})

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_elevation_openmeteo(lats_str: str, lons_str: str) -> List[float]:
    """Fetch from Open-Meteo API"""
    try:
        params = {"latitude": lats_str, "longitude": lons_str}
        response = session.get(ELEVATION_APIS[0]["url"], params=params, timeout=15)
        if response.status_code == 200:
            data = response.json()
            return data.get('elevation', [])
        elif response.status_code == 429:
            time.sleep(2)
            return None
    except Exception as e:
        print(f"Open-Meteo error: {e}")
    return None

def fetch_elevation_openelevation(locations: list) -> list:
    """Fetch from Open-Elevation API (POST with JSON) - No caching due to list input"""
    try:
        if not locations:
            return []
        payload = {"locations": [{"latitude": lat, "longitude": lon} for lat, lon in locations]}
        response = session.post(
            ELEVATION_APIS[1]["url"], 
            json=payload, 
            timeout=30,
            headers={"Accept": "application/json", "Content-Type": "application/json"}
        )
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', [])
            if results:
                return [r.get('elevation', 0) for r in results]
    except Exception as e:
        print(f"Open-Elevation error: {e}")
    return None

def fetch_elevation_opentopodata(locations: list) -> list:
    """Fetch from OpenTopoData API (SRTM) - No caching due to list input"""
    try:
        if not locations:
            return []
        # Format: "lat,lon|lat,lon|..." - limit to 100 per request
        locs_str = "|".join([f"{lat},{lon}" for lat, lon in locations[:100]])
        params = {"locations": locs_str}
        response = session.get(ELEVATION_APIS[2]["url"], params=params, timeout=30)
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', [])
            if results:
                return [r.get('elevation', 0) if r.get('elevation') is not None else 0 for r in results]
    except Exception as e:
        print(f"OpenTopoData error: {e}")
    return None

def get_elevations(locations: List[Tuple[float, float]], batch_size: int = 50) -> List[float]:
    """
    Fetches elevations using multiple APIs with automatic fallback.
    Priority: Open-Meteo -> Open-Elevation -> OpenTopoData
    """
    elevations = [None] * len(locations)
    
    # Process in smaller batches for reliability
    for i in range(0, len(locations), batch_size):
        batch = locations[i:i + batch_size]
        batch_elevs = None
        
        # Try Open-Meteo first (fastest)
        lats = ",".join([str(p[0]) for p in batch])
        lons = ",".join([str(p[1]) for p in batch])
        batch_elevs = fetch_elevation_openmeteo(lats, lons)
        
        # Fallback to Open-Elevation
        if batch_elevs is None or len(batch_elevs) != len(batch):
            print("Trying Open-Elevation fallback...")
            batch_elevs = fetch_elevation_openelevation(batch)
        
        # Fallback to OpenTopoData (limit 100 points per request)
        if batch_elevs is None or len(batch_elevs) != len(batch):
            print("Trying OpenTopoData fallback...")
            batch_elevs = fetch_elevation_opentopodata(batch)
        
        # If all APIs failed, use 0 as placeholder (will trigger error in analysis)
        if batch_elevs is None:
            batch_elevs = [None] * len(batch)
        
        # Store results
        for j, elev in enumerate(batch_elevs):
            if i + j < len(elevations):
                elevations[i + j] = elev
        
        # Small delay between batches to avoid rate limits
        time.sleep(0.3)

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
                # Link Budget Parameters
                eirp_dbm: float = 61.0,
                fading_margin_db: float = 8.0,
                rx_sensitivity_dbm: float = -110.0,
                # Advanced Propagation
                prop_model: str = "COST-231 Hata (Urban)",
                rain_rate_mmh: float = 0.0,
                indoor_loss_db: float = 0.0,
                h_beamwidth: float = 65.0,
                azimuth_to_target: float = 0.0,
                site_azimuth: float = 0.0,
                _cache_version: int = 5) -> Dict:
    """
    Advanced RF Line of Sight Analysis with Fresnel Zone, K-Factor, Tilt, 
    COST-231 Hata, Rain Fade, Indoor Loss, and Antenna Pattern support.
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
        margin = 5.0  # 5m safety margin
        required_height_increase = max_obs_height + margin
    
    # ========================================
    # 8. ADVANCED RF PROPAGATION MODELS
    # ========================================
    
    distance_km = total_distance_m / 1000.0
    frequency_ghz = frequency_mhz / 1000.0
    
    # --- A. PATH LOSS MODEL SELECTION ---
    if prop_model == "Free Space (FSPL)":
        # FSPL (dB) = 20*log10(d) + 20*log10(f) + 32.45
        if distance_km > 0:
            path_loss_model_db = 20 * np.log10(distance_km) + 20 * np.log10(frequency_mhz) + 32.45
        else:
            path_loss_model_db = 0
            
    else:
        # COST-231 Hata Model
        # Valid for: 1500-2000 MHz (extended to 3500 for 5G approximation)
        # L = 46.3 + 33.9*log10(f) - 13.82*log10(hb) - a(hm) + (44.9-6.55*log10(hb))*log10(d) + Cm
        
        hb = max(h_start_agl, 30)  # Base station height (min 30m for formula validity)
        hm = max(h_end_agl, 1.5)   # Mobile height
        
        # Mobile antenna height correction factor (medium-small city)
        a_hm = (1.1 * np.log10(frequency_mhz) - 0.7) * hm - (1.56 * np.log10(frequency_mhz) - 0.8)
        
        # Urban/Suburban correction
        if "Urban" in prop_model:
            Cm = 3  # Urban (metropolitan center)
        else:
            Cm = 0  # Suburban
        
        if distance_km > 0.02:  # Min 20m for formula validity
            path_loss_model_db = (46.3 + 33.9 * np.log10(frequency_mhz) 
                                   - 13.82 * np.log10(hb) - a_hm 
                                   + (44.9 - 6.55 * np.log10(hb)) * np.log10(distance_km) 
                                   + Cm)
            
            # 5G Frequency Extension Correction (3.5 GHz is above COST-231 range)
            # Add empirical correction for higher frequencies
            if frequency_mhz > 2000:
                freq_extension_db = 10 * np.log10(frequency_mhz / 2000)
                path_loss_model_db += freq_extension_db
        else:
            path_loss_model_db = 60  # Minimum loss at very short range
    
    fspl_db = path_loss_model_db  # Store for display (legacy name)
    
    # --- B. DIFFRACTION LOSS (Knife-Edge) ---
    diffraction_loss_db = 0.0
    if blocked:
        wavelength_m = 0.2998 / frequency_ghz  # c/f in meters
        d1_m = distance_km * 500  # Approximate midpoint
        d2_m = distance_km * 500
        if d1_m > 0 and d2_m > 0:
            v = max_obs_height * np.sqrt(2 / (wavelength_m * d1_m * d2_m / (d1_m + d2_m)))
            if v > -0.78:
                diffraction_loss_db = 6.9 + 20 * np.log10(np.sqrt((v - 0.1)**2 + 1) + v - 0.1)
                diffraction_loss_db = max(0, min(diffraction_loss_db, 40))
    
    # --- C. RAIN ATTENUATION (ITU-R P.838) ---
    rain_loss_db = 0.0
    if rain_rate_mmh > 0 and distance_km > 0:
        # Simplified ITU-R P.838 for horizontal polarization at 3.5 GHz
        # gamma_R = k * R^alpha (dB/km)
        # For 3.5 GHz, H-pol: k ≈ 0.00175, alpha ≈ 1.308
        k_rain = 0.00175 * (frequency_ghz / 3.5) ** 1.5  # Scale with frequency
        alpha_rain = 1.1 + 0.06 * np.log10(frequency_ghz)
        gamma_r = k_rain * (rain_rate_mmh ** alpha_rain)  # dB/km
        
        # Effective path length (ITU-R P.530)
        d_eff = distance_km / (1 + distance_km / 35)  # Reduction factor for long paths
        rain_loss_db = gamma_r * d_eff
    
    # --- D. ANTENNA HORIZONTAL PATTERN LOSS ---
    antenna_pattern_loss_db = 0.0
    if h_beamwidth > 0 and h_beamwidth < 360:
        # Calculate angular offset from boresight
        angle_offset = abs(azimuth_to_target - site_azimuth)
        if angle_offset > 180:
            angle_offset = 360 - angle_offset
        
        # 3GPP TS 38.901 Antenna Pattern (simplified)
        # A(φ) = -min(12*(φ/φ_3dB)^2, Am)
        # Am = 30 dB (max attenuation), φ_3dB = beamwidth/2
        phi_3db = h_beamwidth / 2
        if angle_offset > 0:
            pattern_atten = min(12 * (angle_offset / phi_3db) ** 2, 30)
            antenna_pattern_loss_db = pattern_atten
    
    # --- E. CLUTTER LOSS (Environment-based) ---
    clutter_loss_db = {0: 0, 5: 3, 10: 8, 20: 15}.get(int(clutter_height_m), 0)
    
    # --- F. TOTAL PATH LOSS (All Components) ---
    total_path_loss_db = (fspl_db 
                          + diffraction_loss_db 
                          + rain_loss_db 
                          + antenna_pattern_loss_db 
                          + clutter_loss_db 
                          + indoor_loss_db 
                          + fading_margin_db)
    
    # Estimated RSRP (dBm)
    # RSRP = EIRP - Path Loss
    estimated_rsrp_dbm = eirp_dbm - total_path_loss_db
    
    # Link Margin
    link_margin_db = estimated_rsrp_dbm - rx_sensitivity_dbm
    
    # Coverage Verdict
    if blocked and diffraction_loss_db > 20:
        coverage_verdict = "No Coverage"
        coverage_quality = 0
    elif estimated_rsrp_dbm >= -80:
        coverage_verdict = "Excellent"
        coverage_quality = 5
    elif estimated_rsrp_dbm >= -95:
        coverage_verdict = "Good"
        coverage_quality = 4
    elif estimated_rsrp_dbm >= -105:
        coverage_verdict = "Fair"
        coverage_quality = 3
    elif estimated_rsrp_dbm >= -115:
        coverage_verdict = "Poor"
        coverage_quality = 2
    else:
        coverage_verdict = "No Coverage"
        coverage_quality = 1
    
    # Maximum Theoretical Range (for this EIRP at RSRP = -105 dBm threshold)
    # Rearranging FSPL: d = 10^((EIRP - RSRP_threshold - 32.45 - clutter - fading) / 20) / f_mhz
    max_range_budget_db = eirp_dbm - (-105) - 32.45 - clutter_loss_db - fading_margin_db
    if max_range_budget_db > 0:
        max_range_km = (10 ** (max_range_budget_db / 20)) / frequency_mhz
    else:
        max_range_km = 0
    
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
        # RF Analysis Results
        "fspl_db": fspl_db,
        "diffraction_loss_db": diffraction_loss_db,
        "rain_loss_db": rain_loss_db,
        "antenna_pattern_loss_db": antenna_pattern_loss_db,
        "indoor_loss_db": indoor_loss_db,
        "clutter_loss_db": clutter_loss_db,
        "total_path_loss_db": total_path_loss_db,
        "estimated_rsrp_dbm": estimated_rsrp_dbm,
        "link_margin_db": link_margin_db,
        "prop_model": prop_model,
        "coverage_verdict": coverage_verdict,
        "coverage_quality": coverage_quality,
        "max_range_km": max_range_km,
        "eirp_dbm": eirp_dbm,
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
