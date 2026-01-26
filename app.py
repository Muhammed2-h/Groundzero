import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
from terrain_utils import analyze_terrain_profile_v3
from file_parser import parse_site_data

st.set_page_config(page_title="Line of Sight Terrain Analyzer", layout="wide")

def main():
    st.title("🗺️ Terrain Line of Sight Analyzer")
    # Force Reload Trigger
    st.markdown("""
    Check for terrain obstructions between two geographic points using Open-Elevation data.
    Supports **Manual Entry** and **Locked One-to-Many** modes.
    """)

    # --- Sidebar Controls ---
    st.sidebar.header("Configuration")
    
    # --- SITE DATA IMPORT ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("🌍 Import Site Data")
    
    # Template
    site_template = "Site_ID,Sector_ID,Latitude,Longitude,Azimuth,Height\nSITE001,SEC1,20.5937,78.9629,0,30\nSITE001,SEC2,20.5937,78.9629,120,30\nSITE001,SEC3,20.5937,78.9629,240,30"
    st.sidebar.download_button("📥 Download Template CSV", site_template, "sites_template.csv", "text/csv", help="Download sample CSV format with Site_ID, Lat, Lon, Azimuth.")
    
    site_file = st.sidebar.file_uploader("Upload Sites (CSV/KML/KMZ)", type=["csv", "kml", "kmz"])
    
    # Beam Width for Sector Visualization
    beam_width = st.sidebar.slider("Sector Beam Width (°)", min_value=10, max_value=360, value=65, step=5, help="Width of the antenna sector in degrees.")
    sector_radius_km = st.sidebar.slider("Sector Radius (km)", min_value=0.1, max_value=10.0, value=1.0, step=0.1, help="Visual length of the sector wedge.")
    
    # --- ADVANCED RF CONFIGURATION ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("📡 RF Engineering")
    
    # Frequency Selection (5G n78 Band)
    freq_options = {
        "n78 Low (3.3 GHz)": 3300,
        "n78 Mid (3.5 GHz)": 3500,
        "n78 High (3.8 GHz)": 3800
    }
    selected_freq_label = st.sidebar.selectbox("5G Frequency Band", list(freq_options.keys()), index=1)
    frequency_mhz = freq_options[selected_freq_label]
    
    # K-Factor for Earth Curvature
    k_factor = st.sidebar.slider("K-Factor", min_value=1.0, max_value=1.5, value=1.33, step=0.01, 
                                  help="Standard: 1.33. Higher = more refraction (ducting). Lower = less refraction.")
    
    # Antenna Tilt
    antenna_tilt = st.sidebar.number_input("Antenna Downtilt (°)", min_value=-10.0, max_value=20.0, value=0.0, step=0.5,
                                            help="Mechanical + Electrical Tilt. Positive = Down.")
    
    # Clutter Offset (Simplified Land Use)
    clutter_type = st.sidebar.selectbox("Environment Type", ["Open/Rural", "Suburban", "Urban", "Dense Urban"])
    clutter_offsets = {"Open/Rural": 0, "Suburban": 5, "Urban": 10, "Dense Urban": 20}
    clutter_height = clutter_offsets[clutter_type]
    st.sidebar.caption(f"Clutter Height Offset: +{clutter_height}m")
    
    # --- 5G LINK BUDGET PARAMETERS ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("📶 5G Link Budget")
    
    # TX Power (Typical 5G Macro: 40-46 dBm EIRP)
    tx_power_dbm = st.sidebar.number_input("TX Power (dBm)", min_value=20.0, max_value=60.0, value=43.0, step=1.0,
                                            help="Transmitter power. Typical 5G Macro: 43-46 dBm")
    
    # Antenna Gain
    tx_gain_dbi = st.sidebar.number_input("TX Antenna Gain (dBi)", min_value=0.0, max_value=30.0, value=18.0, step=1.0,
                                           help="Typical 5G Massive MIMO: 18-24 dBi")
    
    # Receiver (UE) Sensitivity
    rx_sensitivity_dbm = st.sidebar.number_input("UE Sensitivity (dBm)", min_value=-130.0, max_value=-80.0, value=-110.0, step=1.0,
                                                  help="Typical 5G UE: -110 to -105 dBm")
    
    # RSRP Thresholds
    st.sidebar.markdown("**Coverage Thresholds**")
    rsrp_excellent = st.sidebar.number_input("Excellent RSRP (dBm)", value=-80.0, step=1.0, disabled=True)
    rsrp_good = st.sidebar.number_input("Good RSRP (dBm)", value=-95.0, step=1.0, disabled=True)
    rsrp_fair = st.sidebar.number_input("Fair RSRP (dBm)", value=-105.0, step=1.0, disabled=True)
    
    # Fading/Shadow Margin
    fading_margin_db = st.sidebar.number_input("Fading Margin (dB)", min_value=0.0, max_value=20.0, value=8.0, step=1.0,
                                                help="Shadow fading margin. Typical: 6-10 dB")
    
    # Calculate EIRP
    eirp_dbm = tx_power_dbm + tx_gain_dbi
    st.sidebar.metric("Effective EIRP", f"{eirp_dbm:.1f} dBm")
    
    # --- ADVANCED PROPAGATION SETTINGS ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔬 Advanced Propagation")
    
    # Propagation Model
    prop_model = st.sidebar.selectbox("Propagation Model", 
                                       ["Free Space (FSPL)", "COST-231 Hata (Urban)", "COST-231 Hata (Suburban)"],
                                       index=1,
                                       help="COST-231 Hata is more accurate for cellular in built-up areas.")
    
    # Rain Rate (for ITU-R P.838 rain attenuation)
    rain_rate_mmh = st.sidebar.slider("Rain Rate (mm/h)", min_value=0.0, max_value=100.0, value=0.0, step=5.0,
                                       help="0 = Clear, 10 = Light, 25 = Moderate, 50 = Heavy, 100 = Extreme")
    
    # Indoor/Outdoor Target
    target_environment = st.sidebar.radio("Target Location", ["Outdoor", "Indoor (Light)", "Indoor (Heavy)", "In-Vehicle"],
                                           horizontal=True)
    indoor_loss_map = {"Outdoor": 0, "Indoor (Light)": 15, "Indoor (Heavy)": 25, "In-Vehicle": 10}
    indoor_loss_db = indoor_loss_map[target_environment]
    
    # Antenna Horizontal Beamwidth
    h_beamwidth = st.sidebar.number_input("Horizontal Beamwidth (°)", min_value=30.0, max_value=360.0, value=65.0, step=5.0,
                                           help="Typical 5G sector: 65°. Affects off-boresight loss.")
    
    if 'site_data' not in st.session_state:
        st.session_state.site_data = None
        
    if site_file:
        df_sites, error = parse_site_data(site_file)
        if df_sites is not None:
            st.session_state.site_data = df_sites
            st.session_state.sites_just_loaded = True # Flag to center map
            st.session_state.force_map_update = True # Critical: Force map to respect this update
            st.sidebar.success(f"Loaded {len(df_sites)} sites!")
            st.rerun()  # Trigger immediate map update
        else:
            st.sidebar.error(error)
    else:
        # File removed -> Clear data
        st.session_state.site_data = None
    
    # --- SITE SEARCH FEATURE ---
    if st.session_state.site_data is not None and len(st.session_state.site_data) > 0:
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔍 Search Sites")
        
        # Search input
        search_query = st.sidebar.text_input("Search by ID or Attribute", placeholder="e.g., SITE001, Mumbai, Sector", key="site_search")
        
        # Initialize search results in session state
        if 'search_results' not in st.session_state:
            st.session_state.search_results = []
        
        if search_query:
            # Search across all columns
            df = st.session_state.site_data
            mask = df.apply(lambda row: row.astype(str).str.contains(search_query, case=False).any(), axis=1)
            matches = df[mask]
            
            st.session_state.search_results = matches.to_dict('records') if not matches.empty else []
            
            if not matches.empty:
                st.sidebar.success(f"Found {len(matches)} site(s)")
                
                # Show results as clickable list
                for i, row in matches.iterrows():
                    site_label = f"📍 {row.get('Site_ID', 'Unknown')}"
                    if st.sidebar.button(site_label, key=f"search_result_{i}", use_container_width=True):
                        # Zoom to this site
                        st.session_state.map_center = [row['Latitude'], row['Longitude']]
                        st.session_state.map_zoom = 16
                        st.session_state.selected_search_site = row.to_dict()
                        st.session_state.force_map_update = True
                        st.rerun()
            else:
                st.sidebar.warning("No sites found matching your search.")
        else:
            st.session_state.search_results = []
            st.session_state.selected_search_site = None
    
    # ----------------------
    if 'locked_point' not in st.session_state:
        st.session_state.locked_point = None
    
    # Output Data Container
    if 'results' not in st.session_state:
        st.session_state.results = []
    
    # SYSTEM UPGRADE: Flush logic to remove stale/incorrect results from previous versions
    CURRENT_VERSION = 3
    if 'app_version' not in st.session_state or st.session_state.app_version < CURRENT_VERSION:
        st.session_state.results = [] # Clear old results (fixes "Hump" vs "Flat" data persistence)
        st.session_state.app_version = CURRENT_VERSION
        st.toast("System updated. Cache cleared.", icon="🔄")
    
    # ----------------------
    # MANUAL MODE LOGIC
    # ----------------------
    # Initialize basic states
    if 'coords_target' not in st.session_state: st.session_state.coords_target = ""

    # Helper for parsing (Global Helper)
    def parse_coords(coord_str):
        try:
            if not coord_str: return None, None
            parts = coord_str.split(',')
            if len(parts) != 2: return None, None
            return float(parts[0].strip()), float(parts[1].strip())
        except ValueError:
            return None, None

    # ----------------------
    # GLOBAL MAP INTERFACE
    # ----------------------
    
    # Auto-Zoom on Input Change
    if 'last_coords_target' not in st.session_state:
        st.session_state.last_coords_target = st.session_state.coords_target
        
    if st.session_state.coords_target != st.session_state.last_coords_target:
        # Value changed from Input Box
        lat_check, lon_check = parse_coords(st.session_state.coords_target)
        if lat_check and lon_check:
            st.session_state.map_center = [lat_check, lon_check]
            st.session_state.map_zoom = 15
            st.session_state.force_map_update = True
        
        # Update last known state
        st.session_state.last_coords_target = st.session_state.coords_target

    c_head, c_layer = st.columns([3, 1])
    with c_head:
        st.subheader("🌍 Map Interface")
        st.caption("Select the Marker Tool 📍 (top-left) to drop a pin.")
    with c_layer:
         map_style = st.selectbox("Style", ["Street", "Satellite", "Terrain"], label_visibility="collapsed")
    
    # State: Track Map Center/Zoom to prevent reset on interaction
    if 'map_center' not in st.session_state:
        st.session_state.map_center = [20.5937, 78.9629]
        st.session_state.map_zoom = 5

    # Base Map Center Logic
    if st.session_state.get('sites_just_loaded', False) and st.session_state.site_data is not None:
        sites = st.session_state.site_data
        st.session_state.map_center = [sites['Latitude'].mean(), sites['Longitude'].mean()]
        st.session_state.map_zoom = 10
        st.session_state.sites_just_loaded = False # Reset flag

    # Configure Tiles
    tiles = "OpenStreetMap"
    attr = None
    if map_style == "Satellite":
        tiles = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
        attr = "Esri World Imagery"
    elif map_style == "Terrain":
        tiles = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}"
        attr = "Esri World Topo"

    m = folium.Map(location=st.session_state.map_center, zoom_start=st.session_state.map_zoom, tiles=tiles, attr=attr)
    
    # === AUTO-FIT BOUNDS TO DATA ===
    # Collect all points to fit
    all_points = []
    
    # Add site locations
    if st.session_state.site_data is not None:
        for _, row in st.session_state.site_data.iterrows():
            all_points.append([row['Latitude'], row['Longitude']])
    
    # Add target location
    if st.session_state.coords_target:
        lat_t, lon_t = parse_coords(st.session_state.coords_target)
        if lat_t and lon_t:
            all_points.append([lat_t, lon_t])
    
    # Add analysis result paths
    if st.session_state.results:
        for res in st.session_state.results:
            raw = res.get("Raw", {})
            df = raw.get("dataframe")
            if df is not None and not df.empty:
                # Add start and end of path
                all_points.append([df['lat'].iloc[0], df['lon'].iloc[0]])
                all_points.append([df['lat'].iloc[-1], df['lon'].iloc[-1]])
    
    # Fit bounds if we have points and this is a fresh load
    if all_points and st.session_state.get('force_map_update', False):
        # Calculate bounds
        lats = [p[0] for p in all_points]
        lons = [p[1] for p in all_points]
        sw = [min(lats) - 0.01, min(lons) - 0.01]  # Southwest corner with padding
        ne = [max(lats) + 0.01, max(lons) + 0.01]  # Northeast corner with padding
        m.fit_bounds([sw, ne])
        st.session_state.force_map_update = False  # Reset flag

    # Draw Imported Sites (if any)
    if st.session_state.site_data is not None:
         # Add markers / Sectors
        for _, row in st.session_state.site_data.iterrows():
            # ... (Site drawing logic remains same, just preserving context) ...
            # Check for Azimuth
            has_azimuth = 'Azimuth' in row and pd.notnull(row['Azimuth'])
            
            if has_azimuth:
                az = float(row['Azimuth'])
                from folium.plugins import SemiCircle
                SemiCircle(
                    location=[row['Latitude'], row['Longitude']],
                    radius=sector_radius_km * 1000,
                    start_angle=az - (beam_width / 2),
                    stop_angle=az + (beam_width / 2),
                    color="blue", fill=True, fill_color="blue", fill_opacity=0.3,
                    popup=f"ID: {row['Site_ID']}",
                    tooltip=f"{row['Site_ID']}"
                ).add_to(m)
            else:
                folium.CircleMarker(
                    location=[row['Latitude'], row['Longitude']],
                    radius=5,
                    popup=f"ID: {row['Site_ID']}",
                    tooltip=str(row['Site_ID']),
                    color="blue", fill=True, fill_color="blue"
                ).add_to(m)
    
    # --- HIGHLIGHT SEARCH RESULTS ON MAP ---
    if st.session_state.get('search_results'):
        for site in st.session_state.search_results:
            folium.Marker(
                location=[site['Latitude'], site['Longitude']],
                popup=f"🔍 <b>{site.get('Site_ID', 'Unknown')}</b><br>{site}",
                tooltip=f"🔍 {site.get('Site_ID', 'Search Result')}",
                icon=folium.Icon(color='orange', icon='star', prefix='fa')
            ).add_to(m)
    
    # --- HIGHLIGHT SELECTED SEARCH SITE ---
    if st.session_state.get('selected_search_site'):
        site = st.session_state.selected_search_site
        # Add pulsing circle for selected site
        folium.CircleMarker(
            location=[site['Latitude'], site['Longitude']],
            radius=20,
            color='yellow',
            fill=True,
            fill_color='yellow',
            fill_opacity=0.5,
            weight=3
        ).add_to(m)
        
        # Add info popup
        folium.Marker(
            location=[site['Latitude'], site['Longitude']],
            popup=f"<b>📍 {site.get('Site_ID', 'Selected')}</b><br>Lat: {site['Latitude']}<br>Lon: {site['Longitude']}",
            icon=folium.Icon(color='red', icon='bullseye', prefix='fa')
        ).add_to(m)

    # --- SYNC MANUAL INPUTS TO MARKERS ---
    # Convert text inputs to markers so they appear when typing
    def sync_input_to_marker(key, marker_key):
        val = st.session_state.get(key, "")
        lat_v, lon_v = parse_coords(val)
        if lat_v and lon_v:
            st.session_state[marker_key] = [lat_v, lon_v]

    # Draw Picked Target
    if st.session_state.coords_target:
        lat_t, lon_t = parse_coords(st.session_state.coords_target)
        if lat_t and lon_t:
             folium.Marker(
                 [lat_t, lon_t], 
                 popup="Target Location", 
                 icon=folium.Icon(color='red', icon='crosshairs', prefix='fa')
             ).add_to(m)
        
    # Draw Analysis Results (if any)
    if st.session_state.results:
        # Draw ALL lines, not just one
        from folium import DivIcon
        
        for i, res in enumerate(st.session_state.results):
            raw = res["Raw"]
            color = "red" if raw["blocked"] else "green"
            
            # Path
            path_coords = [(row['lat'], row['lon']) for _, row in raw['dataframe'].iterrows()]
            folium.PolyLine(path_coords, color=color, weight=3, opacity=0.6).add_to(m)
            
            # Numbered Label at Source (Site)
            # Use the first point of the path (Site Location)
            start_lat, start_lon = path_coords[0]
            
            folium.Marker(
                [start_lat, start_lon],
                icon=DivIcon(
                    icon_size=(150,36),
                    icon_anchor=(0,0),
                    html=f'<div style="font-size: 12pt; font-weight: bold; color: {color}; background: white; padding: 2px; border: 1px solid {color}; border-radius: 4px;">#{i+1}</div>'
                )
            ).add_to(m)

            if raw["blocked"] and raw["obstruction_location"]:
                 obs_lat, obs_lon = raw["obstruction_location"]
                 
                 # Prominent Obstruction Marker
                 folium.Marker(
                     [obs_lat, obs_lon],
                     popup=f"⛔ Obstruction at {raw['max_obstruction_height']:.1f}m",
                     icon=folium.Icon(color='red', icon='ban', prefix='fa')
                 ).add_to(m)

    # Draw Control (For Intentional Pin Dropping)
    from folium.plugins import Draw
    
    draw = Draw(
        draw_options={
            'polyline': False,
            'polygon': False,
            'circle': False,
            'marker': True,
            'circlemarker': False,
            'rectangle': False,
        },
        edit_options={'edit': False, 'remove': False}, # Disable edit interface
        position='topleft'
    )
    draw.add_to(m)

    # Render Map & Capture Interaction
    # We use 'last_active_drawing' to capture when a user drops a marker via the toolbar.
    # This solves the "Accidental Touch" issue because it requires selecting the tool first.
    map_out = st_folium(
        m, 
        width=None, 
        height=450, 
        key="main_map_interface",
        returned_objects=["last_active_drawing"]
    )

    # Update Persisted Center/Zoom
    # Default behavior of st_folium handles view state reasonably well between reruns 
    # if we don't force it. Since we removed 'last_clicked' logic, map reruns are minimized
    # to only when drawings occur.

    # Handle Interaction (Draw Event)
    if 'last_processed_drawing' not in st.session_state:
        st.session_state.last_processed_drawing = None

    if map_out and map_out.get("last_active_drawing"):
        drawing = map_out["last_active_drawing"]
        
        # Check against previous to prevent Loop / Stale Overwrite
        if drawing != st.session_state.last_processed_drawing:
            if drawing['geometry']['type'] == 'Point':
                # GeoJSON is [Lon, Lat]
                lon_c, lat_c = drawing['geometry']['coordinates']
                
                # Update Target
                st.session_state.picked_a = [lat_c, lon_c]
                st.toast(f"📍 Target set to {lat_c:.4f}, {lon_c:.4f}")
                
                # Mark as processed
                st.session_state.last_processed_drawing = drawing
                st.session_state.force_map_update = True
                st.rerun()
    
    # Reset Button (Clear All)
    if st.button("Reset"):
        st.session_state.coords_target = ""
        st.session_state.picked_a = None
        st.session_state.results = [] # Clear analysis data
        st.rerun()

    st.divider()

    # ----------------------
    # ANALYSIS CONFIGURATION
    # ----------------------
    st.subheader("📍 Target Location")
    
    col1, col2 = st.columns([2, 1])
    
    # Sync Logic for Single Point
    # We use 'coords_target' as the single source of truth.
    # 'picked_a' is used strictly as a temporary signal from the Map to update the Input.
    if 'coords_target' not in st.session_state: st.session_state.coords_target = ""

    with col1:
         # Check for map picks (Update State from Map -> Input)
        if st.session_state.get('picked_a'):
            st.session_state.coords_target = f"{st.session_state.picked_a[0]:.6f}, {st.session_state.picked_a[1]:.6f}"
            st.session_state.picked_a = None 
            
        t_input = st.text_input("Coordinates (Lat, Lon)", key="coords_target", help="Format: Decimal Degrees (WGS84 / EPSG:4326). Example: 25.2769, 55.2962")
        
    with col2:
        if st.button("⌖", key="zoom_target", help="Center map on Target"):
             cur_val = st.session_state.get("coords_target", "")
             lat_z, lon_z = parse_coords(cur_val)
             if lat_z and lon_z:
                st.session_state.map_center = [lat_z, lon_z]
                st.session_state.map_zoom = 15
                st.session_state.pick_state = 'A'
                st.session_state.force_map_update = True
                st.rerun()

    h_target = st.number_input("Target Height (m)", value=10.0, step=1.0, min_value=0.0, max_value=500.0, help="Height of the receiver/user.")
    
    st.divider()
    
    # Analysis Settings
    st.subheader("⚙️ Search Criteria")
    criteria_mode = st.radio("Search By:", ["Nearest Neighbors (Count)", "Fixed Radius (Distance)"], horizontal=True)
    
    max_results = 5
    search_radius_km = 99999.0 # Infinity effectively
    
    if criteria_mode == "Nearest Neighbors (Count)":
        max_results = st.slider("Number of Sites to Find", min_value=1, max_value=20, value=5)
    else:
        search_radius_km = st.slider("Search Radius (km)", min_value=1.0, max_value=100.0, value=10.0)
        max_results = 100 # High cap for radius mode to show all

    # Action
    if st.button("Find Sites & Analyze Terrain", type="primary"):
        target_lat, target_lon = parse_coords(t_input)
        
        if not target_lat or not target_lon:
            st.error("❌ Please set a valid Target Location.")
            st.stop()
            
        if st.session_state.site_data is None:
            st.error("❌ No Site Data (CSV/KML) imported! Please upload sites in the sidebar to analyze.")
            st.stop()
            
        st.session_state.results = [] # Clear previous
        
        with st.spinner("Finding candidates..."):
            sites = st.session_state.site_data.copy()
            
            # 1. Calculate Distances
            from geopy.distance import geodesic
            
            def get_dist(row):
                return geodesic((target_lat, target_lon), (row['Latitude'], row['Longitude'])).kilometers
                
            sites['Distance_km'] = sites.apply(get_dist, axis=1)
            
            # Filter Logic based on Criteria
            candidates = sites
            
            if criteria_mode == "Fixed Radius (Distance)":
                 # Radius Mode: Hard Filter on Distance first
                 candidates = sites[sites['Distance_km'] <= search_radius_km].copy()
                 if candidates.empty:
                    st.warning(f"No sites found within {search_radius_km}km radius.")
                    st.stop()
            else:
                # Count Mode: Sort by distance immediately to get nearest
                # We do this later in scoring, but good to know
                pass 
                
            # 2. Azimuth Check (if valid)
            import math
            
            def calculate_bearing(lat1, lon1, lat2, lon2):
                # Bearing from Site (lat1, lon1) to Target (lat2, lon2)
                d_lon = lon2 - lon1
                y = math.sin(math.radians(d_lon)) * math.cos(math.radians(lat2))
                x = math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) - \
                    math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(math.radians(d_lon))
                brng = math.degrees(math.atan2(y, x))
                return (brng + 360) % 360

            # 2. Score & Sort (Distance + Azimuth Penalty)
            scored_candidates = []
            
            for idx, row in candidates.iterrows():
                # Check Azimuth
                bearing_to_target = calculate_bearing(row['Latitude'], row['Longitude'], target_lat, target_lon)
                
                score = row['Distance_km'] # Base score
                
                if 'Azimuth' in row and pd.notnull(row['Azimuth']):
                    site_az = float(row['Azimuth'])
                    diff = abs(site_az - bearing_to_target)
                    if diff > 180: diff = 360 - diff
                    
                    if diff > (beam_width / 2):
                       score += 1000 # Penalty
                
                scored_candidates.append((score, row))
            
            # Sort
            scored_candidates.sort(key=lambda x: x[0])
            
            # Select Top N
            # In Radius mode, max_results is 100 (effectively "All in Radius")
            # In Count mode, max_results is user selected (e.g. 5)
            final_selection = [x[1] for x in scored_candidates[:max_results]]
            
            # Create DF
            df_valid = pd.DataFrame(final_selection)
            
            if df_valid.empty:
                st.warning("No valid candidates found.")
                st.stop()
            
            # 3. Run Analysis on Top N
            progress_bar = st.progress(0)
            for i, (idx, row) in enumerate(df_valid.iterrows()):
                
                site_h = float(row.get('Tower_Height', 30.0))
                
                # Using V3 Analysis with Advanced RF Parameters
                # Calculate bearing to target for antenna pattern
                import math
                d_lon = target_lon - row['Longitude']
                y = math.sin(math.radians(d_lon)) * math.cos(math.radians(target_lat))
                x = math.cos(math.radians(row['Latitude'])) * math.sin(math.radians(target_lat)) - \
                    math.sin(math.radians(row['Latitude'])) * math.cos(math.radians(target_lat)) * math.cos(math.radians(d_lon))
                bearing_to_target = (math.degrees(math.atan2(y, x)) + 360) % 360
                
                site_az = float(row.get('Azimuth', 0)) if 'Azimuth' in row and pd.notnull(row.get('Azimuth')) else 0
                
                res = analyze_terrain_profile_v3(
                    row['Latitude'], row['Longitude'],
                    target_lat, target_lon,
                    h_start_agl=site_h,
                    h_end_agl=h_target,
                    frequency_mhz=frequency_mhz,
                    k_factor=k_factor,
                    antenna_tilt_deg=antenna_tilt,
                    clutter_height_m=clutter_height,
                    # Link Budget
                    eirp_dbm=eirp_dbm,
                    fading_margin_db=fading_margin_db,
                    rx_sensitivity_dbm=rx_sensitivity_dbm,
                    # Advanced Propagation
                    prop_model=prop_model,
                    rain_rate_mmh=rain_rate_mmh,
                    indoor_loss_db=indoor_loss_db,
                    h_beamwidth=h_beamwidth,
                    azimuth_to_target=bearing_to_target,
                    site_azimuth=site_az
                )
                
                if res.get("status") == "Success":
                    # Coverage Status Emoji
                    cov_emojis = {5: "🟢", 4: "🟢", 3: "🟡", 2: "🟠", 1: "🔴", 0: "⛔"}
                    cov_emoji = cov_emojis.get(res["coverage_quality"], "❓")
                    
                    st.session_state.results.append({
                        "ID": f"{row['Site_ID']}_{idx}",
                        "Path Name": f"{row['Site_ID']} → Target",
                        "Site ID": row['Site_ID'],
                        "Distance (km)": round(res["dataframe"]["distance_km"].max(), 2),
                        "LoS": "Clear" if not res["blocked"] else "Blocked",
                        "RSRP (dBm)": round(res["estimated_rsrp_dbm"], 1),
                        "Path Loss (dB)": round(res["total_path_loss_db"], 1),
                        "Coverage": f"{cov_emoji} {res['coverage_verdict']}",
                        "Link Margin (dB)": round(res["link_margin_db"], 1),
                        "Raw": res
                    })
                
                progress_bar.progress((i + 1) / len(df_valid))
            
            # Trigger map to auto-fit to results
            st.session_state.force_map_update = True
            st.rerun()

    # ----------------------
    # VISUALIZATION & OUTPUT
    # ----------------------
    if st.session_state.results:
        st.divider()
        
        # Header with Clear Button
        col_res, col_clear = st.columns([6, 2])
        with col_res:
            st.header("Results")
        with col_clear:
            if st.button("🗑️ Clear Results", type="secondary"):
                st.session_state.results = []
                st.rerun()
        
        # Convert to Dataframe for display
        res_df = pd.DataFrame([ {k:v for k,v in r.items() if k != "Raw"} for r in st.session_state.results ])
        
        # Download Button
        csv = res_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Results CSV",
            csv,
            "terrain_analysis_results.csv",
            "text/csv",
            key='download-csv'
        )
        
        # --- DETAIL VIEW ---
        st.subheader("🔍 Analysis Details")
        st.caption("Select a row in the table below to view its Elevation Profile.")

        # Interactive Dataframe
        # We add a 'Select' column or just use the native selection if available.
        # Streamlit 1.35+ supports on_select. Let's assume standard interactive table.
        
        selection = st.dataframe(
            res_df,
            on_select="rerun", # Enable row selection
            selection_mode="single-row",
            use_container_width=True,
            hide_index=True
        )
        
        selected_index = 0 # Default to first
        if selection and selection.selection and selection.selection["rows"]:
             selected_index = selection.selection["rows"][0]
        
        if st.session_state.results:
            # Match the DF row index to the Results list index
            selected_result = st.session_state.results[selected_index]
            raw_data = selected_result["Raw"]
            
            # === RF LINK BUDGET DASHBOARD ===
            st.subheader("📶 RF Link Budget Analysis")
            
            # Main Metrics Row
            col1, col2, col3, col4, col5 = st.columns(5)
            
            # Coverage Quality Color
            cov_colors = {5: "green", 4: "green", 3: "orange", 2: "red", 1: "red", 0: "red"}
            cov_color = cov_colors.get(raw_data["coverage_quality"], "gray")
            
            with col1:
                st.metric("📡 RSRP", f"{raw_data['estimated_rsrp_dbm']:.1f} dBm")
            with col2:
                st.metric("� Coverage", raw_data["coverage_verdict"])
            with col3:
                st.metric("📉 Path Loss", f"{raw_data['total_path_loss_db']:.1f} dB")
            with col4:
                st.metric("📏 Link Margin", f"{raw_data['link_margin_db']:.1f} dB")
            with col5:
                st.metric("🎯 Max Range", f"{raw_data['max_range_km']:.2f} km")
            
            # Path Loss Breakdown (Expanded)
            with st.expander("📋 Path Loss Breakdown (All Components)", expanded=False):
                # Row 1: Model + Base Losses
                col_pl1, col_pl2, col_pl3 = st.columns(3)
                with col_pl1:
                    st.metric("Model Path Loss", f"{raw_data['fspl_db']:.1f} dB", help=raw_data.get('prop_model', 'FSPL'))
                with col_pl2:
                    st.metric("Diffraction", f"{raw_data['diffraction_loss_db']:.1f} dB")
                with col_pl3:
                    st.metric("Clutter", f"{raw_data['clutter_loss_db']:.1f} dB")
                
                # Row 2: Advanced Losses
                col_pl4, col_pl5, col_pl6 = st.columns(3)
                with col_pl4:
                    st.metric("🌧️ Rain Fade", f"{raw_data.get('rain_loss_db', 0):.1f} dB")
                with col_pl5:
                    st.metric("📡 Antenna Pattern", f"{raw_data.get('antenna_pattern_loss_db', 0):.1f} dB")
                with col_pl6:
                    st.metric("🏠 Indoor Loss", f"{raw_data.get('indoor_loss_db', 0):.1f} dB")
                
                st.divider()
                st.caption(f"**Model:** {raw_data.get('prop_model', 'FSPL')} | **EIRP:** {raw_data['eirp_dbm']:.1f} dBm | **Frequency:** {raw_data['frequency_mhz']} MHz | **K-Factor:** {raw_data['k_factor']}")
            
            # Coverage Verdict Alert
            if raw_data["coverage_quality"] >= 4:
                st.success(f"✅ **{raw_data['coverage_verdict']} Coverage** - Target location will receive strong 5G signal from this site.")
            elif raw_data["coverage_quality"] == 3:
                st.warning(f"⚠️ **{raw_data['coverage_verdict']} Coverage** - Signal may be weak. Consider closer site or higher antenna.")
            elif raw_data["coverage_quality"] == 2:
                st.warning(f"🟠 **{raw_data['coverage_verdict']} Coverage** - Marginal signal. Indoor coverage unlikely.")
            else:
                st.error(f"❌ **{raw_data['coverage_verdict']}** - This site cannot provide coverage to the target location.")
            
            # Obstruction Details (if any)
            if raw_data["blocked"] and raw_data["obstruction_location"]:
                st.error(f"⛔ **{raw_data['obstruction_type']}** blocks LoS by {raw_data['max_obstruction_height']:.1f}m. Increase antenna height by at least **+{raw_data['required_height_increase']:.1f}m** to clear.")

            # Line Color
            if raw_data["blocked"]:
                color = "red"
            elif raw_data["fresnel_violated"]:
                color = "orange"
            else:
                color = "green"

            # Elevation Profile
            st.subheader(f"⛰️ Elevation Profile: {selected_result['Path Name']}")
            st.caption(f"Frequency: {raw_data['frequency_mhz']} MHz | K-Factor: {raw_data['k_factor']}")
        
        df_chart = raw_data['dataframe']
        
        fig = go.Figure()
        
        # Terrain fill (with Clutter)
        fig.add_trace(go.Scatter(
            x=df_chart['distance_km'], 
            y=df_chart['elevation_with_clutter'],
            fill='tozeroy',
            mode='lines',
            name='Terrain + Clutter',
            line=dict(color='rgba(128, 128, 128, 0.8)')
        ))
        
        # Raw Terrain (lighter)
        fig.add_trace(go.Scatter(
            x=df_chart['distance_km'], 
            y=df_chart['elevation'],
            mode='lines',
            name='Raw Terrain',
            line=dict(color='rgba(100, 100, 100, 0.4)', width=1)
        ))
        
        # Fresnel Zone Fill (between upper and lower)
        fig.add_trace(go.Scatter(
            x=df_chart['distance_km'],
            y=df_chart['fresnel_upper'],
            mode='lines',
            name='Fresnel Upper',
            line=dict(color='rgba(0, 150, 255, 0.3)', width=1),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=df_chart['distance_km'],
            y=df_chart['fresnel_lower'],
            mode='lines',
            name='1st Fresnel Zone',
            fill='tonexty',
            fillcolor='rgba(0, 150, 255, 0.2)',
            line=dict(color='rgba(0, 150, 255, 0.3)', width=1)
        ))
        
        # LoS Line
        fig.add_trace(go.Scatter(
            x=df_chart['distance_km'],
            y=df_chart['los_elevation'],
            mode='lines',
            name='Line of Sight',
            line=dict(color=color, dash='dash', width=2)
        ))
        
        # Highlight obstruction areas
        if raw_data["blocked"]:
             obstruction_points = df_chart[df_chart['is_obstructing']]
             fig.add_trace(go.Scatter(
                x=obstruction_points['distance_km'],
                y=obstruction_points['elevation_with_clutter'],
                mode='markers',
                name='LoS Obstructions',
                marker=dict(color='red', size=6, symbol='x')
            ))
        
        # Highlight Fresnel violations
        if raw_data["fresnel_violated"] and not raw_data["blocked"]:
             fresnel_blocked_points = df_chart[df_chart['is_fresnel_blocked']]
             fig.add_trace(go.Scatter(
                x=fresnel_blocked_points['distance_km'],
                y=fresnel_blocked_points['elevation_with_clutter'],
                mode='markers',
                name='Fresnel Zone Intrusion',
                marker=dict(color='orange', size=5, symbol='triangle-up')
            ))
        
        fig.update_layout(
            title=f"RF Path Analysis @ {raw_data['frequency_mhz']} MHz (K={raw_data['k_factor']})",
            xaxis_title="Distance (km)",
            yaxis_title="Elevation (m)",
            hovermode="x unified",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
