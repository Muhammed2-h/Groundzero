import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
from terrain_utils import analyze_terrain_profile
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
    
    if 'site_data' not in st.session_state:
        st.session_state.site_data = None
        
    if site_file:
        df_sites, error = parse_site_data(site_file)
        if df_sites is not None:
            st.session_state.site_data = df_sites
            st.session_state.sites_just_loaded = True # Flag to center map
            st.sidebar.success(f"Loaded {len(df_sites)} sites!")
        else:
            st.sidebar.error(error)
    else:
        # File removed -> Clear data
        st.session_state.site_data = None
    
    # ----------------------
    if 'locked_point' not in st.session_state:
        st.session_state.locked_point = None
    
    # Output Data Container
    if 'results' not in st.session_state:
        st.session_state.results = []
    
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
    c_head, c_layer = st.columns([3, 1])
    with c_head:
        st.subheader("🌍 Map Interface")
        st.caption("Tap/Click the map to set analysis target location.")
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
        for res in st.session_state.results:
            raw = res["Raw"]
            color = "red" if raw["blocked"] else "green"
            
            # Path
            path_coords = [(row['lat'], row['lon']) for _, row in raw['dataframe'].iterrows()]
            folium.PolyLine(path_coords, color=color, weight=3, opacity=0.6).add_to(m)
            
            if raw["blocked"] and raw["obstruction_location"]:
                 obs_lat, obs_lon = raw["obstruction_location"]
                 folium.CircleMarker([obs_lat, obs_lon], radius=2, color='red').add_to(m)

    # Render Map & Capture Click
    # STUTTER FIX: We use 'last_clicked' but we need to ensure we don't accidentally
    # trigger meaningful updates on PANS. st_folium handles this by only updating 'last_clicked' on clicks.
    map_out = st_folium(
        m, 
        width=None, 
        height=450, # Slightly reduced height for mobile friendliness
        key="main_map_interface",
        returned_objects=["last_clicked"]
    )

    # Update Persisted Center/Zoom
    # Logic: Sync Frontend State -> Backend State
    if not st.session_state.get('force_map_update', False):
        if map_out:
            # 1. Update Center
            if "center" in map_out and map_out["center"]:
                new_lat = map_out["center"]["lat"]
                new_lon = map_out["center"]["lng"]
                st.session_state.map_center = [new_lat, new_lon]

            # 2. Update Zoom
            if "zoom" in map_out:
                new_zoom = map_out["zoom"]
                st.session_state.map_zoom = new_zoom
            
            # Note: We do NOT trigger st.rerun() here.
            # Doing so causes a "refresh loop" on every zoom interaction.
            # st_folium handles the view state on the frontend.
            # We just capture the new state so FUTURE reruns (e.g. typing coordinates) start from here.
    else:
        # Reset the flag after one render cycle so manual panning works again next time
        st.session_state.force_map_update = False

    # Handle Interaction (Always Active)
    if map_out and map_out.get("last_clicked"):
        lat_c = map_out["last_clicked"]["lat"]
        lng_c = map_out["last_clicked"]["lng"]
        
        # Check if this click is "new" or if we already processed it?
        # st_folium returns the SAME 'last_clicked' object until a new click occurs.
        # We need to compare with the "currently selected" target to avoid re-triggering constantly?
        # Actually, st.rerun() resets the script. 
        # But if the user clicks, we set the coord, rerun. 
        # Next run: last_clicked is STILL the same. Rerun again? No, we check if value changed.
        
        current_target_str = st.session_state.coords_target
        current_lat, current_lon = parse_coords(current_target_str)
        
        # Simulating "New Click Only" Check
        is_new_click = True
        if current_lat and current_lon:
            if abs(current_lat - lat_c) < 0.000001 and abs(current_lon - lng_c) < 0.000001:
                is_new_click = False # Same click as before
        
        if is_new_click:
            st.session_state.picked_a = [lat_c, lng_c] # reuse this temp state
            st.toast(f"📍 Target set to {lat_c:.4f}, {lng_c:.4f}")
            st.session_state.force_map_update = True
            st.rerun()
    
    # Reset Button for Picks
    if st.button("Reset Selection"):
        st.session_state.coords_target = ""
        st.session_state.picked_a = None
        st.rerun()

    st.divider()

    # ----------------------
    # ANALYSIS CONFIGURATION
    # ----------------------
    st.subheader("📍 Target Location")
    
    col1, col2 = st.columns([2, 1])
    
    # Sync Logic for Single Point
    # Reusing the global parse_coords and sync_input_to_marker
    if 'coords_target' not in st.session_state: st.session_state.coords_target = ""
    sync_input_to_marker("coords_target", "picked_a") # Reusing 'picked_a' state for simplified map logic

    with col1:
         # Check for map picks (Update State directly)
        if st.session_state.get('picked_a'):
            st.session_state.coords_target = f"{st.session_state.picked_a[0]:.6f}, {st.session_state.picked_a[1]:.6f}"
            st.session_state.picked_a = None 
            
        t_input = st.text_input("Coordinates (Lat, Lon)", key="coords_target", help="Enter the location you want to analyze coverage for.")
        
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
    c_res, c_rad = st.columns(2)
    with c_res:
        max_results = st.slider("Max Results to View", min_value=1, max_value=50, value=5)
    with c_rad:
        search_radius_km = st.slider("Search Radius (km)", min_value=1.0, max_value=100.0, value=10.0)

    # Action
    if st.button("Find Best Sites & Analyze Terrain", type="primary"):
        target_lat, target_lon = parse_coords(t_input)
        
        if not target_lat or not target_lon:
            st.error("❌ Please set a valid Target Location.")
            st.stop()
            
        if st.session_state.site_data is None:
            st.error("❌ No Site Data (CSV/KML) imported! Please upload sites in the sidebar to analyze.")
            st.stop()
            
        st.session_state.results = [] # Clear previous
        
        with st.spinner("Finding best candidates..."):
            sites = st.session_state.site_data.copy()
            
            # 1. Calculate Distances
            from geopy.distance import geodesic
            
            def get_dist(row):
                return geodesic((target_lat, target_lon), (row['Latitude'], row['Longitude'])).kilometers
                
            sites['Distance_km'] = sites.apply(get_dist, axis=1)
            
            # Filter by Radius
            candidates = sites[sites['Distance_km'] <= search_radius_km].copy()
            
            if candidates.empty:
                st.warning(f"No sites found within {search_radius_km}km radius.")
                st.stop()
                
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

            valid_candidates = []
            
            # Iterate and Score
            # Iterate and Score
            scored_candidates = []
            
            for idx, row in candidates.iterrows():
                # Check Azimuth
                bearing_to_target = calculate_bearing(row['Latitude'], row['Longitude'], target_lat, target_lon)
                
                score = row['Distance_km'] # Base score is distance (lower is better)
                
                if 'Azimuth' in row and pd.notnull(row['Azimuth']):
                    site_az = float(row['Azimuth'])
                    # Difference
                    diff = abs(site_az - bearing_to_target)
                    if diff > 180: diff = 360 - diff
                    
                    # If it's outside the main beam, penalize the score significantly
                    # effectively sorting "Facing" sites to the top, but keeping "Side/Back" sites as backups
                    if diff > (beam_width / 2):
                       score += 1000 # Large penalty to push to bottom of list
                
                # Store tuple (Score, Row)
                scored_candidates.append((score, row))
            
            # Sort by Score (Distance + Penalty)
            scored_candidates.sort(key=lambda x: x[0])
            
            valid_candidates = [x[1] for x in scored_candidates]
            
            if not valid_candidates:
                st.warning("No sites found in radius.")
                st.stop()
                
            # Create DF from valid
            df_valid = pd.DataFrame(valid_candidates)
            df_valid = df_valid.sort_values('Distance_km').head(max_results)
            
            # 3. Run Analysis on Top N
            progress_bar = st.progress(0)
            for i, (idx, row) in enumerate(df_valid.iterrows()):
                
                site_h = float(row.get('Tower_Height', 30.0))
                
                res = analyze_terrain_profile(
                    row['Latitude'], row['Longitude'],
                    target_lat, target_lon,
                    h_start_agl=site_h,
                    h_end_agl=h_target
                )
                
                if res.get("status") == "Success":
                     st.session_state.results.append({
                        "ID": f"{row['Site_ID']}_{idx}",
                        "Path Name": f"{row['Site_ID']} -> Target",
                        "Site ID": row['Site_ID'],
                        "Distance (km)": res["dataframe"]["distance_km"].max(),
                        "Status": "BLOCKED" if res["blocked"] else "CLEAR",
                        "Max Obstruction (m)": res["max_obstruction_height"],
                        "Raw": res
                    })
                
                progress_bar.progress((i + 1) / len(df_valid))

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
        st.dataframe(res_df)
        
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
        # If multiple results, let user select one to visualize
        selected_id = st.selectbox("Select Path to Visualize", [r["ID"] for r in st.session_state.results])
        
        selected_result = next(r for r in st.session_state.results if r["ID"] == selected_id)
        raw_data = selected_result["Raw"]
        
        if raw_data["blocked"] and raw_data["obstruction_location"]:
            # Just show obstruction details in text or smaller UI, since map is above
            st.warning(f"⚠️ Max Obstruction: {raw_data['max_obstruction_height']:.2f}m at {raw_data['obstruction_location']}")

        # Line Color (Re-defined here as previous definition scope might be gone if map logic changes)
        color = "red" if raw_data["blocked"] else "green"

        # 2. Elevation Profile
        st.subheader("⛰️ Elevation Profile")
        
        df_chart = raw_data['dataframe']
        
        fig = go.Figure()
        
        # Terrain fill
        fig.add_trace(go.Scatter(
            x=df_chart['distance_km'], 
            y=df_chart['elevation'],
            fill='tozeroy',
            mode='lines',
            name='Terrain',
            line=dict(color='gray')
        ))
        
        # LoS Line
        fig.add_trace(go.Scatter(
            x=df_chart['distance_km'],
            y=df_chart['los_elevation'],
            mode='lines',
            name='Line of Sight',
            line=dict(color=color, dash='dash')
        ))
        
        # Highlight obstruction areas
        if raw_data["blocked"]:
             obstruction_points = df_chart[df_chart['is_obstructing']]
             fig.add_trace(go.Scatter(
                x=obstruction_points['distance_km'],
                y=obstruction_points['elevation'],
                mode='markers',
                name='Obstructions',
                marker=dict(color='red', size=5)
            ))
        
        fig.update_layout(
            title="Terrain vs Line of Sight",
            xaxis_title="Distance (km)",
            yaxis_title="Elevation (m)",
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
