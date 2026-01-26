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
    # ----------------------
    # GLOBAL MAP INTERFACE
    # ----------------------
    st.subheader("🌍 Map Interface")
    c_head, c_opt = st.columns([4, 2])
    with c_head:
        st.caption("Click on the map to set points for Manual Analysis.")
    with c_opt:
         map_style = st.selectbox("Map Layer", ["Street", "Satellite", "Terrain"], label_visibility="collapsed")
         pick_enabled = st.checkbox("📍 Enable Picking", value=False, help="Turn on to select points from the map.")
    
    # Initialize Pick State
    if 'pick_state' not in st.session_state:
        st.session_state.pick_state = 'A'
    if 'picked_a' not in st.session_state: st.session_state.picked_a = None
    if 'picked_b' not in st.session_state: st.session_state.picked_b = None
    
    # State: Track Map Center/Zoom to prevent reset on interaction
    if 'map_center' not in st.session_state:
        st.session_state.map_center = [20.5937, 78.9629]
        st.session_state.map_zoom = 5

    # Base Map Center Logic
    # FIX: We only set `location` to "Default/Sites" ONCE (when data changes). 
    # Otherwise, we use the `map_center` from session state, which we update from `map_out`.

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

    # Draw Picked Points
    if st.session_state.picked_a:
        folium.Marker(st.session_state.picked_a, popup="Point A", icon=folium.Icon(color='green', icon='play')).add_to(m)
    if st.session_state.picked_b:
        folium.Marker(st.session_state.picked_b, popup="Point B", icon=folium.Icon(color='red', icon='stop')).add_to(m)
        
    # Draw Analysis Results (if any)
    if st.session_state.results:
        # Determine which one to show. Show LAST result by default.
        latest_res = st.session_state.results[-1]["Raw"]
        
        # Line Color
        color = "red" if latest_res["blocked"] else "green"
        
        # Path
        path_coords = [(row['lat'], row['lon']) for _, row in latest_res['dataframe'].iterrows()]
        folium.PolyLine(path_coords, color=color, weight=5, opacity=0.8).add_to(m)
        
        # Obstruction
        if latest_res["blocked"] and latest_res["obstruction_location"]:
            obs_lat, obs_lon = latest_res["obstruction_location"]
            folium.Marker(
                [obs_lat, obs_lon], 
                popup=f"Max Obstruction: {latest_res['max_obstruction_height']:.2f}m",
                icon=folium.Icon(color="red", icon="exclamation-triangle", prefix="fa")
            ).add_to(m)
            
        # Fit bounds if just calculated (optional, maybe distracting if picking multiple?)
        # Let's keep manual pan/zoom for now.

    # Render Map & Capture Click
    map_out = st_folium(m, width=None, height=500, key="main_map_interface")

    # Update Persisted Center/Zoom
    if map_out:
        if "center" in map_out and map_out["center"]:
           st.session_state.map_center = [map_out["center"]["lat"], map_out["center"]["lng"]]
        if "zoom" in map_out:
           st.session_state.map_zoom = map_out["zoom"]

    # Handle Interaction (Picking)
    if pick_enabled and map_out and map_out.get("last_clicked"):
        lat_c = map_out["last_clicked"]["lat"]
        lng_c = map_out["last_clicked"]["lng"]
        
        if st.session_state.pick_state == 'A':
            st.session_state.picked_a = [lat_c, lng_c]
            st.session_state.pick_state = 'B'
            st.toast(f"📍 Point A set")
            st.rerun()
        elif st.session_state.pick_state == 'B':
            st.session_state.picked_b = [lat_c, lng_c]
            st.session_state.pick_state = 'A'
            st.toast(f"📍 Point B set")
            st.rerun()
    
    # Reset Button for Picks
    if st.button("Reset Selection Points"):
        st.session_state.picked_a = None
        st.session_state.picked_b = None
        st.session_state.pick_state = 'A'
        st.rerun()

    st.divider()

    # ----------------------
    # MANUAL MODE LOGIC
    # ----------------------
    # ----------------------
    # MANUAL MODE LOGIC (Now Default)
    # ----------------------
    st.subheader("📍 Coordinate Input")
    
    # Locking Controls
    use_locking = st.checkbox("Enable Point Locking (One-to-Many)", help="Fix one point and analyze multiple targets.")
    
    col1, col2 = st.columns(2)
    
    # Helper for parsing
    def parse_coords(coord_str):
        try:
            if not coord_str: return None, None
            parts = coord_str.split(',')
            if len(parts) != 2: return None, None
            return float(parts[0].strip()), float(parts[1].strip())
        except ValueError:
            return None, None

    # --- INPUT LOGIC (MANUAL + MAP PICK) ---
    a_lat, a_lon, b_lat, b_lon = None, None, None, None

    # Point A Inputs (Manual)
    with col1:
        c_a_h, c_a_z = st.columns([3, 1])
        with c_a_h: st.markdown("### Point A (Origin)")
        with c_a_z:
            if st.button("⌖ Zoom", key="zoom_a", help="Center map on Point A"):
                # Read current input from state
                cur_val = st.session_state.get("coords_a", "")
                lat_z, lon_z = parse_coords(cur_val)
                if lat_z and lon_z:
                    st.session_state.map_center = [lat_z, lon_z]
                    st.session_state.map_zoom = 12
                    st.session_state.pick_state = 'A' # Optional: set expectation
                    st.rerun()
        
        # Check for map picks (Update State directly)
        if st.session_state.get('picked_a'):
            st.session_state.coords_a = f"{st.session_state.picked_a[0]:.6f}, {st.session_state.picked_a[1]:.6f}"
            # Clear pick to prevent overwrite on next type
            st.session_state.picked_a = None 
            
        a_input = st.text_input("Coordinates A (Lat, Lon)", key="coords_a", help="Format: Latitude, Longitude")
        h_a = st.number_input("Tower Height A (m)", value=10.0, step=1.0, min_value=0.0, max_value=500.0, key="h_a")
        
    # Point B Inputs (Manual)
    with col2:
        c_b_h, c_b_z = st.columns([3, 1])
        with c_b_h: st.markdown("### Point B (Target)")
        with c_b_z:
            if st.button("⌖ Zoom", key="zoom_b", help="Center map on Point B"):
                 cur_val = st.session_state.get("coords_b", "")
                 lat_z, lon_z = parse_coords(cur_val)
                 if lat_z and lon_z:
                    st.session_state.map_center = [lat_z, lon_z]
                    st.session_state.map_zoom = 12
                    st.session_state.pick_state = 'B'
                    st.rerun()

        # Check for map picks
        if st.session_state.get('picked_b'):
            st.session_state.coords_b = f"{st.session_state.picked_b[0]:.6f}, {st.session_state.picked_b[1]:.6f}"
            st.session_state.picked_b = None

        b_input = st.text_input("Coordinates B (Lat, Lon)", key="coords_b", help="Format: Latitude, Longitude")
        h_b = st.number_input("Tower Height B (m)", value=10.0, step=1.0, min_value=0.0, max_value=500.0, key="h_b")

    # Parse Inputs
    a_lat, a_lon = parse_coords(a_input)
    b_lat, b_lon = parse_coords(b_input)

    # Logic for "Locking" Feature in Manual Mode
    target_df = None
    
    if use_locking:
        lock_choice = st.radio("Lock which point?", ["Point A", "Point B"], horizontal=True)
        
        st.info(f"🔒 {lock_choice} is locked. Upload a CSV for the other points.")
        
        uploaded_file = st.file_uploader("Upload Targets CSV", type=["csv"])
        if uploaded_file:
            target_df = pd.read_csv(uploaded_file)
            st.write("Preview of targets:", target_df.head())
    
    # Action Button
    if st.button("Run Analysis", type="primary"):
        # Validation
        if a_lat is None or a_lon is None:
            st.error("❌ Invalid coordinates for Point A. Please use format 'lat, lon' (e.g., 9.123, 76.456)")
            st.stop()
        if b_lat is None or b_lon is None:
            st.error("❌ Invalid coordinates for Point B. Please use format 'lat, lon' (e.g., 9.123, 76.456)")
            st.stop()

        st.session_state.results = [] # Clear previous
        with st.spinner("Analyzing terrain..."):
            
            if not use_locking:
                res = analyze_terrain_profile(a_lat, a_lon, b_lat, b_lon, h_start_agl=h_a, h_end_agl=h_b)
                if res.get("status") == "Success":
                    st.session_state.results.append({
                        "ID": "Manual_1",
                        "Path Name": "Manual Path",
                        "Point A": f"{a_lat},{a_lon} ({h_a}m)",
                        "Point B": f"{b_lat},{b_lon} ({h_b}m)",
                        "Distance (km)": res["dataframe"]["distance_km"].max(),
                        "Status": "BLOCKED" if res["blocked"] else "CLEAR",
                        "Max Obstruction (m)": res["max_obstruction_height"],
                        "Raw": res
                    })
                else:
                    st.error(res.get("message"))

            # Case 2: One-to-Many (Locked)
            else:
                if target_df is None:
                    st.error("Please upload a CSV file definition for the target points.")
                else:
                    # We need to identify lat/lon columns loosely
                    cols = [c.lower() for c in target_df.columns]
                    
                    # Simple heuristics to find columns
                    lat_col = next((c for c in target_df.columns if "lat" in c.lower()), None)
                    lon_col = next((c for c in target_df.columns if "lon" in c.lower() or "lng" in c.lower()), None)
                    
                    if not lat_col or not lon_col:
                        st.error(f"Could not automatically detect 'lat' and 'lon' columns in CSV. Found: {target_df.columns.tolist()}")
                    else:
                        # Iterate
                        progress_bar = st.progress(0)
                        for idx, row in target_df.iterrows():
                            t_lat = row[lat_col]
                            t_lon = row[lon_col]
                            
                            # Use locked height vs default 10m for targets (unless CSV has height logic, but for MVP we use default 10m for targets)
                            # Actually, it would be smart to look for "height" column in CSV? 
                            # Let's assume targets are 10m for now unless we wanna over-engineer.
                            h_target = 10.0
                            
                            if lock_choice == "Point A":
                                # Fixed A (height h_a), Varying B (height 10)
                                res = analyze_terrain_profile(a_lat, a_lon, t_lat, t_lon, h_start_agl=h_a, h_end_agl=h_target)
                            else:
                                # Varying A, Fixed B
                                res = analyze_terrain_profile(t_lat, t_lon, b_lat, b_lon, h_start_agl=h_target, h_end_agl=h_b)
                            
                            if res.get("status") == "Success":
                                st.session_state.results.append({
                                    "ID": f"Target_{idx}",
                                    "Point A": f"{a_lat},{a_lon}" if lock_choice == "Point A" else f"{t_lat},{t_lon}",
                                    "Point B": f"{t_lat},{t_lon}" if lock_choice == "Point A" else f"{b_lat},{b_lon}",
                                    "Distance (km)": res["dataframe"]["distance_km"].max(),
                                    "Status": "BLOCKED" if res["blocked"] else "CLEAR",
                                    "Max Obstruction (m)": res["max_obstruction_height"],
                                    "Raw": res
                                })
                            
                            progress_bar.progress((idx + 1) / len(target_df))

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
