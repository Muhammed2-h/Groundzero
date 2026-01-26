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
    Suports **Manual Entry**, **Locked One-to-Many**, and **Batch CSV** modes.
    """)

    # --- Sidebar Controls ---
    st.sidebar.header("Configuration")
    
    # Mode Selection
    mode = st.sidebar.radio("Input Mode", ["Manual", "Batch CSV"])
    
    # --- SITE DATA IMPORT ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("🌍 Import Site Data")
    site_file = st.sidebar.file_uploader("Upload Sites (CSV/KML/KMZ)", type=["csv", "kml", "kmz"])
    
    if 'site_data' not in st.session_state:
        st.session_state.site_data = None
        
    if site_file:
        df_sites, error = parse_site_data(site_file)
        if df_sites is not None:
            st.session_state.site_data = df_sites
            st.sidebar.success(f"Loaded {len(df_sites)} sites!")
        else:
            st.sidebar.error(error)
    
    # ----------------------
    if 'locked_point' not in st.session_state:
        st.session_state.locked_point = None
    
    # Output Data Container
    if 'results' not in st.session_state:
        st.session_state.results = []
    
    # ----------------------
    # MANUAL MODE LOGIC
    # ----------------------
    if mode == "Manual":
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

        # --- INPUT LOGIC (TEXT vs SELECTBOX) ---
        # Toggle if sites are available
        use_site_list = False
        if st.session_state.site_data is not None:
            use_site_list = st.checkbox("Select from Imported Sites", value=True)
            
        a_lat, a_lon, b_lat, b_lon = None, None, None, None
        
        if use_site_list:
            sites = st.session_state.site_data
            site_options = sites['Site_ID'].tolist()
            
            with col1:
                st.markdown("### Point A (Origin)")
                selected_a = st.selectbox("Select Site A", site_options, key="sel_a")
                row_a = sites[sites['Site_ID'] == selected_a].iloc[0]
                a_lat, a_lon = row_a['Latitude'], row_a['Longitude']
                default_h_a = float(row_a['Tower_Height'])
                h_a = st.number_input("Tower Height A (m)", value=default_h_a, step=1.0, min_value=0.0, max_value=500.0, key="h_a")
                st.caption(f"📍 {a_lat:.5f}, {a_lon:.5f}")
                
            with col2:
                st.markdown("### Point B (Target)")
                selected_b = st.selectbox("Select Site B", site_options, index=min(1, len(site_options)-1), key="sel_b")
                row_b = sites[sites['Site_ID'] == selected_b].iloc[0]
                b_lat, b_lon = row_b['Latitude'], row_b['Longitude']
                default_h_b = float(row_b['Tower_Height'])
                h_b = st.number_input("Tower Height B (m)", value=default_h_b, step=1.0, min_value=0.0, max_value=500.0, key="h_b")
                st.caption(f"📍 {b_lat:.5f}, {b_lon:.5f}")
                
        else:
            # Point A Inputs (Manual)
            with col1:
                st.markdown("### Point A (Origin)")
                a_input = st.text_input("Coordinates A (Lat, Lon)", value="0.0, 0.0", help="Format: Latitude, Longitude")
                h_a = st.number_input("Tower Height A (m)", value=10.0, step=1.0, min_value=0.0, max_value=500.0, key="h_a")
                
            # Point B Inputs (Manual)
            with col2:
                st.markdown("### Point B (Target)")
                b_input = st.text_input("Coordinates B (Lat, Lon)", value="0.0, 0.0", help="Format: Latitude, Longitude")
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
    # BATCH CSV MODE
    # ----------------------
    elif mode == "Batch CSV":
        st.subheader("📂 Batch Analysis")
        
        # Template Download
        template_csv = "A_name,A_lat,A_long,B_name,B_lat,B_long\nKochi,9.9312,76.2673,Munnar,10.0889,77.0595"
        st.download_button(
            label="Download CSV Template",
            data=template_csv,
            file_name="terrain_analysis_template.csv",
            mime="text/csv",
            help="Download a sample CSV file to fill in your coordinates."
        )
        
        st.markdown("Upload a CSV with columns: `A_lat`, `A_long`, `B_lat`, `B_long` (Names optional)")
        
        batch_file = st.file_uploader("Upload Batch CSV", type=["csv"], key="batch_upload")
        
        if batch_file and st.button("Run Batch Analysis"):
            st.session_state.results = [] # Clear previous
            df = pd.read_csv(batch_file)
            # Normalize Headers
            df.columns = df.columns.str.lower().str.strip()
            
            # Map columns
            try:
                # Required
                a_lat_col = next(c for c in df.columns if 'a_' in c and 'lat' in c)
                a_lon_col = next(c for c in df.columns if 'a_' in c and ('lon' in c or 'lng' in c))
                b_lat_col = next(c for c in df.columns if 'b_' in c and 'lat' in c)
                b_lon_col = next(c for c in df.columns if 'b_' in c and ('lon' in c or 'lng' in c))
                
                # Optional Names
                a_name_col = next((c for c in df.columns if 'a_' in c and 'name' in c), None)
                b_name_col = next((c for c in df.columns if 'b_' in c and 'name' in c), None)
                
                with st.spinner("Processing batch..."):
                    bar = st.progress(0)
                    for i, row in df.iterrows():
                        # Extract Names
                        name_a = str(row[a_name_col]) if a_name_col else f"Point A ({i+1})"
                        name_b = str(row[b_name_col]) if b_name_col else f"Point B ({i+1})"
                        
                        # Run Analysis
                        res = analyze_terrain_profile(
                            row[a_lat_col], row[a_lon_col], 
                            row[b_lat_col], row[b_lon_col],
                            name_a=name_a, name_b=name_b
                        )
                        
                        if res.get("status") == "Success":
                            st.session_state.results.append({
                                "ID": f"Batch_{i}",
                                "Path Name": f"{name_a} -> {name_b}", # New friendly column
                                "Point A": f"{row[a_lat_col]},{row[a_lon_col]}",
                                "Point B": f"{row[b_lat_col]},{row[b_lon_col]}",
                                "Distance (km)": res["dataframe"]["distance_km"].max(),
                                "Status": "BLOCKED" if res["blocked"] else "CLEAR",
                                "Max Obstruction (m)": res["max_obstruction_height"],
                                "Raw": res
                            })
                        bar.progress((i + 1) / len(df))
                        
            except StopIteration:
                st.error("Could not find required columns (A_lat, A_long, B_lat, B_long) in CSV.")

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
        
        # 1. Folium Map
        st.subheader("🗺️ Map Visualization")
        
        # Center map
        start_pt = raw_data["start_point"]
        end_pt = raw_data["end_point"]
        mid_lat = (start_pt[0] + end_pt[0]) / 2
        mid_lon = (start_pt[1] + end_pt[1]) / 2
        
        m = folium.Map(location=[mid_lat, mid_lon], zoom_start=12)
        
        # Line Color
        color = "red" if raw_data["blocked"] else "green"
        
        # Markers
        folium.Marker([start_pt[0], start_pt[1]], popup="Point A", icon=folium.Icon(color="blue", icon="play")).add_to(m)
        folium.Marker([end_pt[0], end_pt[1]], popup="Point B", icon=folium.Icon(color="blue", icon="stop")).add_to(m)
        
        # Polyline
        path_coords = [(row['lat'], row['lon']) for _, row in raw_data['dataframe'].iterrows()]
        folium.PolyLine(path_coords, color=color, weight=5, opacity=0.8).add_to(m)
        
        # Obstruction Marker
        if raw_data["blocked"] and raw_data["obstruction_location"]:
            obs_lat, obs_lon = raw_data["obstruction_location"]
            folium.Marker(
                [obs_lat, obs_lon], 
                popup=f"Max Obstruction: {raw_data['max_obstruction_height']:.2f}m",
                icon=folium.Icon(color="red", icon="exclamation-triangle", prefix="fa")
            ).add_to(m)
            
        st_folium(m, width=None, height=500)
        
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
