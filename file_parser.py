import pandas as pd
import zipfile
from fastkml import kml
import io

def parse_site_data(uploaded_file):
    """
    Parses CSV, KML, or KMZ files into a standardized DataFrame.
    Expected Columns (CSV): Site_ID, Latitude, Longitude, [Tower_Height]
    Returns: DataFrame with columns [Site_ID, Latitude, Longitude, Tower_Height]
    """
    filename = uploaded_file.name.lower()
    
    df = pd.DataFrame()
    
    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            # Normalize columns
            df.columns = df.columns.str.lower().str.strip()
            
            # Smart Column Mapping
            col_map = {
                'id': ['site_id', 'siteid', 'id', 'name', 'site identifier'],
                'lat': ['latitude', 'lat', 'y'],
                'lon': ['longitude', 'long', 'lon', 'lng', 'x'],
                'height': ['tower_height', 'height', 'tower height', 'antenna height', 'h']
            }
            
            final_cols = {}
            for target, options in col_map.items():
                found = next((c for c in df.columns if c in options), None)
                if not found and target in ['lat', 'lon']:
                    # Try partial match for critical columns
                    found = next((c for c in df.columns if target in c), None)
                final_cols[target] = found
            
            if not final_cols['lat'] or not final_cols['lon']:
                return None, "Missing Latitude or Longitude columns."
                
            # Construct standard DF
            out_df = pd.DataFrame()
            out_df['Site_ID'] = df[final_cols['id']] if final_cols['id'] else df.index.astype(str)
            out_df['Latitude'] = df[final_cols['lat']]
            out_df['Longitude'] = df[final_cols['lon']]
            out_df['Tower_Height'] = df[final_cols['height']] if final_cols['height'] else 10.0
            
            return out_df, None

        elif filename.endswith('.kml') or filename.endswith('.kmz'):
            try:
                content = uploaded_file.read()
                
                if filename.endswith('.kmz'):
                    with zipfile.ZipFile(io.BytesIO(content)) as z:
                        kml_filename = [n for n in z.namelist() if n.endswith('.kml')][0]
                        content = z.read(kml_filename)
                
                k = kml.KML()
                k.from_string(content)
                
                features = list(k.features())
                placemarks = []
                
                def traverse(feats):
                    for f in feats:
                        if hasattr(f, 'features'):
                            traverse(f.features())
                        if hasattr(f, 'geometry'):
                             if f.geometry.geom_type == 'Point':
                                 coords = f.geometry.coords[0]
                                 # KML is usually Lon, Lat, Alt
                                 placemarks.append({
                                     'Site_ID': f.name if f.name else "Unknown",
                                     'Latitude': coords[1],
                                     'Longitude': coords[0],
                                     'Tower_Height': coords[2] if len(coords) > 2 else 10.0
                                 })
                
                traverse(features)
                
                if not placemarks:
                    return None, "No Placemarks found in KML/KMZ."
                    
                return pd.DataFrame(placemarks), None
                
            except Exception as e:
                return None, f"KML/KMZ Parsing Error: {str(e)}"
        
        else:
            return None, "Unsupported file format."
            
    except Exception as e:
        return None, f"General Error: {str(e)}"
