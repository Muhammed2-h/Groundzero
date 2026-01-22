from utils import analyze_path
import time

s = time.time()
print("Starting debug analysis for Kochi -> Munnar...")

# Use the exact defaults from app.py
# A: 9.9312, 76.2673
# B: 10.0889, 77.0595
res = analyze_path(9.9312, 76.2673, 10.0889, 77.0595)

e = time.time()
print(f"Analysis took {e-s:.2f} seconds")

if res['status'] == 'Success':
    print("Success!")
    print(f"Distance: {res['dataframe']['distance_km'].max()} km")
    print(f"Max Obstruction: {res['max_obstruction_height']}")
    print(f"Rows: {len(res['dataframe'])}")
else:
    print("Failed")
    print(res)
