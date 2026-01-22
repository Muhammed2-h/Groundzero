# 🗺️ Terrain Line of Sight Analyzer

A professional-grade **rf/terrain analysis tool** built with Python and Streamlit. This application calculates the Fresnel Zone and Line of Sight (LoS) between two geographic points using real-world elevation data, accounting for Earth's curvature.

## 🚀 Live Demo
*(Once you deploy, put your link here: https://your-app-name.streamlit.app)*

## ✨ Features
*   **Real-time Elevation Profile**: Visualizes terrain between any two global coordinates.
*   **Earth Curvature & Refraction**: Uses `k=1.33` effective earth radius model for accurate RF planning.
*   **Dynamic Tower Heights**: Adjustable antenna heights (0-500m) to simulate towers, masts, or drones.
*   **Obstruction Detection**: Automatically highlights terrain blocking the signal path.
*   **Batch Analysis**: Upload a CSV to process hundreds of links at once.
*   **Adaptive Sampling**: Intelligent resolution scaling for fast results (5x faster than standard).

---

## 🛠️ How to Run Locally

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/your-username/terrain-analyzer.git
    cd terrain-analyzer
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the App**
    ```bash
    streamlit run app.py
    ```
    The app will open in your browser at `http://localhost:8501`.

---

## ☁️ How to Deploy (Free)

The easiest way to host this for free is **Streamlit Community Cloud**.

### Steps:
1.  **Push to GitHub**:
    *   Upload all files (`app.py`, `terrain_utils.py`, `requirements.txt`) to a GitHub repository.

2.  **Deploy on Streamlit**:
    *   Go to [share.streamlit.io](https://share.streamlit.io/).
    *   Log in with GitHub.
    *   Click **"New App"**.
    *   Select your repository (`terrain-analyzer`).
    *   Set **Main file path** to `app.py`.
    *   Click **Deploy**.

That's it! Your app will be live in minutes.

---

### 📂 Batch CSV Format
If using Batch Mode, your CSV should look like this:
```csv
A_name, A_lat, A_long, B_name, B_lat, B_long
Station1, 9.9312, 76.2673, Remote1, 10.0889, 77.0595
Station2, 28.6139, 77.2090, Remote2, 19.0760, 72.8777
```
*(A template is also available for download inside the app)*
