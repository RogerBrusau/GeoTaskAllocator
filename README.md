# GeoTaskAllocator

GeoTaskAllocator is a Python tool for **balanced spatial clustering and territory allocation**.  
It was developed to optimize the distribution of geographically dispersed assets or tasks among multiple operators, ensuring both **fair workload balance** and **geographic efficiency**.

---

## ✨ Key Features
- **Balanced K-Means clustering** with minimum and maximum group size constraints.  
- **Finite Voronoi tessellation** clipped to geographic bounds for intuitive territory visualization.  
- **Flexible input**: any Excel file with latitude/longitude coordinates and optional operator names.  
- **Dual usage modes**:  
  - **CLI** for reproducible batch execution.  
  - **Streamlit UI** for interactive use and visual inspection.  
- **Automatic reports**: generates an Excel file (new allocation + summary) and an interactive HTML map.

---

## 🔧 Technical Overview
- **Algorithms**:  
  - Constrained K-Means clustering (`k-means-constrained`).  
  - Custom Voronoi polygon construction with bounding box clipping.  
- **Libraries**:  
  - `numpy`, `pandas`, `scipy`, `k-means-constrained`, `folium`, `openpyxl`, `streamlit` (optional UI).  
- **Reproducibility**:  
  - Random seed control ensures stable and repeatable assignments.  
- **Code Quality**:  
  - Modular design (`geo_task_allocator.py`, `app.py`, `src/` modules).  
  - English docstrings and comments only where necessary for clarity.

---

## 📂 Repository Structure

```text
GeoTaskAllocator/
├── geo_task_allocator.py    # Core CLI script (balanced clustering + Voronoi map)
├── app.py                   # Optional Streamlit interface
├── requirements.txt         # Core dependencies (no pinned versions)
├── README.md                # Project documentation
├── data/                    # Example anonymized inputs (optional)
└── salida_ui/               # Auto-created output folder (maps, excels)
