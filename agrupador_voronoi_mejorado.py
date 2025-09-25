# agrupador_voronoi_mejorado.py
# Balanced geo-clustering into K groups + finite Voronoi map.
# Key ideas:
# - Local metric projection for distance in meters (approx).
# - Finite Voronoi regions clipped to a bounding box.
# - Soft workload balance via size_min/size_max constraints.
# - Optional operator names from CLI.
# - Stable coloring and simple legend.

import argparse
import math
import numpy as np
import pandas as pd
from k_means_constrained import KMeansConstrained
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import unary_union
from scipy.spatial import Voronoi
import folium
import colorsys
import itertools

# ===================== Column normalization =====================

ALIASES = {
    "codigo":"codigo","codi":"codigo","codig":"codigo","códig":"codigo",
    "domicilio":"domicilio","domicili":"domicilio",
    "población":"poblacion","poblacio":"poblacion","poblacion":"poblacion",
    "zona":"zona","zona asignació":"zona","zona asignacion":"zona",
    "operario":"operario","operario avisos":"operario",
    "direccion completa":"direccion","dirección completa":"direccion",
    "latitud":"lat","longitud":"lon","lat":"lat","lon":"lon",
}

def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize header names to a small, consistent set."""
    def _n(s: str) -> str:
        s = s.strip().lower()
        rep=(("á","a"),("à","a"),("é","e"),("è","e"),("í","i"),("ï","i"),
             ("ó","o"),("ò","o"),("ú","u"),("ü","u"))
        for a,b in rep: s=s.replace(a,b)
        return ALIASES.get(s, s)
    return df.rename(columns={c: _n(c) for c in df.columns})

def to_float_comma(x):
    """Robust float parser that handles comma as decimal separator."""
    if pd.isna(x): return np.nan
    if isinstance(x, (int,float,np.integer,np.floating)): return float(x)
    s = str(x).strip()
    if "," in s and "." in s: s = s.replace(".", "")  # thousands
    s = s.replace(",", ".")
    try: return float(s)
    except: return np.nan

# ===================== Load and clean =====================

DEFAULT_BAD_LAT = 41.3873974
DEFAULT_BAD_LON = 2.168568

def load_and_clean(path: str, bad_lat=DEFAULT_BAD_LAT, bad_lon=DEFAULT_BAD_LON) -> pd.DataFrame:
    """Load Excel, normalize columns, parse coords, flag unusable rows."""
    df = pd.read_excel(path)
    df = norm_cols(df)

    if "lat" not in df.columns or "lon" not in df.columns:
        raise ValueError("No encuentro columnas de coordenadas ('Latitud/Longitud' o 'lat/lon').")

    if "zona" in df.columns:
        df = df.rename(columns={"zona":"zona_actual"})

    df["lat"] = df["lat"].apply(to_float_comma)
    df["lon"] = df["lon"].apply(to_float_comma)

    df["coord_sospechosa"] = (
        (df["lat"].round(7) == round(bad_lat,7)) &
        (df["lon"].round(6) == round(bad_lon,6))
    )
    df["coord_fuera_rango"] = (~df["lat"].between(-90,90)) | (~df["lon"].between(-180,180))
    df["usable"] = (~df["coord_sospechosa"]) & (~df["coord_fuera_rango"]) & df["lat"].notna() & df["lon"].notna()

    return df

# ===================== Local metric projection (meters) =====================

def project_to_meters(lat, lon, lat0=None):
    """Project (lat, lon) to a local equirectangular metric plane (meters approx.)."""
    R = 6371000.0
    if lat0 is None:
        lat0 = np.nanmean(lat)
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    lat0_rad = math.radians(lat0)
    x = R * (lon_rad - np.nanmean(lon_rad)) * math.cos(lat0_rad)
    y = R * (lat_rad - np.nanmean(lat_rad))
    return x, y, lat0

# ===================== Balanced K-Means =====================

def balanced_clustering(df: pd.DataFrame, k: int, random_state: int = 42):
    """Constrained K-Means on projected coordinates to balance workload sizes."""
    usable = df[df["usable"]].copy()
    if usable.empty:
        raise ValueError("No hay puntos utilizables tras la limpieza.")

    x, y, lat0 = project_to_meters(usable["lat"].to_numpy(), usable["lon"].to_numpy())
    XY = np.column_stack([x,y])
    n = len(usable)

    size_min = n // k
    size_max = size_min + (1 if n % k else 0)

    kmc = KMeansConstrained(
        n_clusters=k,
        size_min=size_min,
        size_max=size_max,
        random_state=random_state
    )
    labels = kmc.fit_predict(XY)

    df["cluster_id"] = np.nan
    df.loc[usable.index, "cluster_id"] = labels

    # Centroids in projected plane
    cent_xy = np.vstack([XY[labels==c].mean(axis=0) for c in range(k)])

    # Approximate inverse projection back to (lat, lon) for display
    lat_mean = float(np.nanmean(usable["lat"]))
    lon_mean = float(np.nanmean(usable["lon"]))
    R = 6371000.0
    cent_lats = (cent_xy[:,1] / R) + math.radians(lat_mean)
    cent_lats = np.degrees(cent_lats)
    cent_lons = (cent_xy[:,0] / (R * math.cos(math.radians(lat_mean)))) + math.radians(lon_mean)
    cent_lons = np.degrees(cent_lons)
    centroids = np.column_stack([cent_lats, cent_lons])

    return df, centroids

# ===================== Finite Voronoi =====================

def voronoi_finite_polygons_2d(vor, radius=1e6):
    """
    Build finite polygons from a 2D Voronoi diagram.
    Returns (regions, vertices). Adapted from SciPy Cookbook.
    """
    if vor.points.shape[1] != 2:
        raise ValueError("Solo 2D soportado.")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if np.any(vor.ridge_vertices == -1):
        ptp_bound = vor.points.ptp(axis=0)
    else:
        ptp_bound = np.array([0,0])

    for point_idx, region_idx in enumerate(vor.point_region):
        vertices = vor.regions[region_idx]

        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue

        # Reconstruct infinite region
        ridges = vor.ridge_vertices
        ridge_points = vor.ridge_points
        new_region = [v for v in vertices if v >= 0]

        for (p1, p2), (v1, v2) in zip(ridge_points, ridges):
            if p1 != point_idx and p2 != point_idx:
                continue
            if v1 >= 0 and v2 >= 0:
                continue

            t = vor.points[p2] - vor.points[p1]
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # outward normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v1 if v1 >= 0 else v2] + direction * ptp_bound.max() * 2

            new_vertices.append(far_point.tolist())
            new_region.append(len(new_vertices) - 1)

        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = [v for _, v in sorted(zip(angles, new_region))]
        new_regions.append(new_region)

    return new_regions, np.asarray(new_vertices)

def make_bbox(lats, lons, expand=0.02):
    """Build a padded geographic bounding box polygon."""
    minx,maxx=float(np.nanmin(lons)),float(np.nanmax(lons))
    miny,maxy=float(np.nanmin(lats)),float(np.nanmax(lats))
    dx,dy=(maxx-minx),(maxy-miny)
    pad_x=max(dx*0.1,expand); pad_y=max(dy*0.1,expand)
    return Polygon([
        (minx-pad_x,miny-pad_y),(maxx+pad_x,miny-pad_y),
        (maxx+pad_x,maxy+pad_y),(minx-pad_x,maxy+pad_y)
    ])

def voronoi_polys_finite(centroids, bbox_poly):
    """Compute finite Voronoi cells and clip them to the bbox."""
    pts = np.column_stack([centroids[:,1], centroids[:,0]])  # (lon,lat)
    vor = Voronoi(pts)
    regions, vertices = voronoi_finite_polygons_2d(vor)

    polys = []
    for reg in regions:
        poly_coords = vertices[reg]
        poly = Polygon(poly_coords)
        poly = poly.intersection(bbox_poly)
        polys.append(poly)
    return polys

# ===================== Colors =====================

def stable_colors(n, s=0.55, v=0.95):
    """Generate n distinct hex colors spaced around HSV hue circle."""
    hues = np.linspace(0, 1, n, endpoint=False)
    cols = []
    for h in hues:
        r,g,b = colorsys.hsv_to_rgb(h, s, v)
        cols.append("#%02x%02x%02x" % (int(r*255), int(g*255), int(b*255)))
    return cols

# ===================== Map =====================

def build_map(df, centroids, grupos, html_out):
    """Render Voronoi polygons and points into a Folium map."""
    dfu = df[df["usable"]]
    m = folium.Map(location=[dfu["lat"].mean(), dfu["lon"].mean()], zoom_start=12)

    colors = stable_colors(len(grupos))
    color_by_group = {g:c for g,c in zip(grupos, colors)}

    bbox = make_bbox(df["lat"].values, df["lon"].values, expand=0.03)
    polys = voronoi_polys_finite(centroids, bbox)

    # Voronoi polygons
    for cid, poly in enumerate(polys):
        if poly.is_empty: 
            continue
        if isinstance(poly, (LineString, Point)):
            continue
        coords = [(y,x) for x,y in list(poly.exterior.coords)]
        folium.Polygon(locations=coords, color=color_by_group[grupos[cid]],
                       fill=True, fill_opacity=0.15, weight=2,
                       popup=f"Zona {grupos[cid]}").add_to(m)

    # Valid points
    for _, r in dfu.iterrows():
        g = r["grupo_nuevo"]
        col = color_by_group.get(g, "#666666")
        folium.CircleMarker([r["lat"],r["lon"]], radius=3,
                            color=col, fill=True, fill_color=col,
                            popup=f"{g} · Código: {r.get('codigo','')}").add_to(m)

    # Suspicious coords
    for _, r in df[df["coord_sospechosa"]].iterrows():
        folium.CircleMarker([r["lat"],r["lon"]], radius=4,
                            color="#ff0000", fill=True, fill_color="#ff0000",
                            popup="COORD SOSPECHOSA").add_to(m)

    # Centroids
    for cid,(la,lo) in enumerate(centroids):
        folium.Marker([la,lo], tooltip=f"{grupos[cid]}",
                      icon=folium.Icon(icon="wrench", prefix="fa")).add_to(m)

    # Simple legend
    legend_html = """<div style="position: fixed; 
                                 bottom: 20px; left: 20px; width: 260px; 
                                 z-index:9999; font-size:12px; 
                                 background: white; padding:10px; 
                                 border:1px solid #999; border-radius:8px;">
    <b>Grupos / Operarios</b><br>
    """ + "<br>".join([f'<span style="display:inline-block;width:12px;height:12px;background:{color_by_group[g]};margin-right:6px;"></span>{g}' for g in grupos]) + "</div>"
    m.get_root().html.add_child(folium.Element(legend_html))

    m.save(html_out)
    print(f"🗺️  Mapa guardado en {html_out}")

# ===================== Main =====================

def main():
    ap = argparse.ArgumentParser(description="Agrupación balanceada con Voronoi finito y proyección métrica.")
    ap.add_argument("excel_path", type=str, help="Ruta al Excel de entrada")
    ap.add_argument("--k", type=int, default=None, help="Nº de grupos/operarios. Si no se da, usa len(operarios).")
    ap.add_argument("--operarios", type=str, default="", help="Lista separada por comas con los nombres de operarios.")
    ap.add_argument("--salida-excel", type=str, default="asignacion_balanceada.xlsx")
    ap.add_argument("--salida-mapa", type=str, default="territorios_voronoi.html")
    ap.add_argument("--bad-lat", type=float, default=DEFAULT_BAD_LAT)
    ap.add_argument("--bad-lon", type=float, default=DEFAULT_BAD_LON)
    ap.add_argument("--seed", type=int, default=42, help="Semilla aleatoria para reproducibilidad.")
    args = ap.parse_args()

    # Group names
    grupos = [s.strip() for s in args.operarios.split(",") if s.strip()]
    if args.k is None:
        if grupos:
            k = len(grupos)
        else:
            k = 12
            grupos = [f"Grupo {i+1}" for i in range(k)]
    else:
        k = int(args.k)
        if not grupos:
            grupos = [f"Grupo {i+1}" for i in range(k)]
        elif len(grupos) != k:
            raise ValueError(f"Número de operarios ({len(grupos)}) != k ({k}).")

    df = load_and_clean(args.excel_path, args.bad_lat, args.bad_lon)
    df_out, centroids = balanced_clustering(df, k=k, random_state=args.seed)

    # Map cluster_id → group label
    df_out["grupo_nuevo"] = df_out["cluster_id"].apply(
        lambda v: grupos[int(v)] if pd.notna(v) and int(v)<len(grupos) else ""
    )

    # Size summary
    sizes = (df_out[df_out["usable"]]
             .groupby("grupo_nuevo")
             .size()
             .rename("total")
             .reset_index()
             .sort_values("total", ascending=False))

    # Save Excel
    with pd.ExcelWriter(args.salida_excel, engine="xlsxwriter") as w:
        df_out.to_excel(w, index=False, sheet_name="Agrupacion_Nueva")
        df_out[df_out["coord_sospechosa"]].to_excel(w, index=False, sheet_name="Coord_Sospechosas")
        sizes.to_excel(w, index=False, sheet_name="Resumen")

    print(f"✅ Excel guardado en {args.salida_excel}")
    build_map(df_out, centroids, grupos, args.salida_mapa)

if __name__ == "__main__":
    main()
