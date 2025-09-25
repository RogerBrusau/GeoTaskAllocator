# app.py
# Streamlit UI wrapper for the CLI script `agrupador_voronoi_mejorado.py`.
# Purpose:
# - Accept an Excel with coordinates and optional operators list.
# - Invoke the CLI with user-selected params (k, seed).
# - Provide downloads (Excel) and render the generated HTML map.
# Notes:
# - Keeps dependencies minimal; runs the core logic out of process via subprocess.
# - Forces UTF-8 to avoid Unicode issues on Windows consoles.

import streamlit as st
import tempfile, subprocess, os
from pathlib import Path
from datetime import datetime

st.set_page_config(page_title="Reparto óptimo de ascensores", layout="wide")

st.title("Reparto óptimo de ascensores")
st.caption("UI para agrupador_voronoi_mejorado.py")

# --- Inputs ---
xlsx_file = st.file_uploader("Sube Llistat_Ascensor_Geo.xlsx", type=["xlsx"])
with st.expander("Opcional: lista de operarios (ruta o archivo)"):
    operarios_txt = st.text_input("Ruta local a archivo de operarios (opcional)", value="")
    operarios_upload = st.file_uploader("…o sube un archivo de operarios", type=["txt","csv"], key="ops")

col1, col2 = st.columns(2)
k = col1.number_input("Número de grupos (k)", min_value=2, max_value=30, value=12, step=1)
seed = col2.number_input("Seed", min_value=0, value=42, step=1)

run_btn = st.button("Ejecutar")

# Output directory (created once)
out_dir = Path("salida_ui")
out_dir.mkdir(exist_ok=True)

def save_upload_temp(upload, suffix):
    """Persist an uploaded file to a named temporary file. Returns filesystem path."""
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tf.write(upload.read())
    tf.flush()
    tf.close()
    return tf.name

if run_btn:
    if not xlsx_file:
        st.error("Falta el Excel de entrada.")
        st.stop()

    # Persist inputs to disk for the CLI call
    in_excel_path = save_upload_temp(xlsx_file, ".xlsx")

    operarios_path = None
    if operarios_upload is not None:
        operarios_path = save_upload_temp(operarios_upload, ".txt")
    elif operarios_txt.strip():
        operarios_path = operarios_txt.strip()

    # Timestamped outputs to defeat browser cache and allow multiple runs
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    xlsx_out = out_dir / f"asignaciones_{ts}.xlsx"
    html_out = out_dir / f"mapa_{ts}.html"

    # CLI command mapped to the script's current interface
    cmd = [
        "python", "agrupador_voronoi_mejorado.py",
        in_excel_path,                     # positional: excel_path
        "--k", str(k),
        "--seed", str(seed),
        "--salida-excel", str(xlsx_out),
        "--salida-mapa", str(html_out),
    ]
    if operarios_path:
        cmd.extend(["--operarios", str(operarios_path)])

    # Ensure UTF-8 to avoid UnicodeEncodeError on some Windows setups
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)

    st.subheader("Log de ejecución")
    st.code((res.stdout or "") + ("\n" + res.stderr if res.stderr else ""))

    if res.returncode != 0:
        st.error("El proceso devolvió un código distinto de 0. Revisa el log.")
        st.stop()

    # Downloads (Excel results)
    if xlsx_out.exists():
        st.download_button(
            "Descargar asignaciones.xlsx",
            data=xlsx_out.read_bytes(),
            file_name=xlsx_out.name
        )
    else:
        st.warning("No se encontró el Excel de salida esperado.")

    # Inline map rendering (HTML produced by the CLI)
    if html_out.exists():
        html = html_out.read_text(encoding="utf-8")
        st.subheader("Mapa generado")
        st.components.v1.html(html, height=720, scrolling=True)
    else:
        st.warning("No se encontró el HTML del mapa de salida.")
