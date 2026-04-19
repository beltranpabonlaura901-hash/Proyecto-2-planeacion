# app.py
import warnings
import dash
import dash_bootstrap_components as dbc

from core.datos import PRODUCTOS
from core.calculos import (
    calc_agregacion, calc_desagregacion, calc_simulacion,
    calc_utilizacion, calc_kpis
)
from ui.controles import NAV
from ui.layout import build_layout
from ui.callbacks import register_callbacks

warnings.filterwarnings("ignore")

print("▶ Calculando plan agregado inicial...")
DF_AGR_INIT, COSTO_AGR_INIT = calc_agregacion()
PROD_HH_INIT = dict(zip(DF_AGR_INIT["Mes"], DF_AGR_INIT["Produccion_HH"]))

print("▶ Calculando desagregación inicial...")
DESAG_INIT = calc_desagregacion(PROD_HH_INIT)

PLAN_MES_INIT = {p: int(DESAG_INIT[p]["Produccion_und"].mean()) for p in PRODUCTOS}

print("▶ Ejecutando simulación base...")
DF_LOTES_INIT, DF_USO_INIT, DF_SENS_INIT = calc_simulacion(PLAN_MES_INIT)
DF_UTIL_INIT = calc_utilizacion(DF_USO_INIT)
DF_KPI_INIT = calc_kpis(DF_LOTES_INIT, PLAN_MES_INIT)
print("✓ Todos los datos iniciales listos.")

init_data = {
    "DF_AGR_INIT": DF_AGR_INIT,
    "COSTO_AGR_INIT": COSTO_AGR_INIT,
    "DESAG_INIT": DESAG_INIT,
    "PLAN_MES_INIT": PLAN_MES_INIT,
    "DF_LOTES_INIT": DF_LOTES_INIT,
    "DF_USO_INIT": DF_USO_INIT,
    "DF_SENS_INIT": DF_SENS_INIT,
    "DF_UTIL_INIT": DF_UTIL_INIT,
    "DF_KPI_INIT": DF_KPI_INIT,
}

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    title="Gemelo Digital — Dora del Hoyo"
)
server = app.server

app.layout = build_layout(NAV, init_data)
register_callbacks(app)

if __name__ == "__main__":
    app.run(debug=True, port=8050)
