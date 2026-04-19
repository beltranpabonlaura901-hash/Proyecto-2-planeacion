import dash
from dash import Input, Output, State, callback_context, html, dcc
import pandas as pd
import dash_bootstrap_components as dbc

from core.datos import (
    PRODUCTOS, MESES, TAMANO_LOTE_BASE,
    CAPACIDAD_BASE, DEM_HISTORICA, DEM_HH, PROD_COLORS
)

from core.calculos import (
    calc_agregacion, calc_desagregacion,
    calc_simulacion, calc_kpis, calc_utilizacion
)

from core.utils_json import _j, _rj, _j_dict, _rj_dict

from ui.controles import CTRL_DEM, CTRL_MAP, NAV

from ui.componentes import titulo, kpi_box, tbl, no_data

from ui.graficas import (
    fig_dem_barras, fig_dem_heatmap, fig_dem_hh,
    fig_agr_plan, fig_agr_laboral,
    fig_desag_grid, fig_desag_hh,
    fig_gantt, fig_colas,
    fig_gauges, fig_radar,
    fig_comparacion, fig_sensores
)

def register_callbacks(app):


  @app.callback(
    Output("st-tab","data"),
    Output("hdr","children"),
    Output("ctrl-panel","children"),
    [Input(f"nav-{t}","n_clicks") for _,_,t in NAV],
    prevent_initial_call=False,
)
def nav(*_):
    ctx = callback_context
    if not ctx.triggered or ctx.triggered[0]["prop_id"] == ".":
        return "tab-dem", "DEMANDA", CTRL_DEM

    tid = ctx.triggered[0]["prop_id"].split(".")[0].replace("nav-","")
    lbl_ = next(l for _,l,t in NAV if t==tid)

    return tid, lbl_, CTRL_MAP.get(tid, html.Div())

@app.callback(
    Output("st-agr","data"),
    Output("st-costo","data"),
    Output("agr-status","children"),
    Input("btn-agr","n_clicks"),
    State("agr-ct","value"),
    State("agr-ht","value"),
    State("agr-pit","value"),
    State("agr-crt","value"),
    State("agr-cot","value"),
    State("agr-cwm","value"),
    State("agr-cwd","value"),
    State("agr-trab","value"),
    State("agr-hsem","value"),
    State("agr-nsem","value"),
    State("agr-dw","value"),
    prevent_initial_call=True,
)
def cb_agr(n, ct, ht, pit, crt, cot, cwm, cwd, trab, hsem, nsem, dw):
    if not n:
        return dash.no_update, dash.no_update, dash.no_update

    try:
        lr = (trab or 10) * (hsem or 44) * (nsem or 4)

        df, costo = calc_agregacion(
            lr_ini=lr,
            ct=ct or 4310,
            ht=ht or 100000,
            pit=pit or 100000,
            crt=crt or 11364,
            cot=cot or 14205,
            cwm=cwm or 14204,
            cwd=cwd or 15061,
            dw=dw or 50
        )

        return _j(df), costo, f"✓ Costo óptimo: {costo}"

    except Exception as e:
        return dash.no_update, dash.no_update, f"Error: {e}"

  @app.callback(
    Output("st-desag","data"),
    Output("desag-status","children"),
    Input("btn-desag","n_clicks"),
    State("st-agr","data"),
    State("desag-cp","value"),
    State("desag-ci","value"),
    prevent_initial_call=True,
)
def cb_desag(n, agr_j, cp, ci):
    if not n:
        return dash.no_update, dash.no_update

    try:
        df_agr = _rj(agr_j)
        prod_hh = dict(zip(df_agr["Mes"], df_agr["Produccion_HH"]))

        res = calc_desagregacion(prod_hh, cp or 1.0, ci or 1.0)

        return _j_dict(res), "✓ OK"

    except Exception as e:
        return dash.no_update, f"Error: {e}"
      @app.callback(
    Output("st-lotes","data"),
    Output("st-uso","data"),
    Output("st-sens","data"),
    Output("st-kpi","data"),
    Output("st-util","data"),
    Output("st-plan","data"),
    Output("sim-status","children"),
    Input("btn-sim","n_clicks"),
    State("st-desag","data"),
    State("sim-mes","value"),
    prevent_initial_call=True,
)
def cb_sim(n, desag_j, mes_idx):
    if not n:
        return [dash.no_update]*7

    try:
        desag = _rj_dict(desag_j)
        mes_nm = MESES[mes_idx or 1]

        plan = {
            p: int(desag[p].loc[desag[p]["Mes"]==mes_nm,"Produccion_und"])
            for p in PRODUCTOS
        }

        df_l, df_u, df_s = calc_simulacion(plan)
        df_kpi = calc_kpis(df_l, plan)
        df_ut = calc_utilizacion(df_u)

        return _j(df_l), _j(df_u), _j(df_s), _j(df_kpi), _j(df_ut), plan, "✓ Sim OK"

    except Exception as e:
        return [dash.no_update]*7

@app.callback(
    Output("st-lotes","data"),
    Output("st-uso","data"),
    Output("st-sens","data"),
    Output("st-kpi","data"),
    Output("st-util","data"),
    Output("st-plan","data"),
    Output("sim-status","children"),
    Input("btn-sim","n_clicks"),
    State("st-desag","data"),
    State("sim-mes","value"),
    prevent_initial_call=True,
)
def cb_sim(n, desag_j, mes_idx):
    if not n:
        return [dash.no_update]*7

    try:
        desag = _rj_dict(desag_j)
        mes_nm = MESES[mes_idx or 1]

        plan = {
            p: int(desag[p].loc[desag[p]["Mes"]==mes_nm,"Produccion_und"])
            for p in PRODUCTOS
        }

        df_l, df_u, df_s = calc_simulacion(plan)
        df_kpi = calc_kpis(df_l, plan)
        df_ut = calc_utilizacion(df_u)

        return _j(df_l), _j(df_u), _j(df_s), _j(df_kpi), _j(df_ut), plan, "✓ Sim OK"

    except Exception as e:
        return [dash.no_update]*7

def register_callbacks(app):

    @app.callback(...)
    def nav(...):
        ...

    @app.callback(...)
    def cb_agr(...):
        ...

    @app.callback(...)
    def cb_desag(...):
        ...

    @app.callback(...)
    def cb_sim(...):
        ...

    @app.callback(...)
    def render(...):
        ...
