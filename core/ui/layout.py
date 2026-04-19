# ui/layout.py
from dash import dcc, html

from core.utils_json import _j, _j_dict

def build_sidebar(NAV):
    return html.Div([
        html.Div([
            html.Div("◈", style={"fontSize":"28px","color":"#E8A838"}),
            html.Div("DORA DEL HOYO", style={"fontSize":"11px","fontWeight":"700"}),
            html.Div("GEMELO DIGITAL", style={"fontSize":"9px","color":"#E8A838"}),
        ], style={"padding":"22px 16px 18px","borderBottom":"1px solid #dee2e6","marginBottom":"10px"}),
        *[
            html.Button(
                [html.Span(n, style={"fontSize":"9px","color":"#E8A838","marginRight":"8px"}),
                 html.Span(l, style={"fontSize":"12px","letterSpacing":"0.1em"})],
                id=f"nav-{t}",
                n_clicks=0,
                style={
                    "background":"transparent","border":"none","color":"#6c757d",
                    "width":"100%","textAlign":"left","padding":"10px 16px","cursor":"pointer"
                }
            )
            for n, l, t in NAV
        ],
    ], style={
        "width":"200px","minHeight":"100vh","background":"#f8f9fa",
        "borderRight":"1px solid #dee2e6","position":"fixed","top":0,"left":0,"zIndex":100
    })

def build_stores(init_data):
    return html.Div([
        dcc.Store(id="st-tab", data="tab-dem"),
        dcc.Store(id="st-agr", data=_j(init_data["DF_AGR_INIT"])),
        dcc.Store(id="st-costo", data=init_data["COSTO_AGR_INIT"]),
        dcc.Store(id="st-desag", data=_j_dict(init_data["DESAG_INIT"])),
        dcc.Store(id="st-lotes", data=_j(init_data["DF_LOTES_INIT"])),
        dcc.Store(id="st-uso", data=_j(init_data["DF_USO_INIT"])),
        dcc.Store(id="st-sens", data=_j(init_data["DF_SENS_INIT"])),
        dcc.Store(id="st-kpi", data=_j(init_data["DF_KPI_INIT"])),
        dcc.Store(id="st-util", data=_j(init_data["DF_UTIL_INIT"])),
        dcc.Store(id="st-plan", data=init_data["PLAN_MES_INIT"]),
        dcc.Store(id="st-esc", data={}),
        dcc.Store(id="st-temp", data={"base":160, "lim":200}),
    ])

def build_layout(NAV, init_data):
    stores = build_stores(init_data)
    sidebar = build_sidebar(NAV)

    return html.Div([
        stores,
        sidebar,
        html.Div([
            html.Div([
                html.Div(id="hdr", style={"fontSize":"28px","fontWeight":"700","color":"#212529"}),
            ], style={"padding":"18px 24px 14px","borderBottom":"1px solid #dee2e6","background":"#f8f9fa"}),
            html.Div(id="ctrl-panel", style={"padding":"14px 24px 0"}),
            html.Hr(style={"borderColor":"#dee2e6","margin":"0 24px"}),
            dcc.Loading(type="dot", color="#E8A838",
                children=html.Div(id="content", style={"padding":"16px 24px 40px"})
            ),
        ], style={"marginLeft":"200px","minHeight":"100vh","background":"#f0f2f5"}),
    ])
