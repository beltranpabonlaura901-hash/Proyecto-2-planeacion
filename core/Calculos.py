# core/calculos.py
import math
import random
import numpy as np
import pandas as pd
import simpy

from pulp import LpProblem, LpMinimize, LpVariable, lpSum, value, PULP_CBC_CMD

from core.datos import (
    PRODUCTOS, MESES, DEM_HISTORICA, HORAS_PRODUCTO,
    INV_INICIAL, RUTAS, TAMANO_LOTE_BASE, CAPACIDAD_BASE, DEM_HH
)

def calc_agregacion(lr_ini=1760.0, ct=4310, ht=100000, pit=100000,
                    crt=11364, cot=14205, cwm=14204, cwd=15061, M=1, dw=50):
    mdl = LpProblem("Agr", LpMinimize)
    P  = LpVariable.dicts("P",  MESES, lowBound=0)
    Iv = LpVariable.dicts("I",  MESES, lowBound=0)
    S  = LpVariable.dicts("S",  MESES, lowBound=0)
    LR = LpVariable.dicts("LR", MESES, lowBound=0)
    LO = LpVariable.dicts("LO", MESES, lowBound=0)
    LU = LpVariable.dicts("LU", MESES, lowBound=0)
    NI = LpVariable.dicts("NI", MESES)
    Wm = LpVariable.dicts("Wm", MESES, lowBound=0)
    Wd = LpVariable.dicts("Wd", MESES, lowBound=0)

    mdl += lpSum(ct*P[t] + ht*Iv[t] + pit*S[t] + crt*LR[t] + cot*LO[t] + cwm*Wm[t] + cwd*Wd[t] for t in MESES)

    for idx, t in enumerate(MESES):
        d = DEM_HH[t]
        tp = MESES[idx - 1] if idx > 0 else None
        mdl += (NI[t] == P[t] - d) if idx == 0 else (NI[t] == NI[tp] + P[t] - d)
        mdl += NI[t] == Iv[t] - S[t]
        mdl += LU[t] + LO[t] == M * P[t]
        mdl += LU[t] <= LR[t]
        mdl += (LR[t] == lr_ini + Wm[t] - Wd[t]) if idx == 0 else (LR[t] == LR[tp] + Wm[t] - Wd[t])
        mdl += Wm[t] <= dw
        mdl += Wd[t] <= dw

    mdl.solve(PULP_CBC_CMD(msg=False))
    costo = value(mdl.objective) or 0

    ini_l, fin_l = [], []
    for idx, t in enumerate(MESES):
        ini = 0.0 if idx == 0 else fin_l[-1]
        ini_l.append(ini)
        fin_l.append(ini + (P[t].varValue or 0) - DEM_HH[t])

    df = pd.DataFrame({
        "Mes": MESES,
        "Demanda_HH": [round(DEM_HH[t], 2) for t in MESES],
        "Produccion_HH": [round(P[t].varValue or 0, 2) for t in MESES],
        "Inv_Ini_HH": [round(v, 2) for v in ini_l],
        "Inv_Fin_HH": [round(v, 2) for v in fin_l],
        "Backlog_HH": [round(S[t].varValue or 0, 2) for t in MESES],
        "H_Regulares": [round(LR[t].varValue or 0, 2) for t in MESES],
        "H_Extras": [round(LO[t].varValue or 0, 2) for t in MESES],
        "Contratacion": [round(Wm[t].varValue or 0, 2) for t in MESES],
        "Despidos": [round(Wd[t].varValue or 0, 2) for t in MESES],
    })
    return df, costo

def calc_desagregacion(prod_hh, cost_prod=1.0, cost_inv=1.0):
    mdl = LpProblem("Desag", LpMinimize)
    X  = {(p,t): LpVariable(f"X_{p}_{t}", lowBound=0) for p in PRODUCTOS for t in MESES}
    Iv = {(p,t): LpVariable(f"I_{p}_{t}", lowBound=0) for p in PRODUCTOS for t in MESES}
    Sv = {(p,t): LpVariable(f"S_{p}_{t}", lowBound=0) for p in PRODUCTOS for t in MESES}

    mdl += lpSum(cost_inv * Iv[p,t] + cost_prod * 10000 * Sv[p,t] for p in PRODUCTOS for t in MESES)

    for idx, t in enumerate(MESES):
        tp = MESES[idx - 1] if idx > 0 else None
        mdl += lpSum(HORAS_PRODUCTO[p] * X[p,t] for p in PRODUCTOS) <= prod_hh[t]
        for p in PRODUCTOS:
            d = DEM_HISTORICA[p][idx]
            if idx == 0:
                mdl += Iv[p,t] - Sv[p,t] == INV_INICIAL[p] + X[p,t] - d
            else:
                mdl += Iv[p,t] - Sv[p,t] == Iv[p,tp] - Sv[p,tp] + X[p,t] - d

    mdl.solve(PULP_CBC_CMD(msg=False))

    results = {}
    for p in PRODUCTOS:
        rows = []
        for idx, t in enumerate(MESES):
            xv = round(X[p,t].varValue or 0, 2)
            iv = round(Iv[p,t].varValue or 0, 2)
            sv = round(Sv[p,t].varValue or 0, 2)
            ini = INV_INICIAL[p] if idx == 0 else round(Iv[p,MESES[idx-1]].varValue or 0, 2)
            hh = round(xv * HORAS_PRODUCTO[p], 3)
            rows.append({
                "Mes": t,
                "Demanda_und": DEM_HISTORICA[p][idx],
                "Produccion_und": xv,
                "Produccion_HH": hh,
                "Inv_Ini": ini,
                "Inv_Fin": iv,
                "Backlog": sv
            })
        results[p] = pd.DataFrame(rows)
    return results

def calc_simulacion(plan_und, cap_r=None, falla=False, factor_t=1.0,
                    tam_lote=None, semilla=42, tiempos_c=None, temp_horno_base=160):
    random.seed(semilla)
    np.random.seed(semilla)

    if cap_r is None:
        cap_r = CAPACIDAD_BASE.copy()
    if tam_lote is None:
        tam_lote = TAMANO_LOTE_BASE.copy()

    rutas_ef = {}
    for p, etapas in RUTAS.items():
        rutas_ef[p] = [
            (
                eta, rec,
                tiempos_c[rec][0] if tiempos_c and rec in tiempos_c else tmin,
                tiempos_c[rec][1] if tiempos_c and rec in tiempos_c else tmax
            )
            for eta, rec, tmin, tmax in etapas
        ]

    lotes_data, uso_rec, sensores = [], [], []

    def reg_uso(env, recursos):
        ts = round(env.now, 3)
        for nm, r in recursos.items():
            uso_rec.append({
                "tiempo": ts,
                "recurso": nm,
                "ocupados": r.count,
                "cola": len(r.queue),
                "capacidad": r.capacity
            })

    def sensor_horno(env, recursos):
        while True:
            ocp = recursos["horno"].count
            sensores.append({
                "tiempo": round(env.now, 1),
                "temperatura": round(np.random.normal(temp_horno_base + ocp * 20, 5), 2),
                "horno_ocup": ocp,
                "horno_cola": len(recursos["horno"].queue),
            })
            yield env.timeout(10)

    def proceso_lote(env, lid, prod, tam, recursos):
        t0 = env.now
        espera_tot = 0
        for _, rec_nm, tmin, tmax in rutas_ef[prod]:
            escala = math.sqrt(tam / TAMANO_LOTE_BASE[prod])
            tp_proc = random.uniform(tmin, tmax) * escala * factor_t
            if falla and rec_nm == "horno":
                tp_proc += random.uniform(10, 30)
            t_ei = env.now
            with recursos[rec_nm].request() as req:
                yield req
                espera_tot += env.now - t_ei
                reg_uso(env, recursos)
                yield env.timeout(tp_proc)
            reg_uso(env, recursos)

        lotes_data.append({
            "lote_id": lid,
            "producto": prod,
            "tamano": tam,
            "t_creacion": round(t0, 3),
            "t_fin": round(env.now, 3),
            "tiempo_sistema": round(env.now - t0, 3),
            "espera_total": round(espera_tot, 3)
        })

    env = simpy.Environment()
    recursos = {nm: simpy.Resource(env, capacity=cap) for nm, cap in cap_r.items()}
    env.process(sensor_horno(env, recursos))

    dur_mes = 44 * 4 * 60
    lotes = []
    ctr = [0]

    for prod, unid in plan_und.items():
        if unid <= 0:
            continue
        tam = tam_lote[prod]
        n = math.ceil(unid / tam)
        tasa = dur_mes / max(n, 1)
        ta = random.expovariate(1 / max(tasa, 1))
        rem = unid
        for _ in range(n):
            lotes.append((round(ta, 2), prod, min(tam, int(rem))))
            rem -= tam
            ta += random.expovariate(1 / max(tasa, 1))

    lotes.sort(key=lambda x: x[0])

    def lanzador():
        for ta, prod, tam in lotes:
            yield env.timeout(max(ta - env.now, 0))
            lid = f"{prod[:3].upper()}_{ctr[0]:04d}"
            ctr[0] += 1
            env.process(proceso_lote(env, lid, prod, tam, recursos))

    env.process(lanzador())
    env.run(until=dur_mes * 1.8)

    df_l = pd.DataFrame(lotes_data) if lotes_data else pd.DataFrame()
    df_u = pd.DataFrame(uso_rec) if uso_rec else pd.DataFrame()
    df_s = pd.DataFrame(sensores) if sensores else pd.DataFrame()
    return df_l, df_u, df_s

def calc_utilizacion(df_u):
    if df_u.empty:
        return pd.DataFrame()

    filas = []
    for rec, grp in df_u.groupby("recurso"):
        grp = grp.sort_values("tiempo")
        cap = grp["capacidad"].iloc[0]
        t = grp["tiempo"].values
        ocp = grp["ocupados"].values
        fn = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        util = round(fn(ocp, t) / (cap * (t[-1] - t[0]) * max(1e-9, 1)) * 100, 2) if len(t) > 1 and t[-1] > t[0] else 0
        filas.append({
            "Recurso": rec,
            "Utilización_%": util,
            "Cola_Prom": round(grp["cola"].mean(), 3),
            "Cola_Max": int(grp["cola"].max()),
            "Capacidad": int(cap),
            "Cuello_Botella": util >= 80 or grp["cola"].mean() > 0.5
        })
    return pd.DataFrame(filas).sort_values("Utilización_%", ascending=False).reset_index(drop=True)

def calc_kpis(df_l, plan):
    if df_l.empty:
        return pd.DataFrame()

    dur = (df_l["t_fin"].max() - df_l["t_creacion"].min()) / 60
    rows = []
    for p in PRODUCTOS:
        sub = df_l[df_l["producto"] == p]
        if sub.empty:
            continue
        und = sub["tamano"].sum()
        pu = plan.get(p, 0)
        tp = round(und / max(dur, 0.01), 3)
        lt = round(sub["tiempo_sistema"].mean(), 3)
        rows.append({
            "Producto": p,
            "Und_Prod": und,
            "Plan": pu,
            "Throughput_und_h": tp,
            "CycleTime_min_und": round((sub["tiempo_sistema"] / sub["tamano"]).mean(), 3),
            "LeadTime_min_lote": lt,
            "WIP_Prom": round(tp * (lt / 60), 2),
            "TaktTime_min_lote": round((44*4*60) / max((sum(DEM_HISTORICA[p]) / 12) / TAMANO_LOTE_BASE[p], 1), 2),
            "Cumplimiento_%": round(min(und / max(pu, 1) * 100, 100), 2),
        })
    return pd.DataFrame(rows)
