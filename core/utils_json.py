# core/utils_json.py
import io
import pandas as pd

def _j(df):
    return df.to_json(orient="split") if not df.empty else "{}"

def _rj(j):
    if not j or j in ("{}", "null", "None"):
        return pd.DataFrame()
    try:
        return pd.read_json(io.StringIO(j), orient="split")
    except Exception:
        return pd.DataFrame()

def _j_dict(d):
    return {k: _j(v) for k, v in d.items()}

def _rj_dict(d):
    return {k: _rj(v) for k, v in d.items()}
