#!/usr/bin/env python
"""
Engine Health Monitoring – Anomaly Detection on Gas‑Turbine Data
---------------------------------------------------------------
* Reads   data/gas_turbine.csv
* Cleans  NaNs / duplicates
* Renames columns if needed (e.g., 'NOx (mg/m3)' → 'NOx')
* Creates Z‑score flags on key sensors
* Trains  Isolation Forest
* Fits    regression (TEY expectation) → residual flags
* Exports output/anomaly_report.csv and two diagnostic plots
"""

# ──────────────────────────────────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────────────────────────────────
import pathlib, numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression

# ──────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────
ROOT   = pathlib.Path(__file__).resolve().parent.parent
DATA   = ROOT / "data" / "gas_turbine.csv"
OUTDIR = ROOT / "output"
OUTDIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────
# Load & initial clean
# ──────────────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA)
df = df.drop_duplicates().reset_index(drop=True)
df = df.ffill()                     # forward‑fill NaNs (new syntax)

# Optional: rename columns to standard names
rename_map = {
    "NOx (mg/m3)": "NOx",
    "NOX": "NOx",
    "TIT (°C)": "TIT",
    "TAT (°C)": "TAT",
}
df = df.rename(columns=rename_map)

# ──────────────────────────────────────────────────────────────────────────
# Choose sensor columns that actually exist
# ──────────────────────────────────────────────────────────────────────────
candidate_cols = ["TIT", "TAT", "TEY", "CDP", "CO", "NOx"]
key_cols = [c for c in candidate_cols if c in df.columns]
if not key_cols:
    raise ValueError(
        "❌ None of the expected sensor columns found.\n"
        f"Available columns are: {list(df.columns)}"
    )

# ──────────────────────────────────────────────────────────────────────────
# Z‑score anomaly flag
# ──────────────────────────────────────────────────────────────────────────
z_thresh = 3.0
z_scores = df[key_cols].apply(lambda x: (x - x.mean()) / x.std())
df["z_flag"] = (np.abs(z_scores) > z_thresh).any(axis=1).astype(int)

# ──────────────────────────────────────────────────────────────────────────
# Isolation Forest anomaly flag
# ──────────────────────────────────────────────────────────────────────────
iso = IsolationForest(
    contamination=0.02, random_state=42, n_estimators=200
)
iso.fit(StandardScaler().fit_transform(df[key_cols]))
df["iso_flag"] = (iso.predict(df[key_cols]) == -1).astype(int)

# ──────────────────────────────────────────────────────────────────────────
# Regression residual anomaly flag (TEY expectation model)
# ──────────────────────────────────────────────────────────────────────────
reg_inputs = [c for c in ["AT", "AP", "RH", "CDP"] if c in df.columns]
if len(reg_inputs) < 2 or "TEY" not in df.columns:
    raise ValueError("Need TEY plus at least two of AT, AP, RH, CDP for "
                     "the residual model. Check your CSV columns.")

X = df[reg_inputs]
y = df["TEY"]

reg = LinearRegression().fit(X, y)
df["TEY_pred"] = reg.predict(X)
residual_std   = df["TEY"] - df["TEY_pred"]
three_sigma    = 3 * residual_std.std()
df["residual"] = residual_std
df["res_flag"] = (np.abs(df["residual"]) > three_sigma).astype(int)

# ──────────────────────────────────────────────────────────────────────────
# Combined anomaly flag
# ──────────────────────────────────────────────────────────────────────────
df["anomaly"] = (
    df["z_flag"] | df["iso_flag"] | df["res_flag"]
).astype(int)

# ──────────────────────────────────────────────────────────────────────────
# Save CSV report
# ──────────────────────────────────────────────────────────────────────────
report_cols = (
    ["anomaly", "z_flag", "iso_flag", "res_flag"]
    + key_cols
    + ["TEY_pred", "residual"]
)
df[report_cols].to_csv(OUTDIR / "anomaly_report.csv", index=False)
print(f"✅  anomaly_report.csv saved to {OUTDIR}")

# ──────────────────────────────────────────────────────────────────────────
# Quick visualisations
# ──────────────────────────────────────────────────────────────────────────
# 1) TIT over time with anomalies
if "TIT" in df.columns:
    plt.figure(figsize=(10, 4))
    sns.lineplot(x=df.index, y="TIT", data=df, label="TIT (°C)")
    sns.scatterplot(
        x=df[df.anomaly == 1].index,
        y="TIT",
        data=df[df.anomaly == 1],
        color="red",
        s=20,
        label="Anomaly",
    )
    plt.title("Turbine Inlet Temperature with Anomalies")
    plt.tight_layout()
    plt.savefig(OUTDIR / "TIT_anomalies.png", dpi=120)
    plt.close()

# 2) Residual distribution
plt.figure(figsize=(8, 4))
sns.histplot(df["residual"], bins=60, kde=True)
plt.axvline(three_sigma,  color="red", linestyle="--")
plt.axvline(-three_sigma, color="red", linestyle="--")
plt.title("Residual Distribution: TEY – Predicted TEY")
plt.tight_layout()
plt.savefig(OUTDIR / "TEY_residuals.png", dpi=120)
plt.close()

print(f"Plots saved in {OUTDIR}")
