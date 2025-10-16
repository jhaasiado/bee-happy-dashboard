import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="SKU Forecast Dashboard", layout="wide")

# ------------ Sidebar: load data ------------
st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload long-format CSV", type=["csv"])
date_weeks_actual = st.sidebar.number_input("Actuals window (weeks)", 1, 260, 65)
date_weeks_fcst   = st.sidebar.number_input("Forecast window (weeks)", 1, 52, 12)

@st.cache_data
def _read_csv(f):
    df = pd.read_csv(f)
    return df

if uploaded is None:
    st.info("Upload a CSV with columns at least: ['unique_id','ds','y','y_hat','yhat_lower','yhat_upper','item_name','variation_name'] (optional: 'model').")
    st.stop()

df0 = _read_csv(uploaded).copy()

# ------------ Dtype hygiene ------------
def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "ds" in d.columns:
        d["ds"] = pd.to_datetime(d["ds"], errors="coerce")
    for c in ["y", "y_hat", "yhat_lower", "yhat_upper"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d

df = _coerce(df0)
df = df.dropna(subset=["ds"])
if "model" in df.columns:
    # keep only RandomForest rows for predictions (if provided)
    df_rf = df[df["model"].astype(str) == "RandomForest"].copy()
else:
    df_rf = df.copy()

# ------------ Build selector keys ------------
# Combine item_name + variation_name; fall back gracefully
def _combine_name(row):
    item = str(row.get("item_name", ""))
    var  = str(row.get("variation_name", ""))
    uid  = str(row.get("unique_id", ""))
    left = item if item and item != "nan" else "(no item_name)"
    right = var if var and var != "nan" else "(no variation)"
    return f"{left} — {right}  |  {uid}"

names = (
    df.drop_duplicates(subset=["unique_id"])
      .loc[:, ["unique_id", "item_name", "variation_name"]]
      .assign(select_label=lambda x: x.apply(_combine_name, axis=1))
      .sort_values("select_label")
)

st.sidebar.header("Filters")
label = st.sidebar.selectbox("Select SKU (by Item × Variation)", options=names["select_label"].tolist())
sel_uid = names.loc[names["select_label"] == label, "unique_id"].iloc[0]

# ------------ Window selection ------------
# Find last timestamp seen for either actuals or preds for that SKU
last_actual = df.loc[df["unique_id"] == sel_uid, "ds"].max()
last_pred   = df_rf.loc[df_rf["unique_id"] == sel_uid, "ds"].max()
last_ds = max(x for x in [last_actual, last_pred] if pd.notna(x))

actual_start = last_ds - pd.to_timedelta(date_weeks_actual + date_weeks_fcst - 1, unit="W")
pred_start   = last_ds - pd.to_timedelta(date_weeks_fcst - 1, unit="W")

actuals = df[(df["unique_id"] == sel_uid) & (df["ds"] >= actual_start)].copy()
preds   = df_rf[(df_rf["unique_id"] == sel_uid) & (df_rf["ds"] >= pred_start)].copy()

# ------------ Plot ------------
st.header("Actual vs Forecast")

fig, ax = plt.subplots(figsize=(10, 5))

# Actuals (black)
if not actuals.empty:
    ax.plot(actuals["ds"], actuals["y"], color="black", linewidth=2.0, label="Actual")

# Forecast + CI (blue dashed + band)
if not preds.empty:
    ax.plot(preds["ds"], preds["y_hat"], color="tab:blue", linewidth=2.0, linestyle="--", label="RandomForest")

    x  = preds["ds"].to_numpy()
    lo = np.clip(preds["yhat_lower"].astype(float).to_numpy(), 0, None)  # clip to zero
    hi = preds["yhat_upper"].astype(float).to_numpy()
    ax.fill_between(x, lo, hi, color="tab:blue", alpha=0.20, label="95% CI")

# Force y=0 baseline at x-axis
ax.axhline(0, color="black", linewidth=0.8)
ax.spines["bottom"].set_position(("data", 0))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.grid(alpha=0.3)
ax.tick_params(axis="x", rotation=45)
ax.set_xlabel("")
ax.set_ylabel("y")

# Title uses wrapped item name
item_nm = str(actuals["item_name"].dropna().iloc[0]) if "item_name" in actuals.columns and not actuals["item_name"].dropna().empty else "(no item_name)"
title = f"{sel_uid}\n{textwrap.fill(item_nm, width=60)}"
ax.set_title(title, fontweight="bold")

# Legend
handles = [
    plt.Line2D([0],[0], color="black", lw=2, label="Actual"),
    plt.Line2D([0],[0], color="tab:blue", lw=2, linestyle="--", label="RandomForest"),
    plt.Rectangle((0,0),1,1, color="tab:blue", alpha=0.20, label="95% CI"),
]
ax.legend(handles=handles, loc="upper left", frameon=True)

st.pyplot(fig)

# ------------ Optional: details ------------
with st.expander("Show sample data (current SKU)"):
    st.write("Actuals (window):", actuals.shape)
    st.dataframe(actuals.sort_values("ds"))
    st.write("Forecast (window):", preds.shape)
    st.dataframe(preds.sort_values("ds"))

st.caption("Tip: Adjust the weeks in the sidebar to widen/narrow the context window.")
