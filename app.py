# app.py
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="SKU Forecast Dashboard", layout="wide")

# --- data loader: expects repo_root/data/dashboard.csv ---
APP_DIR   = Path(__file__).resolve().parent
# If app.py is under a subfolder (e.g., repo_root/app/app.py), keep .parent:
REPO_ROOT = APP_DIR.parent
# If app.py is at repo root, use: REPO_ROOT = APP_DIR

CANDIDATES = [
    REPO_ROOT / "data" / "dashboard.csv",
    Path.cwd() / "data" / "dashboard.csv",
]
DATA_PATH = next((p for p in CANDIDATES if p.exists()), None)
if DATA_PATH is None:
    st.error("❌ data/dashboard.csv not found at repo_root/data/dashboard.csv.")
    st.stop()

@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "ds" in d.columns:
        d["ds"] = pd.to_datetime(d["ds"], errors="coerce")
    for c in ["y", "y_hat", "yhat_lower", "yhat_upper"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d

# Load once; no uploader needed
df0 = _coerce(load_data(DATA_PATH)).dropna(subset=["ds"])

# -------- Global date range (sidebar) --------
st.sidebar.header("Date Range")

_ds_min = pd.to_datetime(df0["ds"].min())
_ds_max = pd.to_datetime(df0["ds"].max())

date_lo, date_hi = st.sidebar.slider(
    "Filter data between",
    min_value=_ds_min.to_pydatetime(),
    max_value=_ds_max.to_pydatetime(),
    value=(_ds_min.to_pydatetime(), _ds_max.to_pydatetime()),
    format="YYYY-MM-DD",
)

# Apply date mask globally
df0 = df0[(df0["ds"] >= pd.to_datetime(date_lo)) & (df0["ds"] <= pd.to_datetime(date_hi))].copy()

# --- Header: logo + intro + attribution ---
LOGO_CANDIDATES = [
    REPO_ROOT / "bhc-logo.png",
    REPO_ROOT / "data" / "bhc-logo.png",
    REPO_ROOT / "bhc-logo.webp",
    REPO_ROOT / "data" / "bhc-logo.webp",
    Path.cwd() / "bhc-logo.png",
    Path.cwd() / "data" / "bhc-logo.png",
]
LOGO_PATH = next((p for p in LOGO_CANDIDATES if p.exists()), None)

# feature-safe image call
def show_logo(path: str):
    try:
        st.image(path, use_container_width=True)   # new API
    except TypeError:
        st.image(path, use_column_width=True)      # old API fallback

col_logo, col_text = st.columns([1, 5], vertical_alignment="center")

with col_logo:
    if LOGO_PATH is not None:
        show_logo(str(LOGO_PATH))
    else:
        st.write("🧵")

with col_text:
    st.markdown("### Bee Happy Forecast Dashboard")
    st.markdown(
        "Track weekly actuals (`y`) and RandomForest forecasts (`y_hat`) with confidence bands "
        "(`yhat_lower`, `yhat_upper`). Use filters to pick a SKU (Item × Variation), adjust the "
        "**actuals context** and **forecast horizon**, and switch to **monthly** aggregation when needed."
    )
    st.caption("This demo is made by **Capstone Team 6** of MS in Data Science Part-Time B of the **Asian Institute of Management** for **Bee Happy Crafts**.")
st.divider()


# ---------------- Sidebar: windows & filters ----------------
st.sidebar.header("Data Window")
actual_weeks   = st.sidebar.number_input("Actuals window (weeks BEFORE forecast start)", 1, 260, 65)
forecast_weeks = st.sidebar.number_input("Forecast window length (weeks FROM forecast start)", 1, 52, 12)

st.sidebar.header("Filters")

# Keep only RF preds if model column exists
df_preds_all = df0[df0["model"].astype(str) == "RandomForest"] if "model" in df0.columns else df0.copy()

# Cluster filter
cluster_vals = sorted(df0["cluster_km8"].dropna().unique().tolist()) if "cluster_km8" in df0.columns else []
cluster_choice = st.sidebar.selectbox("Cluster", ["All"] + cluster_vals if cluster_vals else ["All"])

# Only SKUs with predictions
only_with_preds = st.sidebar.checkbox("Only SKUs with predictions", value=True)

# Date granularity
granularity = st.sidebar.radio("Date granularity", ["Weekly", "Monthly"], index=0)

# Show/Hide CI
show_ci = st.sidebar.checkbox("Show 95% confidence intervals", value=True)

# Search box (case-insensitive; matches in item_name or variation_name)
search_txt = st.sidebar.text_input("Search item/variation", value="").strip().lower()

# Demand filter: choose metric & range
st.sidebar.markdown("**Demand filter (on actuals)**")
metric_choice = st.sidebar.selectbox("Metric", ["Total demand (sum y)", "Average weekly demand (mean y)"], index=0)

# Build base filter DF (apply cluster & prediction availability first)
df_filter = df0.copy()
if cluster_choice != "All" and "cluster_km8" in df_filter.columns:
    df_filter = df_filter[df_filter["cluster_km8"] == cluster_choice]

if only_with_preds:
    have_pred_uids = set(df_preds_all["unique_id"].unique())
    df_filter = df_filter[df_filter["unique_id"].isin(have_pred_uids)]

# Apply search filter
if search_txt:
    _txt = (
        df_filter.get("item_name", "").astype(str).str.lower().fillna("")
        + " "
        + df_filter.get("variation_name", "").astype(str).str.lower().fillna("")
    )
    df_filter = df_filter[_txt.str.contains(search_txt, na=False)]

# Demand metric (based on actual y)
agg = df0.dropna(subset=["y"])[["unique_id", "y"]].groupby("unique_id").agg(
    total=("y", "sum"),
    mean=("y", "mean")
).reset_index()

metric_col = "total" if metric_choice.startswith("Total") else "mean"
lo, hi = float(agg[metric_col].min()), float(agg[metric_col].max())
sel_lo, sel_hi = st.sidebar.slider(
    "Demand range",
    min_value=float(np.floor(lo)),
    max_value=float(np.ceil(hi) if hi > 0 else 1.0),
    value=(float(np.floor(lo)), float(np.ceil(hi) if hi > 0 else 1.0)),
    step=1.0
)
allowed_uids = set(agg[(agg[metric_col] >= sel_lo) & (agg[metric_col] <= sel_hi)]["unique_id"])
df_filter = df_filter[df_filter["unique_id"].isin(allowed_uids)]

# Build SKU selector
def _label(row):
    item = str(row.get("item_name", "") or "")
    var  = str(row.get("variation_name", "") or "")
    uid  = str(row.get("unique_id", "") or "")
    left = item if item else "(no item_name)"
    right = var if var else "(no variation)"
    return f"{left} — {right}  |  {uid}"

names = (
    df_filter.drop_duplicates(subset=["unique_id"])
             .loc[:, ["unique_id", "item_name", "variation_name"]]
             .assign(select_label=lambda x: x.apply(_label, axis=1))
             .sort_values("select_label")
)

if names.empty:
    st.warning("No SKUs match the current filters. Adjust filters/search/range.")
    st.stop()

label = st.sidebar.selectbox("Select SKU", names["select_label"].tolist())
sel_uid = names.loc[names["select_label"] == label, "unique_id"].iloc[0]

# ---------------- Window selection (anchor at forecast START) ----------------
sku_all = df0[df0["unique_id"] == sel_uid].copy()
sku_preds_all = df_preds_all[df_preds_all["unique_id"] == sel_uid].copy().sort_values("ds")

if sku_preds_all.empty:
    st.warning("No predictions found for this SKU. Showing recent actuals only.")
    last_actual = sku_all["ds"].max()
    actual_start = last_actual - pd.to_timedelta(actual_weeks - 1, unit="W")
    actuals_w = sku_all[sku_all["ds"] >= actual_start].copy()
    preds_w = sku_preds_all.copy()
else:
    forecast_start = sku_preds_all["ds"].min()
    forecast_end   = forecast_start + pd.to_timedelta(max(int(forecast_weeks), 1) - 1, unit="W")
    forecast_end   = min(forecast_end, sku_preds_all["ds"].max())

    preds_w = sku_preds_all[(sku_preds_all["ds"] >= forecast_start) &
                            (sku_preds_all["ds"] <= forecast_end)].copy()

    actual_end   = forecast_start - pd.to_timedelta(1, unit="W")
    actual_start = actual_end - pd.to_timedelta(max(int(actual_weeks), 1) - 1, unit="W")
    actuals_w = sku_all[(sku_all["ds"] >= actual_start) &
                        (sku_all["ds"] <= actual_end)].copy()

# ---------------- Monthly aggregation (project CI) ----------------
Z = 1.96  # ~95%

def monthly_from_weekly_preds(df_pred_weekly: pd.DataFrame) -> pd.DataFrame:
    if df_pred_weekly.empty:
        return df_pred_weekly.copy()
    tmp = df_pred_weekly.copy()
    tmp["month"] = tmp["ds"].dt.to_period("M").dt.to_timestamp("M")
    half = (tmp["yhat_upper"] - tmp["yhat_lower"]) / 2.0
    var  = (half / Z) ** 2
    tmp["_var"] = var.clip(lower=0)

    g = tmp.groupby("month", as_index=False).agg(
        y_hat=("y_hat", "sum"),
        _var=("_var", "sum"),
    )
    g["_half"] = Z * np.sqrt(g["_var"])
    g["yhat_lower"] = np.clip(g["y_hat"] - g["_half"], 0, None)
    g["yhat_upper"] = g["y_hat"] + g["_half"]
    return g.rename(columns={"month": "ds"})[["ds", "y_hat", "yhat_lower", "yhat_upper"]]

def monthly_from_weekly_actuals(df_actual_weekly: pd.DataFrame) -> pd.DataFrame:
    if df_actual_weekly.empty:
        return df_actual_weekly.copy()
    tmp = df_actual_weekly.copy()
    tmp["month"] = tmp["ds"].dt.to_period("M").dt.to_timestamp("M")
    g = tmp.groupby("month", as_index=False).agg(y=("y", "sum"))
    return g.rename(columns={"month": "ds"})

def compute_totals(actuals_df: pd.DataFrame, preds_df: pd.DataFrame, Z: float = 1.96):
    """Returns (hist_total, fcst_total, fcst_lo, fcst_hi) for the displayed window."""
    hist_total = float(actuals_df["y"].sum()) if not actuals_df.empty else None

    if preds_df.empty:
        return hist_total, None, None, None

    fcst_total = float(preds_df["y_hat"].sum())

    # Combine CI correctly by summing variances implied by half-widths
    fcst_lo = fcst_hi = None
    if {"yhat_lower", "yhat_upper"}.issubset(preds_df.columns):
        half = (preds_df["yhat_upper"] - preds_df["yhat_lower"]) / 2.0
        var = (half / Z) ** 2
        half_total = float(Z * np.sqrt(var.sum()))
        fcst_lo = max(fcst_total - half_total, 0.0)
        fcst_hi = fcst_total + half_total

    return hist_total, fcst_total, fcst_lo, fcst_hi


# Choose granularity
if granularity == "Weekly":
    actuals = actuals_w[["ds", "y"]].sort_values("ds")
    preds   = preds_w[["ds", "y_hat", "yhat_lower", "yhat_upper"]].sort_values("ds")
else:
    actuals = monthly_from_weekly_actuals(actuals_w)
    preds   = monthly_from_weekly_preds(preds_w)

# ---------------- Plot ----------------
st.header(f"Actual vs Forecast — {granularity}")

fig, ax = plt.subplots(figsize=(10, 5))

# Actuals (black)
if not actuals.empty:
    ax.plot(actuals["ds"], actuals["y"], color="black", linewidth=2.0, label="Actual")

# Forecast (blue dashed) + optional CI
if not preds.empty:
    ax.plot(preds["ds"], preds["y_hat"], color="tab:blue", linewidth=2.0, linestyle="--", label="RandomForest")
    if show_ci:
        x  = preds["ds"].to_numpy()
        lo = np.clip(preds["yhat_lower"].to_numpy(dtype=float), 0, None)
        hi = preds["yhat_upper"].to_numpy(dtype=float)
        ax.fill_between(x, lo, hi, color="tab:blue", alpha=0.20, label="95% CI")

# Linear-axis cosmetics only
ax.axhline(0, color="black", linewidth=0.8)
ax.spines["bottom"].set_position(("data", 0))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(alpha=0.3)
ax.tick_params(axis="x", rotation=45)
ax.set_xlabel("")
ax.set_ylabel("y")

item_nm = str((sku_all["item_name"].dropna().iloc[0]) if "item_name" in sku_all.columns and not sku_all["item_name"].dropna().empty else "(no item_name)")
ax.set_title(f"{sel_uid}\n{textwrap.fill(item_nm, width=60)}", fontweight="bold")

handles = [plt.Line2D([0],[0], color="black", lw=2, label="Actual"),
           plt.Line2D([0],[0], color="tab:blue", lw=2, linestyle="--", label="RandomForest")]
if show_ci:
    handles.append(plt.Rectangle((0,0),1,1, color="tab:blue", alpha=0.20, label="95% CI"))
ax.legend(handles=handles, loc="upper left", frameon=True)

fig.set_dpi(100)
st.pyplot(fig, clear_figure=True)

# --- Totals under the plot ---
hist_total, fcst_total, fcst_lo, fcst_hi = compute_totals(actuals, preds, Z=Z)

st.markdown("#### Totals for Selected Window")
c1, c2, c3 = st.columns([1, 1, 1])

with c1:
    if hist_total is not None:
        st.metric("Total Historical Orders", f"{hist_total:,.0f}")
    else:
        st.metric("Total Historical Orders", "—")

with c2:
    if fcst_total is not None:
        st.metric("Total Forecasted Orders", f"{fcst_total:,.0f}")
    else:
        st.metric("Total Forecasted Orders", "—")

with c3:
    if show_ci and fcst_total is not None and fcst_lo is not None:
        st.write(f"**95% CI (Forecast Total):** {fcst_lo:,.0f} – {fcst_hi:,.0f}")
    else:
        st.write(" ")

# Peek
with st.expander("Show current data"):
    st.write("Actuals:", actuals.shape)
    st.dataframe(actuals.head(200))
    st.write("Forecast:", preds.shape)
    st.dataframe(preds.head(200))
