# app.py
# ==========================================================
# Crowdfunding Dashboard (Before vs After Cleaning)
# - Login + Demo Account button
# - Onboarding / Quick Guide after login
# - 4 Modules
# - Filters build once (shared)
# - Light Blue + White theme, dark-gray text
# - Google Drive CSV loader (robust)
# - No matplotlib (Plotly only)
# - Outlier section: Goal & Pledged (Before vs After) with Log Transformation + Median/IQR
# ==========================================================

import re
import io
import numpy as np
import pandas as pd
import requests
import plotly.express as px
import streamlit as st

# -----------------------------
# Page Config (MUST be first)
# -----------------------------
st.set_page_config(
    page_title="Crowdfunding Dashboard (Before vs After Cleaning)",
    page_icon="📊",
    layout="wide",
)

# -----------------------------
# Google Drive URLs
# -----------------------------
BEFORE_URL = "https://drive.google.com/file/d/1qRTrEuENBRdrx4aVzT7WwDg8qsCAEIlh/view?usp=sharing"
AFTER_URL  = "https://drive.google.com/file/d/15gI9_y2FWKLwvxTvfpjy39sMtuf7bs-i/view?usp=sharing"


# -----------------------------
# Theme / CSS (Light mode + readable dropdown)
# -----------------------------
st.markdown(
    """
<style>
/* App background + base text */
.stApp { background: #f4f9ff; }
html, body, [class*="css"], p, span, label, small, div { color: #0f172a !important; }

/* Titles */
h1, h2, h3, h4 { color: #0b3d91 !important; }

/* Sidebar */
section[data-testid="stSidebar"] {
  background: #e6f2ff !important;
  border-right: 1px solid rgba(15,23,42,0.10);
}
section[data-testid="stSidebar"] * { color: #0f172a !important; }

/* Cards */
.card {
  background: #ffffff;
  border: 1px solid rgba(15, 23, 42, 0.12);
  border-radius: 14px;
  padding: 14px 14px;
  box-shadow: 0 8px 18px rgba(2, 8, 23, 0.06);
}
.mini { font-size: 12px; opacity: 0.8; }

/* Buttons */
.stButton > button {
  background: #1d4ed8 !important;
  color: #ffffff !important;
  border: 0 !important;
  border-radius: 12px !important;
  padding: 0.55rem 0.9rem !important;
  font-weight: 600 !important;
}
.stButton > button:hover { filter: brightness(1.05); }

/* Inputs: make dropdown/selected text readable */
div[data-baseweb="select"] * { color: #0f172a !important; }
div[data-baseweb="select"] input { color: #0f172a !important; }
div[data-baseweb="popover"] * { color: #0f172a !important; }
div[role="listbox"] * { color: #0f172a !important; }

/* Tabs */
button[data-baseweb="tab"] * { color: #0f172a !important; font-weight: 600; }

/* Dataframe header */
thead tr th { color: #0f172a !important; }
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------
# Helpers
# -----------------------------
def median_iqr(s: pd.Series):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) == 0:
        return np.nan, np.nan, np.nan
    q1 = s.quantile(0.25)
    med = s.quantile(0.50)
    q3 = s.quantile(0.75)
    return med, q1, q3

def money_short(x):
    try:
        if pd.isna(x):
            return "-"
        x = float(x)
        if abs(x) >= 1_000_000_000:
            return f"{x/1_000_000_000:.2f}B"
        if abs(x) >= 1_000_000:
            return f"{x/1_000_000:.2f}M"
        if abs(x) >= 1_000:
            return f"{x/1_000:.2f}K"
        return f"{x:.0f}"
    except Exception:
        return "-"

def drive_id(url: str) -> str:
    m = re.search(r"/d/([^/]+)", url)
    return m.group(1) if m else url

def _get_confirm_token(resp: requests.Response) -> str | None:
    for k, v in resp.cookies.items():
        if k.startswith("download_warning"):
            return v
    return None

@st.cache_data(show_spinner=False)
def load_drive_csv(url: str) -> pd.DataFrame:
    """
    โหลด CSV จาก Google Drive (Drive) แบบรองรับไฟล์ใหญ่:
    - ดึง confirm token ถ้ามี
    - อ่านผ่าน BytesIO แล้วค่อย pd.read_csv
    """
    fid = drive_id(url)
    session = requests.Session()

    base = "https://drive.google.com/uc?export=download"
    resp = session.get(base, params={"id": fid}, stream=True, timeout=60)

    token = _get_confirm_token(resp)
    if token:
        resp = session.get(base, params={"id": fid, "confirm": token}, stream=True, timeout=60)

    resp.raise_for_status()
    content = resp.content
    return pd.read_csv(io.BytesIO(content))

def safe_to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")

def add_working_time_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    เพิ่มคอลัมน์ใช้งานชั่วคราว (temporary) สำหรับฟิลเตอร์/EDA:
    - Launched_dt, Deadline_dt, DurationDays
    NOTE: ไม่ได้แปลว่าไฟล์หลังคลีนต้องมีคอลัมน์พวกนี้
    """
    out = df.copy()
    if "Launched" in out.columns:
        out["Launched_dt"] = safe_to_datetime(out["Launched"])
    else:
        out["Launched_dt"] = pd.NaT

    if "Deadline" in out.columns:
        out["Deadline_dt"] = safe_to_datetime(out["Deadline"])
    else:
        out["Deadline_dt"] = pd.NaT

    if out["Launched_dt"].notna().any() and out["Deadline_dt"].notna().any():
        out["DurationDays"] = (out["Deadline_dt"] - out["Launched_dt"]).dt.days
    else:
        out["DurationDays"] = np.nan

    return out

def drop_temp_cols(df: pd.DataFrame) -> pd.DataFrame:
    temp = ["Launched_dt", "Deadline_dt", "DurationDays"]
    return df.drop(columns=[c for c in temp if c in df.columns], errors="ignore")

def kpi_block(title: str, value: str, note: str = ""):
    st.markdown(
        f"""
<div class="card">
  <div class="mini">{title}</div>
  <div style="font-size: 26px; font-weight: 800; margin-top: 4px;">{value}</div>
  <div class="mini" style="margin-top: 6px;">{note}</div>
</div>
""",
        unsafe_allow_html=True,
    )

def pct(x: float) -> str:
    return f"{x*100:.2f}%"

def money_short(x) -> str:
    """
    แสดงตัวเลขแบบย่อ (abbrev) เช่น 5,000 -> 5.0K, 50,000,000 -> 50.0M
    """
    try:
        if pd.isna(x):
            return "-"
        x = float(x)
        ax = abs(x)
        if ax >= 1e9:
            return f"{x/1e9:.2f}B"
        if ax >= 1e6:
            return f"{x/1e6:.2f}M"
        if ax >= 1e3:
            return f"{x/1e3:.2f}K"
        return f"{x:.0f}"
    except Exception:
        return "-"

def show_plot(fig, key: str):
    st.plotly_chart(fig, use_container_width=True, key=key)

def build_filters(df_work: pd.DataFrame, key_prefix: str = "main") -> pd.DataFrame:
    """
    ฟิลเตอร์ (Filters) ด้านซ้าย สร้างครั้งเดียว:
    - Search Name
    - Multiselect: State, Category, Subcategory, Country
    - Date range: Launched_dt
    - Sliders: Goal, Pledged, Backers, DurationDays
    """
    st.sidebar.markdown("## ตัวกรอง (Filters)")
    df2 = df_work.copy()

    # Search
    q = st.sidebar.text_input(
        "ค้นหาชื่อโครงการ (Search Name)",
        value="",
        key=f"{key_prefix}_search_name",
    )

    # Categorical filters
    cat_cols = ["State", "Category", "Subcategory", "Country"]
    selected = {}
    for col in cat_cols:
        if col in df2.columns:
            options = sorted(df2[col].dropna().astype(str).unique().tolist())
            selected[col] = st.sidebar.multiselect(
                f"เลือก {col}",
                options=options,
                default=options,
                key=f"{key_prefix}_{col}_ms",
            )

    # Date range (Launched)
    launched_range = None
    if "Launched_dt" in df2.columns and df2["Launched_dt"].notna().any():
        min_d = df2["Launched_dt"].min().date()
        max_d = df2["Launched_dt"].max().date()
        launched_range = st.sidebar.date_input(
            "ช่วงวันเริ่มระดมทุน (Launched range)",
            value=(min_d, max_d),
            key=f"{key_prefix}_launched_range",
        )

    # Numeric sliders
    num_cols = ["Goal", "Pledged", "Backers", "DurationDays"]
    ranges = {}
    for col in num_cols:
        if col in df2.columns and pd.api.types.is_numeric_dtype(df2[col]):
            valid = df2[col].dropna()
            if len(valid) > 0:
                vmin = float(valid.min())
                vmax = float(valid.max())
                # กันเคส vmin == vmax
                if vmin == vmax:
                    vmin = vmin - 1.0
                    vmax = vmax + 1.0
                ranges[col] = st.sidebar.slider(
                    f"ช่วง {col}",
                    min_value=vmin,
                    max_value=vmax,
                    value=(vmin, vmax),
                    key=f"{key_prefix}_{col}_slider",
                )

    # Apply filters
    mask = pd.Series(True, index=df2.index)

    if q.strip() and "Name" in df2.columns:
        mask &= df2["Name"].astype(str).str.contains(q.strip(), case=False, na=False)

    for col, vals in selected.items():
        if vals:
            mask &= df2[col].astype(str).isin(vals)

    if launched_range and "Launched_dt" in df2.columns:
        start, end = launched_range
        mask &= df2["Launched_dt"].dt.date.between(start, end)

    for col, (lo, hi) in ranges.items():
        mask &= df2[col].between(lo, hi)

    out = df2.loc[mask].copy()
    st.sidebar.markdown("---")
    st.sidebar.caption(f"ผลลัพธ์หลังกรอง: {len(out):,} แถว")
    return out

def median_iqr(series: pd.Series) -> tuple[float, float, float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return np.nan, np.nan, np.nan
    q1 = s.quantile(0.25)
    med = s.quantile(0.50)
    q3 = s.quantile(0.75)
    return med, q1, q3


# -----------------------------
# Authentication (simple)
# -----------------------------
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "show_guide" not in st.session_state:
    st.session_state["show_guide"] = True

def show_onboarding():
    st.markdown("### คู่มือใช้งานเบื้องต้น (Quick Guide)")
    st.info(
        """
**เส้นทางการใช้งาน (Overview → Detail)**
1) ไปที่ **ภาพรวม (Overview)** เพื่อดู KPI และสัดส่วนความสำเร็จ  
2) ใช้ **ตัวกรอง (Filters)** ด้านซ้าย: Category / Country / State / วันที่ / ช่วง Goal-Pledged-Backers  
3) ไปที่ **คุณภาพข้อมูล & ขั้นตอนทำความสะอาด** เพื่อเห็น Before vs After  
4) ไปที่ **EDA & ความสัมพันธ์** เพื่อดูการกระจาย + ความสัมพันธ์ (Correlation)  
5) ไปที่ **Insights** เพื่อดู What–Why–So What พร้อมกราฟยืนยัน
        """
    )
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("เข้าใจแล้ว (ไม่ต้องแสดงอีก)", key="hide_guide"):
            st.session_state["show_guide"] = False
            st.rerun()
    with c2:
        st.caption("หากอยากดูอีกครั้ง เปิดได้จากปุ่ม “เปิดคู่มือ” บนหน้าเว็บ")

def login_view():
    st.markdown("## 🔐 เข้าสู่ระบบ (Login)")
    st.caption("มีปุ่มบัญชีทดลองสำหรับกรรมการ/ผู้ทดสอบ กดครั้งเดียวเข้าใช้งานได้ทันที")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    u = st.text_input("Username", value="", key="login_username")
    p = st.text_input("Password", value="", type="password", key="login_password")

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("Login", key="btn_login"):
            if (u == "admin" and p == "admin123") or (u == "judge" and p == "judge123"):
                st.session_state["logged_in"] = True
                st.success("เข้าสู่ระบบสำเร็จ")
                st.rerun()
            else:
                st.error("ชื่อผู้ใช้/รหัสผ่านไม่ถูกต้อง (ลองใช้บัญชีทดลอง)")
    with col2:
        if st.button("บัญชีทดลอง (Demo)", key="btn_demo"):
            st.session_state["logged_in"] = True
            st.success("เข้าสู่ระบบด้วยบัญชีทดลองแล้ว")
            st.rerun()
    with col3:
        st.caption("บัญชีตัวอย่าง: admin/admin123 หรือ judge/judge123 (หรือกด Demo)")
    st.markdown("</div>", unsafe_allow_html=True)

if not st.session_state["logged_in"]:
    login_view()
    st.stop()

# -----------------------------
# Header
# -----------------------------
top = st.container()
with top:
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown("# 📊 Crowdfunding Dashboard")
        st.caption("Before vs After — ใช้ไฟล์จาก Google Drive (Drive) + ฟิลเตอร์ + EDA + Insight")
    with c2:
        if st.button("ออกจากระบบ (Logout)", key="btn_logout"):
            st.session_state["logged_in"] = False
            st.session_state["show_guide"] = True
            st.rerun()

# Onboarding
if st.session_state.get("show_guide", True):
    with st.expander("👋 หน้าต่างสอนใช้เบื้องต้น (กดเพื่อดู/ซ่อน)", expanded=True):
        show_onboarding()
else:
    if st.button("📘 เปิดคู่มือ (Quick Guide)", key="btn_open_guide"):
        st.session_state["show_guide"] = True
        st.rerun()

# -----------------------------
# Load data from Google Drive
# -----------------------------
with st.spinner("กำลังโหลดข้อมูลจาก Google Drive (Drive)..."):
    df_before_raw = load_drive_csv(BEFORE_URL)
    df_after_raw  = load_drive_csv(AFTER_URL)

# Working copy for filters/EDA (temporary time cols)
df_before_work = add_working_time_cols(df_before_raw)
df_after_work  = add_working_time_cols(df_after_raw)

# ✅ Build filters once (shared)
filtered_df = build_filters(df_after_work, key_prefix="main")

# -----------------------------
# Tabs (4 Modules)
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "1) ภาพรวม (Overview)",
        "2) คุณภาพข้อมูล & ขั้นตอนทำความสะอาด",
        "3) EDA & ความสัมพันธ์",
        "4) Insights (What–Why–So What)",
    ]
)

# ==========================================================
# TAB 1: Overview
# ==========================================================
with tab1:
    st.markdown("## ภาพรวมข้อมูลหลังทำความสะอาด (After Cleaning)")
    dff = filtered_df.copy()
    total = len(dff)

    # KPI
    state_norm = dff["State"].astype(str).str.strip().str.lower() if "State" in dff.columns else pd.Series([], dtype=str)
    success_rate = (state_norm == "successful").mean() if total > 0 and len(state_norm) else 0.0

    med_goal = dff["Goal"].median() if "Goal" in dff.columns and total > 0 else np.nan
    med_pledged = dff["Pledged"].median() if "Pledged" in dff.columns and total > 0 else np.nan
    med_backers = dff["Backers"].median() if "Backers" in dff.columns and total > 0 else np.nan

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1: kpi_block("จำนวนโครงการ (Projects)", f"{total:,}", "หลังกรองด้วย Filters")
    with k2: kpi_block("อัตราสำเร็จ (Success)", pct(success_rate), "State = successful")
    with k3: kpi_block("Median Goal", money_short(med_goal), "ค่ากลางเป้าหมาย (Median)")
    with k4: kpi_block("Median Pledged", money_short(med_pledged), "ค่ากลางเงินที่ได้จริง (Median)")
    with k5: kpi_block("Median Backers", money_short(med_backers), "ค่ากลางผู้สนับสนุน (Median)")

    st.markdown("---")

    c1, c2 = st.columns([1.25, 1])
    with c1:
        st.markdown("### สัดส่วนสถานะโครงการ (State Share)")
        if "State" in dff.columns and total > 0:
            s = dff["State"].astype(str).str.strip().str.title().value_counts(dropna=False).reset_index()
            s.columns = ["State", "Count"]
            s["Percent"] = s["Count"] / s["Count"].sum() * 100
            fig = px.bar(s, x="State", y="Count", text=s["Percent"].map(lambda x: f"{x:.2f}%"))
            fig.update_traces(textposition="outside")
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
            show_plot(fig, "ov_state_share")
        else:
            st.warning("ไม่พบคอลัมน์ State หรือไม่มีข้อมูลหลังกรอง")

    with c2:
        st.markdown("### Top Category (ตามจำนวนโครงการ)")
        if "Category" in dff.columns and total > 0:
            top_cat = dff["Category"].astype(str).str.strip().value_counts().head(10).reset_index()
            top_cat.columns = ["Category", "Count"]
            fig = px.bar(top_cat, x="Count", y="Category", orientation="h", text="Count")
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
            show_plot(fig, "ov_top_category")
        else:
            st.warning("ไม่พบคอลัมน์ Category หรือไม่มีข้อมูลหลังกรอง")

    st.markdown("### ตารางข้อมูล (หลังกรอง)")
    st.dataframe(drop_temp_cols(dff).head(200), use_container_width=True)

    # Download filtered (remove temp cols)
    csv = drop_temp_cols(dff).to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ ดาวน์โหลดข้อมูลหลังกรอง (CSV)",
        data=csv,
        file_name="filtered_after_cleaning.csv",
        mime="text/csv",
        key="btn_download_filtered",
    )

# ==========================================================
# TAB 2: Data Quality & Cleaning Steps (+ Outlier Before/After)
# ==========================================================
with tab2:
    st.markdown("## Before vs After (คุณภาพข้อมูล)")
    st.caption("แสดงโครงสร้างข้อมูล + ขั้นตอนทำความสะอาด + การจัดการ Outlier (ไม่ลบ, ใช้ Log Transformation, ดู Median/IQR)")
    

    b1, b2 = st.columns(2)
    with b1:
        st.markdown("### ก่อนทำความสะอาด (Before)")
        st.write(f"Shape: **{df_before_raw.shape[0]:,} แถว × {df_before_raw.shape[1]:,} คอลัมน์**")
        st.dataframe(
            df_before_raw.dtypes.astype(str).reset_index().rename(columns={"index": "Feature", 0: "dtype"}),
            use_container_width=True,
        )

    with b2:
        st.markdown("### หลังทำความสะอาด (After)")
        st.write(f"Shape: **{df_after_raw.shape[0]:,} แถว × {df_after_raw.shape[1]:,} คอลัมน์**")
        st.dataframe(
            df_after_raw.dtypes.astype(str).reset_index().rename(columns={"index": "Feature", 0: "dtype"}),
            use_container_width=True,
        )
        st.markdown("---")
st.markdown("### Outlier (Before vs After) — ไม่ลบค่า ใช้ Log Transformation")

outlier_panel(df_before, "ก่อนทำความสะอาด (Before)", "out_before")
outlier_panel(df_after,  "หลังทำความสะอาด (After)",  "out_after")

    st.markdown("---")
    st.markdown("## ขั้นตอนการจัดการ Outlier (Outlier) — Goal และ Pledged")

    st.info(
        """
- **ไม่ลบค่าที่สูงผิดปกติออก (Do not remove extreme values)** เพราะอาจเป็น “โครงการขนาดใหญ่” ที่เกิดขึ้นจริง  
- ใช้ **การแปลงลอการิทึม (Log Transformation / log1p)** เพื่อลดความเบ้ (Skewness)  
- วิเคราะห์ร่วมกับ **ค่ากลาง (Median)** และ **ช่วงการกระจาย (IQR = Q3-Q1)** เพื่อสรุปภาพรวมได้แม่นกว่า mean
        """
    )

   def outlier_panel(df_src: pd.DataFrame, title_prefix: str, fig_key_prefix: str):
    st.markdown(f"#### {title_prefix} — Outlier (Outlier) ของ Goal และ Pledged")

    # Controls (กันค้าง)
    cA, cB, cC = st.columns([1, 1, 2])
    with cA:
        sample_n = st.number_input(
            "จำนวนตัวอย่างสำหรับ plot (sample size)",
            min_value=1000,
            max_value=20000,
            value=5000,
            step=1000,
            key=f"{fig_key_prefix}_sample_n",
        )
    with cB:
        show_points = st.toggle(
            "โชว์จุด outlier (points)",
            value=False,  # ปิดไว้ก่อนกันค้าง
            key=f"{fig_key_prefix}_show_points",
        )
    with cC:
        st.caption("แนะนำ: ปิด points และใช้ sample 3k–10k จะลื่นสุด")

    # sample
    plot_df = df_src
    if len(plot_df) > sample_n:
        plot_df = plot_df.sample(n=int(sample_n), random_state=42)

    points_mode = "outliers" if show_points else False

    c1, c2 = st.columns(2)

    # Goal
    with c1:
        st.markdown("**Goal**")
        if "Goal" in df_src.columns:
            med, q1, q3 = median_iqr(df_src["Goal"])
            iqr = (q3 - q1) if pd.notna(q3) and pd.notna(q1) else np.nan
            st.caption(f"Median={money_short(med)} | IQR={money_short(iqr)} (Q1={money_short(q1)}, Q3={money_short(q3)})")

            fig = px.box(plot_df, x="Goal", points=points_mode, title="Boxplot: Goal (Raw)")
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
            show_plot(fig, f"{fig_key_prefix}_goal_raw")

            # log1p
            goal_log_all = np.log1p(pd.to_numeric(df_src["Goal"], errors="coerce"))
            med2, q12, q32 = median_iqr(goal_log_all)
            iqr2 = (q32 - q12) if pd.notna(q32) and pd.notna(q12) else np.nan
            st.caption(f"log1p(Goal): Median={med2:.3f} | IQR={iqr2:.3f}")

            goal_log_plot = np.log1p(pd.to_numeric(plot_df["Goal"], errors="coerce"))
            fig2 = px.box(pd.DataFrame({"goal_log": goal_log_plot}), x="goal_log", points=points_mode, title="Boxplot: Goal (Log)")
            fig2.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
            show_plot(fig2, f"{fig_key_prefix}_goal_log")
        else:
            st.warning("ไม่พบคอลัมน์ Goal")

    # Pledged
    with c2:
        st.markdown("**Pledged**")
        if "Pledged" in df_src.columns:
            med, q1, q3 = median_iqr(df_src["Pledged"])
            iqr = (q3 - q1) if pd.notna(q3) and pd.notna(q1) else np.nan
            st.caption(f"Median={money_short(med)} | IQR={money_short(iqr)} (Q1={money_short(q1)}, Q3={money_short(q3)})")

            fig = px.box(plot_df, x="Pledged", points=points_mode, title="Boxplot: Pledged (Raw)")
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
            show_plot(fig, f"{fig_key_prefix}_pledged_raw")

            pledged_log_all = np.log1p(pd.to_numeric(df_src["Pledged"], errors="coerce"))
            med2, q12, q32 = median_iqr(pledged_log_all)
            iqr2 = (q32 - q12) if pd.notna(q32) and pd.notna(q12) else np.nan
            st.caption(f"log1p(Pledged): Median={med2:.3f} | IQR={iqr2:.3f}")

            pledged_log_plot = np.log1p(pd.to_numeric(plot_df["Pledged"], errors="coerce"))
            fig2 = px.box(pd.DataFrame({"pledged_log": pledged_log_plot}), x="pledged_log", points=points_mode, title="Boxplot: Pledged (Log)")
            fig2.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
            show_plot(fig2, f"{fig_key_prefix}_pledged_log")
        else:
            st.warning("ไม่พบคอลัมน์ Pledged")



    st.markdown("### ก่อนทำความสะอาด (Before) — Outlier View")
    outlier_panel(df_before_raw, "Before", "out_before")

    st.markdown("---")
    st.markdown("### หลังทำความสะอาด (After) — Outlier View")
    outlier_panel(df_after_raw, "After", "out_after")

    st.markdown("---")
    with st.expander("สรุปขั้นตอนทำความสะอาด (Cleaning Steps)"):
        st.markdown(
            """
1) ตรวจสอบคุณภาพข้อมูล (Data Quality): shape, dtype, missing, ค่า invalid, ข้อมูลซ้ำ  
2) แปลงวันเวลา (Datetime Parsing): `Launched`, `Deadline` เป็น datetime (datetime) เพื่อคำนวณ Duration  
3) ตรวจค่าไม่สมเหตุสมผล (Business Invalid): เช่น Goal ≤ 0 (ถ้ากำหนดให้ลบ/กรอง)  
4) จัดการ Outlier (Outlier Handling): ไม่ลบ → ใช้ log1p ลด skew + ใช้ Median/IQR อธิบายร่วม  
            """
        )

# ==========================================================
# TAB 3: EDA & Correlation
# ==========================================================
with tab3:
    st.markdown("## EDA (หลังทำความสะอาด) + ความสัมพันธ์ (Correlation)")
    dff = filtered_df.copy()

    st.markdown("### การกระจาย (Distribution) — Goal / Pledged / Backers")
    n1, n2, n3 = st.columns(3)

    with n1:
        if "Goal" in dff.columns and len(dff) > 0:
            fig = px.histogram(dff, x="Goal", nbins=60, title="Distribution: Goal")
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10))
            show_plot(fig, "eda_goal_hist")
        else:
            st.caption("ไม่พบ Goal หรือไม่มีข้อมูล")

    with n2:
        if "Pledged" in dff.columns and len(dff) > 0:
            fig = px.histogram(dff, x="Pledged", nbins=60, title="Distribution: Pledged")
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10))
            show_plot(fig, "eda_pledged_hist")
        else:
            st.caption("ไม่พบ Pledged หรือไม่มีข้อมูล")

    with n3:
        if "Backers" in dff.columns and len(dff) > 0:
            fig = px.histogram(dff, x="Backers", nbins=60, title="Distribution: Backers")
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10))
            show_plot(fig, "eda_backers_hist")
        else:
            st.caption("ไม่พบ Backers หรือไม่มีข้อมูล")

    st.markdown("---")
    st.markdown("### ความสัมพันธ์ (Correlation) — ตัวแปรตัวเลข")
    num_cols = [c for c in ["Goal", "Pledged", "Backers", "DurationDays"] if c in dff.columns]
    if len(num_cols) >= 2 and len(dff) > 0:
        corr = dff[num_cols].corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Heatmap")
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
        show_plot(fig, "eda_corr")

        if "Pledged" in dff.columns and "Backers" in dff.columns:
            fig2 = px.scatter(dff, x="Backers", y="Pledged", title="Backers vs Pledged")
            fig2.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
            show_plot(fig2, "eda_scatter_backers_pledged")
    else:
        st.warning("ตัวแปรตัวเลขไม่พอสำหรับ correlation หรือไม่มีข้อมูลหลังกรอง")

# ==========================================================
# TAB 4: Insights (What–Why–So What)
# ==========================================================
with tab4:
    st.markdown("## Insights (What–Why–So What) + หลักฐาน (Evidence)")

    dff = filtered_df.copy()
    total = len(dff)

    # Insight 1
    st.markdown("### 🔍 Insight 1: อัตราความสำเร็จ (Success) โดยรวม")
    if "State" in dff.columns and total > 0:
        state_norm = dff["State"].astype(str).str.strip().str.lower()
        order = ["failed", "successful", "canceled", "suspended"]
        counts = [(state_norm == s).sum() for s in order]
        denom = sum(counts) if sum(counts) else 1
        perc = [c / denom * 100 for c in counts]
        labels = [s.title() for s in order]

        sr = (state_norm == "successful").mean()

        st.write(f"**What (พบอะไร):** Success ≈ **{sr*100:.2f}%** (จากข้อมูลหลังคลีน)")
        st.write("**Why (ทำไมเป็นแบบนี้):** มักสะท้อนการตั้งเป้า (Goal) / การสื่อสารแคมเปญ / แรงสนับสนุนช่วงต้น (Early backers)")
        st.write("**So What (ใช้ประโยชน์อย่างไร):** ก่อนตั้ง Goal ควรเทียบโปรเจกต์ที่สำเร็จในหมวดเดียวกัน และติดตาม Backers ช่วงต้นเป็นสัญญาณเตือน")

        df_plot = pd.DataFrame({"State": labels, "Count": counts, "Percent": perc})
        fig = px.bar(df_plot, x="State", y="Count", text=df_plot["Percent"].map(lambda x: f"{x:.2f}%"), title="Project Outcome (Count + %)")
        fig.update_traces(textposition="outside")
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10))
        show_plot(fig, "ins1_outcome")

    st.markdown("---")

    # Insight 2
    st.markdown("### 🔍 Insight 2: หมวด (Category) ที่สำเร็จสูง")
    if all(c in dff.columns for c in ["Category", "State"]) and total > 0:
        tmp = dff.copy()
        tmp["state_norm"] = tmp["State"].astype(str).str.strip().str.lower()
        tmp["is_success"] = (tmp["state_norm"] == "successful").astype(int)

        grp = tmp.groupby("Category", dropna=False).agg(total=("is_success", "size"), success=("is_success", "sum")).reset_index()
        grp["success_rate"] = grp["success"] / grp["total"] * 100
        top = grp.sort_values("success_rate", ascending=False).head(10)

        st.write(f"**What:** Top success rate เช่น {', '.join(top['Category'].astype(str).head(3).tolist())}")
        st.write("**Why:** หมวดสาย creative มักมี community support และฐานแฟนคลับช่วยดันแคมเปญ")
        st.write("**So What:** ใช้ Category เป็นตัวกรองเบื้องต้นเพื่อเพิ่มโอกาสสำเร็จ (โดยเฉพาะงานที่พึ่งพาชุมชน)")

        fig = px.bar(top, x="success_rate", y="Category", orientation="h",
                     text=top["success_rate"].map(lambda x: f"{x:.2f}%"),
                     title="Top Categories by Success Rate")
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10), xaxis_title="Success Rate (%)")
        show_plot(fig, "ins2_cat_success")

    st.markdown("---")

    # Insight 3
    st.markdown("### 🔍 Insight 3: ประเทศ (Country) ที่ครองสัดส่วนโครงการ")
    if "Country" in dff.columns and total > 0:
        cc = dff["Country"].astype(str).str.strip().value_counts().head(12).reset_index()
        cc.columns = ["Country", "Count"]
        cc["Percent"] = cc["Count"] / cc["Count"].sum() * 100

        st.write("**What:** ประเทศอันดับ 1 มีสัดส่วนสูงมากเมื่อเทียบประเทศอื่น")
        st.write("**Why:** ระบบนิเวศ crowdfunding แข็งแรงกว่าในบางประเทศ ทำให้มีจำนวนโปรเจกต์เยอะ")
        st.write("**So What:** เวลาสรุปภาพรวมควรแยกวิเคราะห์ตามประเทศเพื่อลด bias จากประเทศที่มีข้อมูลเยอะ")

        fig = px.bar(cc, x="Count", y="Country", orientation="h", text=cc["Percent"].map(lambda x: f"{x:.2f}%"),
                     title="Top Countries (Share % within Top)")
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
        show_plot(fig, "ins3_country_share")

    st.markdown("---")

    # Insight 4
    st.markdown("### 🔍 Insight 4: Backers สัมพันธ์กับ Pledged สูง")
    if all(c in dff.columns for c in ["Pledged", "Backers"]) and total > 0:
        tmp = dff.dropna(subset=["Pledged", "Backers"]).copy()
        tmp["pledged_log_tmp"] = np.log1p(pd.to_numeric(tmp["Pledged"], errors="coerce"))
        corr_val = tmp[["Pledged", "Backers"]].corr().iloc[0, 1]

        st.write(f"**What:** Correlation(Pledged, Backers) ≈ **{corr_val:.4f}**")
        st.write("**Why:** Backers คือแรงขับหลักของยอดเงิน และเกิดผลเครือข่าย (Network effect) เมื่อแคมเปญเริ่มดัง")
        st.write("**So What:** ใช้การโตของ Backers ช่วงต้นเป็นตัวชี้วัด (Signal) เพื่อทำนายโอกาสสำเร็จและปรับแผนทันเวลา")

        fig = px.scatter(tmp, x="Backers", y="pledged_log_tmp", title="Backers vs Log(Pledged)")
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10), yaxis_title="log1p(Pledged)")
        show_plot(fig, "ins4_scatter")

    st.markdown("---")

    # Insight 5
    st.markdown("### 🔍 Insight 5: ระยะเวลาแคมเปญ (Duration) กับโอกาสสำเร็จ")
    if all(c in dff.columns for c in ["DurationDays", "State"]) and total > 0:
        tmp = dff.dropna(subset=["DurationDays"]).copy()
        tmp["dur_bin"] = pd.cut(
            tmp["DurationDays"],
            bins=[-np.inf, 15, 30, 60, 90, 180, np.inf],
            labels=["0–15", "15–30", "30–60", "60–90", "90–180", "180+"],
        )
        tmp["is_success"] = tmp["State"].astype(str).str.strip().str.lower().eq("successful").astype(int)
        by = tmp.groupby("dur_bin", as_index=False)["is_success"].mean()
        by["success_rate"] = by["is_success"] * 100

        st.write("**What:** กลุ่มระยะเวลาสั้นบางช่วงมี success rate สูงกว่า")
        st.write("**Why:** ความเร่งด่วน (Urgency) ทำให้ตัดสินใจเร็ว และแคมเปญที่เตรียมพร้อมมักไม่ต้องเปิดนาน")
        st.write("**So What:** เลือกระยะเวลาให้เหมาะ (เช่น 15–30 วัน) และโฟกัสแรงสนับสนุนช่วงต้น")

        fig = px.bar(by, x="dur_bin", y="success_rate",
                     text=by["success_rate"].map(lambda x: f"{x:.2f}%"),
                     title="Success Rate by Campaign Duration")
        fig.update_traces(textposition="outside")
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10),
                          xaxis_title="Duration (days)", yaxis_title="Success Rate (%)")
        show_plot(fig, "ins5_duration")

    st.caption("หมายเหตุ: กราฟทุกอันใส่ key แล้ว ป้องกัน StreamlitDuplicateElementId และไม่ใช้ matplotlib เพื่อดีพลอยบน Streamlit Cloud ได้ชัวร์")


