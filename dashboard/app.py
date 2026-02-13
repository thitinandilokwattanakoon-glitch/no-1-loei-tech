# app.py
# ==========================================================
# Crowdfunding Dashboard (Before vs After Cleaning)
# - Login + Demo Account button
# - Onboarding / Quick Guide popup after login
# - 4 Modules (Overview -> Detail)
# - Modern filters (search, multiselect, date range, sliders)
# - Light Blue + White theme, dark-gray text (high contrast)
# ==========================================================

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import matplotlib.pyplot as plt



# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Crowdfunding Dashboard (Before vs After Cleaning)",
    page_icon="📊",
    layout="wide",
)

# -----------------------------
# Theme / CSS (fix dropdown text + light mode look)
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
section[data-testid="stSidebar"] { background: #e6f2ff !important; border-right: 1px solid rgba(15,23,42,0.10); }
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
@st.cache_data(show_spinner=False)
def plot(fig, prefix="plot"):
    st.session_state.setdefault("_plot_i", 0)
    st.session_state["_plot_i"] += 1
    st.plotly_chart(
        fig,
        use_container_width=True,
        key=f"{prefix}_{st.session_state['_plot_i']}"
    )
def show_plot(fig, key: str):
    st.plotly_chart(fig, use_container_width=True, key=key)

def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def safe_to_datetime(series: pd.Series):
    return pd.to_datetime(series, errors="coerce")

def compute_duration_days(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Launched" in out.columns:
        out["Launched_dt"] = safe_to_datetime(out["Launched"])
    if "Deadline" in out.columns:
        out["Deadline_dt"] = safe_to_datetime(out["Deadline"])
    if "Launched_dt" in out.columns and "Deadline_dt" in out.columns:
        out["DurationDays"] = (out["Deadline_dt"] - out["Launched_dt"]).dt.days
    else:
        out["DurationDays"] = np.nan
    return out

def kpi_block(title, value, note=""):
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

def pct(x):
    return f"{x*100:.2f}%"

def money(x):
    try:
        if pd.isna(x):
            return "-"
        return f"{int(round(float(x))):,}"
    except Exception:
        return "-"

def build_filters(df: pd.DataFrame, key_prefix: str = "main") -> pd.DataFrame:
    """
    Sidebar filters (build ONCE):
    - search by Name
    - multiselect (default select ALL): State, Category, Subcategory, Country
    - date range: Launched_dt, Deadline_dt
    - sliders: Goal, Pledged, Backers, DurationDays
    """
    st.sidebar.markdown("## ตัวกรอง (Filters)")
    df2 = df.copy()

    # Search
    q = st.sidebar.text_input(
        "ค้นหาชื่อโครงการ (Search Name)",
        value="",
        key=f"{key_prefix}_search_name",
    )

    # Categorical filters (default select ALL)
    cat_cols = ["State", "Category", "Subcategory", "Country"]
    selected = {}
    for col in cat_cols:
        if col in df2.columns:
            options = sorted(df2[col].dropna().unique().tolist())
            selected[col] = st.sidebar.multiselect(
                f"เลือก {col}",
                options=options,
                default=options,  # ✅ เลือกทั้งหมดไว้ก่อน
                key=f"{key_prefix}_{col}_ms",
            )

    # Date range
    launched_range = None
    if "Launched_dt" in df2.columns and df2["Launched_dt"].notna().any():
        min_d = df2["Launched_dt"].min().date()
        max_d = df2["Launched_dt"].max().date()
        launched_range = st.sidebar.date_input(
            "ช่วงวันเริ่มระดมทุน (Launched range)",
            value=(min_d, max_d),
            key=f"{key_prefix}_launched_range",
        )

    deadline_range = None
    if "Deadline_dt" in df2.columns and df2["Deadline_dt"].notna().any():
        min_dd = df2["Deadline_dt"].min().date()
        max_dd = df2["Deadline_dt"].max().date()
        deadline_range = st.sidebar.date_input(
            "ช่วงวันสิ้นสุด (Deadline range)",
            value=(min_dd, max_dd),
            key=f"{key_prefix}_deadline_range",
        )

    # Numeric sliders (use full min-max to keep intuitive)
    num_cols = ["Goal", "Pledged", "Backers", "DurationDays"]
    ranges = {}
    for col in num_cols:
        if col in df2.columns and pd.api.types.is_numeric_dtype(df2[col]):
            valid = df2[col].dropna()
            if len(valid) > 0:
                ranges[col] = st.sidebar.slider(
                    f"ช่วง {col}",
                    min_value=float(valid.min()),
                    max_value=float(valid.max()),
                    value=(float(valid.min()), float(valid.max())),
                    key=f"{key_prefix}_{col}_slider",
                )

    # Apply filters
    mask = pd.Series(True, index=df2.index)

    if q.strip() and "Name" in df2.columns:
        mask &= df2["Name"].astype(str).str.contains(q.strip(), case=False, na=False)

    for col, vals in selected.items():
        if vals:
            mask &= df2[col].isin(vals)

    if launched_range and "Launched_dt" in df2.columns:
        start, end = launched_range
        mask &= df2["Launched_dt"].dt.date.between(start, end)

    if deadline_range and "Deadline_dt" in df2.columns:
        start, end = deadline_range
        mask &= df2["Deadline_dt"].dt.date.between(start, end)

    for col, (lo, hi) in ranges.items():
        mask &= df2[col].between(lo, hi)

    out = df2.loc[mask].copy()

    st.sidebar.markdown("---")
    st.sidebar.caption(f"ผลลัพธ์หลังกรอง: {len(out):,} แถว")
    return out

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

# -----------------------------
# Authentication (simple)
# -----------------------------
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "show_guide" not in st.session_state:
    st.session_state["show_guide"] = True

def login_view():
    st.markdown("## 🔐 เข้าสู่ระบบ (Login)")
    st.caption("มีปุ่มบัญชีทดลองสำหรับกรรมการ/ผู้ทดสอบ กดครั้งเดียวเข้าใช้งานได้ทันที")

    with st.container():
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

# -----------------------------
# Main App
# -----------------------------
if not st.session_state["logged_in"]:
    login_view()
    st.stop()

# Header
top = st.container()
with top:
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown("# 📊 Crowdfunding Dashboard")
        st.caption("Before (dataset3.csv) vs After (dataclean5.csv) — พร้อมตัวกรอง, EDA, และ Insight แบบมีหลักฐาน")
    with c2:
        if st.button("ออกจากระบบ (Logout)", key="btn_logout"):
            st.session_state["logged_in"] = False
            st.session_state["show_guide"] = True
            st.rerun()

# Onboarding panel
if st.session_state.get("show_guide", True):
    with st.expander("👋 หน้าต่างสอนใช้เบื้องต้น (กดเพื่อดู/ซ่อน)", expanded=True):
        show_onboarding()
else:
    if st.button("📘 เปิดคู่มือ (Quick Guide)", key="btn_open_guide"):
        st.session_state["show_guide"] = True
        st.rerun()

# Load data
# -----------------------------
# Load from Google Drive
# -----------------------------
@st.cache_data(show_spinner=False)
def load_from_drive(file_id):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    return pd.read_csv(url)

df_before = load_from_drive("1qRTrEuENBRdrx4aVzT7WwDg8qsCAEIlh")
df_after  = load_from_drive("15gI9_y2FWKLwvxTvfpjy39sMtuf7bs-i")

with st.spinner("กำลังโหลดข้อมูลจาก Google Drive..."):
    df_before = load_from_drive("1qRTrEuENBRdrx4aVzT7WwDg8qsCAEIlh")
    df_after  = load_from_drive("15gI9_y2FWKLwvxTvfpjy39sMtuf7bs-i")


# Standardize derived fields (IMPORTANT: do BOTH)
# df_before = compute_duration_days(df_before)
df_after = df_after = load_csv("dataclean5.csv")

# -----------------------------
# 1) ลบคอลัมน์ที่ไม่ต้องการ
# -----------------------------
cols_to_drop = ["Launched_dt", "Deadline_dt", "DurationDays"]
df_after = df_after.drop(columns=[c for c in cols_to_drop if c in df_after.columns])

# -----------------------------
# 2) แปลง Launched / Deadline เป็น datetime
# -----------------------------
if "Launched" in df_after.columns:
    df_after["Launched"] = pd.to_datetime(df_after["Launched"], errors="coerce")

if "Deadline" in df_after.columns:
    df_after["Deadline"] = pd.to_datetime(df_after["Deadline"], errors="coerce")


# ✅ Build filters once (shared across tabs)
filtered_df = build_filters(df_after, key_prefix="main")

# Tabs = 4 Modules
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
    dff = filtered_df

    total = len(dff)

    if "State" in dff.columns and total > 0:
        success_rate = dff["State"].eq("successful").mean()
        fail_rate = dff["State"].eq("failed").mean()
    else:
        success_rate, fail_rate = 0, 0

    med_goal = dff["Goal"].median() if "Goal" in dff.columns and total > 0 else np.nan
    med_pledged = dff["Pledged"].median() if "Pledged" in dff.columns and total > 0 else np.nan
    med_backers = dff["Backers"].median() if "Backers" in dff.columns and total > 0 else np.nan

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1: kpi_block("จำนวนโครงการ (Projects)", f"{total:,}", "หลังกรองด้วย Filters")
    with k2: kpi_block("อัตราสำเร็จ (Success)", pct(success_rate), "State = successful")
    with k3: kpi_block("Median Goal", money(med_goal), "ค่ากลางเป้าหมาย")
    with k4: kpi_block("Median Pledged", money(med_pledged), "ค่ากลางเงินที่ได้จริง")
    with k5: kpi_block("Median Backers", money(med_backers), "ค่ากลางจำนวนผู้สนับสนุน")

    st.markdown("---")

    c1, c2 = st.columns([1.25, 1])

    with c1:
        st.markdown("### สัดส่วนสถานะโครงการ (State Share)")
        if "State" in dff.columns and total > 0:
            s = dff["State"].value_counts(dropna=False).reset_index()
            s.columns = ["State", "Count"]
            s["Percent"] = s["Count"] / s["Count"].sum() * 100
            fig = px.bar(s, x="State", y="Count", text=s["Percent"].map(lambda x: f"{x:.1f}%"))
            fig.update_traces(textposition="outside")
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
            show_plot(fig, "unique_key_here")
        else:
            st.warning("ไม่พบคอลัมน์ State หรือไม่มีข้อมูลหลังกรอง")

    with c2:
        st.markdown("### Top Category (ตามจำนวนโครงการ)")
        if "Category" in dff.columns and total > 0:
            top_cat = dff["Category"].value_counts().head(10).reset_index()
            top_cat.columns = ["Category", "Count"]
            fig = px.bar(top_cat, x="Count", y="Category", orientation="h", text="Count")
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True, key="plot_602")
        else:
            st.warning("ไม่พบคอลัมน์ Category หรือไม่มีข้อมูลหลังกรอง")

    st.markdown("### ตารางข้อมูล (หลังกรอง)")
    st.dataframe(dff.head(200), use_container_width=True)

    csv = dff.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ ดาวน์โหลดข้อมูลหลังกรอง (CSV)",
        data=csv,
        file_name="filtered_after_cleaning.csv",
        mime="text/csv",
        key="btn_download_filtered",
    )

# ==========================================================
# TAB 2: Data Quality & Cleaning Steps
# ==========================================================
with tab2:
    st.markdown("## Before vs After (คุณภาพข้อมูล)")
    st.caption("สรุปตามขั้นตอนในใบงาน: แปลงวันเวลา + ลบ Goal ที่ผิดตรรกะ + รับมือ Outlier ด้วย Log Transform (Goal, Pledged)")

    b1, b2 = st.columns(2)

    with b1:
        st.markdown("### ก่อนทำความสะอาด (dataset3.csv)")
        st.write(f"Shape: **{df_before.shape[0]:,} แถว × {df_before.shape[1]:,} คอลัมน์**")
        st.dataframe(
            df_before.dtypes.astype(str).reset_index().rename(columns={"index": "Feature", 0: "dtype"}),
            use_container_width=True,
        )

        if "Goal" in df_before.columns:
            invalid_goal = (df_before["Goal"] <= 0).sum()
            st.info(f"พบ Goal ≤ 0 (ผิดตรรกะธุรกิจ): **{invalid_goal:,} แถว**")

    with b2:
        st.markdown("### หลังทำความสะอาด (dataclean5.csv)")
        st.write(f"Shape: **{df_after.shape[0]:,} แถว × {df_after.shape[1]:,} คอลัมน์**")
        st.dataframe(
            df_after.dtypes.astype(str).reset_index().rename(columns={"index": "Feature", 0: "dtype"}),
            use_container_width=True,
        )

        if "Goal" in df_after.columns:
            invalid_goal2 = (df_after["Goal"] <= 0).sum()
            st.success(f"หลังทำความสะอาด Goal ≤ 0 เหลือ: **{invalid_goal2:,} แถว**")

    st.markdown("---")
    st.markdown("## ขั้นตอนการจัดการข้อมูล (Cleaning Steps)")

    with st.expander("ดูขั้นตอนแบบสรุป (ทำซ้ำได้)", expanded=True):
        st.markdown(
            """
**Step 1: ตรวจสอบคุณภาพข้อมูล (Data Quality Check)**  
- ตรวจจำนวนแถว/คอลัมน์, ชนิดข้อมูล (dtype), missing, ค่า invalid, ข้อมูลซ้ำ

**Step 2: แก้ชนิดข้อมูลวันเวลา (Datetime Parsing)**  
- `Launched`, `Deadline` จากข้อความ → `datetime` เพื่อใช้วิเคราะห์ช่วงเวลา/Duration

**Step 3: แก้ข้อมูลผิดตรรกะธุรกิจ (Business Invalid)**  
- ลบแถวที่ `Goal <= 0` (เป้าหมายระดมทุนไม่ควรเป็น 0 หรือค่าติดลบ)

**Step 4: จัดการ Outlier (ไม่ลบ แต่ลดความเบ้)**  
- ใช้ `Log Transformation` กับ `Goal` และ `Pledged`  
- อธิบายผลด้วย Median/IQR ร่วม (เหมาะกับข้อมูลเบ้มาก)
            """
        )

    with st.expander("โค้ดตัวอย่าง Cleaning (ย่อและอ่านง่าย)"):
        st.code(
            """
import pandas as pd
import numpy as np

df = pd.read_csv("dataset3.csv")

# 1) Parse datetime
df["Launched"] = pd.to_datetime(df["Launched"], errors="coerce")
df["Deadline"] = pd.to_datetime(df["Deadline"], errors="coerce")

# 2) Remove invalid goal
df = df[df["Goal"] > 0].copy()

# 3) Duration
df["DurationDays"] = (df["Deadline"] - df["Launched"]).dt.days

# 4) Log transform (keep original too)
df["log_goal"] = np.log1p(df["Goal"])
df["log_pledged"] = np.log1p(df["Pledged"])

df.to_csv("dataclean5.csv", index=False)
            """,
            language="python",
        )

# ==========================================================
# TAB 3: EDA & Correlation
# ==========================================================
with tab3:
    st.markdown("## EDA (หลังทำความสะอาด) + ความสัมพันธ์ตัวแปร")
    dff = filtered_df

    st.markdown("### การกระจายของตัวแปรเชิงตัวเลข (Distribution)")
    n1, n2, n3 = st.columns(3)

    with n1:
        if "Goal" in dff.columns and len(dff) > 0:
            fig = px.histogram(dff, x="Goal", nbins=60)
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("ไม่พบคอลัมน์ Goal หรือไม่มีข้อมูล")

    with n2:
        if "Pledged" in dff.columns and len(dff) > 0:
            fig = px.histogram(dff, x="Pledged", nbins=60)
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10))
            plot(fig, "fix602")

        else:
            st.caption("ไม่พบคอลัมน์ Pledged หรือไม่มีข้อมูล")

    with n3:
        if "Backers" in dff.columns and len(dff) > 0:
            fig = px.histogram(dff, x="Backers", nbins=60)
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10))
            plot(fig, "auto")
        else:
            st.caption("ไม่พบคอลัมน์ Backers หรือไม่มีข้อมูล")

    st.markdown("---")
    st.markdown("### ความสัมพันธ์ (Correlation) ระหว่างตัวแปรเชิงตัวเลข")

    num_cols = [c for c in ["Goal", "Pledged", "Backers", "DurationDays"] if c in dff.columns]
    if len(num_cols) >= 2 and len(dff) > 0:
        corr = dff[num_cols].corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=True, aspect="auto")
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
        plot(fig, "auto")

        if "Pledged" in dff.columns and "Backers" in dff.columns:
            fig = px.scatter(dff, x="Backers", y="Pledged", trendline="ols")
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
            plot(fig, "auto")
    else:
        st.warning("ตัวแปรเชิงตัวเลขไม่พอสำหรับ correlation หรือไม่มีข้อมูลหลังกรอง")

# ==========================================================
# TAB 4: Insights (What–Why–So What)
# ==========================================================
with tab4:
    st.markdown("## Insights (มีหลักฐานด้วยกราฟ/ตัวเลข)")

    dff = filtered_df.copy()

    # --- Helper: plot bar with percent labels (matplotlib) ---
    def bar_with_percent(x_labels, values, title, xlabel, ylabel, percent_values=None, rotate=0):
        fig, ax = plt.subplots()
        bars = ax.bar(x_labels, values)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # label percent on top
        if percent_values is None:
            total = sum(values) if sum(values) != 0 else 1
            percent_values = [(v / total) * 100 for v in values]

        for b, p in zip(bars, percent_values):
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height(),
                f"{p:.2f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        if rotate != 0:
            plt.xticks(rotation=rotate, ha="right")

        st.pyplot(fig)

    # ==========================================================
    # Insight 1: Project Outcome (Count + %)
    # ต้องมี Failed / Successful / Canceled / Suspended และโชว์ % บนแท่ง
    # ==========================================================
    st.markdown("### 🔍 Insight 1: อัตราความสำเร็จโดยรวมค่อนข้างต่ำ (~38.5%)")

    if "State" in dff.columns and len(dff) > 0:
        # ทำให้ชื่อ state เป็นรูปแบบเดียวกับที่คุณทำในรูป (title case)
        state_series = dff["State"].astype(str).str.strip().str.lower()

        # จัด order ตามรูป: Failed, Successful, Canceled, Suspended
        order = ["failed", "successful", "canceled", "suspended"]
        state_counts = state_series.value_counts()

        counts = [int(state_counts.get(s, 0)) for s in order]
        total = sum(counts) if sum(counts) != 0 else 1
        percents = [(c / total) * 100 for c in counts]
        labels = [s.capitalize() for s in order]

        st.write(f"**What (พบอะไร):** Successful ≈ **{percents[1]:.2f}%** | Failed ≈ **{percents[0]:.2f}%**")
        st.write("**Why (ทำไมเป็นแบบนี้):** Goal อาจตั้งสูงเกินไป + การแข่งขันสูง + แคมเปญไม่ปังช่วงต้น")
        st.write("**So What (ใช้ประโยชน์อย่างไร):** ดูโครงการที่สำเร็จในหมวดเดียวกันก่อนตั้ง Goal และติดตาม Backers ช่วงต้น")

        bar_with_percent(
            x_labels=labels,
            values=counts,
            title="Project Outcome (Count + %)",
            xlabel="State",
            ylabel="Count",
            percent_values=percents,
        )
    else:
        st.warning("ไม่พบคอลัมน์ State หรือไม่มีข้อมูลหลังกรอง")

    st.markdown("---")

    # ==========================================================
    # Insight 2: Top Categories by Success Rate (%)
    # success_rate = successful / total ใน Category
    # ==========================================================
    st.markdown('### 🔍 Insight 2: หมวด “Dance” และ “Theater” มีอัตราความสำเร็จสูงที่สุด')

    if all(c in dff.columns for c in ["Category", "State"]) and len(dff) > 0:
        tmp = dff.copy()
        tmp["State_norm"] = tmp["State"].astype(str).str.strip().str.lower()
        tmp["is_success"] = (tmp["State_norm"] == "successful").astype(int)

        grp = tmp.groupby("Category", dropna=False).agg(
            total=("is_success", "size"),
            success=("is_success", "sum")
        ).reset_index()

        grp["success_rate"] = (grp["success"] / grp["total"]) * 100
        top = grp.sort_values("success_rate", ascending=False).head(10)

        st.write(
            f"**What:** Top Success Rate = "
            f"{top.iloc[0]['Category']} ~ {top.iloc[0]['success_rate']:.0f}%, "
            f"{top.iloc[1]['Category']} ~ {top.iloc[1]['success_rate']:.0f}%, "
            f"{top.iloc[2]['Category']} ~ {top.iloc[2]['success_rate']:.0f}%"
        )
        st.write("**Why:** หมวดสายการแสดงมี community support สูง มักตั้ง Goal ไม่สูงมาก และฐานแฟนคลับชัดเจน")
        st.write("**So What:** ถ้าทำโปรเจกต์ใหม่ เลือกหมวด Creative Arts มีโอกาสสำเร็จสูงกว่า และใช้ Category เป็นตัวกรองเบื้องต้น")

        fig, ax = plt.subplots()
        ax.bar(top["Category"].astype(str), top["success_rate"])
        ax.set_title("Top Categories by Success Rate")
        ax.set_xlabel("Category")
        ax.set_ylabel("Percent (%)")

        for i, v in enumerate(top["success_rate"].values):
            ax.text(i, v, f"{v:.2f}%", ha="center", va="bottom", fontsize=9)

        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)
    else:
        st.warning("ไม่พบคอลัมน์ Category/State หรือไม่มีข้อมูลหลังกรอง")

    st.markdown("---")

    # ==========================================================
    # Insight 3: Top Countries by Project Share (%)
    # share = count(country)/total
    # ==========================================================
    st.markdown("### 🔍 Insight 3: สหรัฐอเมริกาครองสัดส่วนโครงการส่วนใหญ่")

    if "Country" in dff.columns and len(dff) > 0:
        country_counts = dff["Country"].astype(str).str.strip().value_counts()
        topc = country_counts.head(5)

        total = int(country_counts.sum()) if int(country_counts.sum()) != 0 else 1
        percents = (topc / total) * 100

        st.write(
            f"**What:** United States ≈ {percents.iloc[0]:.2f}% "
            f"(ตัวอย่าง Top 5 ตามกราฟ)"
        )
        st.write("**Why:** ตลาด crowdfunding ใหญ่ + ecosystem แข็งแรงในบางประเทศ (โดยเฉพาะ US)")
        st.write("**So What:** ระวัง bias จาก US สูงมาก ถ้าสรุปเชิงนโยบายควรแยกประเทศก่อนวิเคราะห์")

        fig, ax = plt.subplots()
        ax.bar(topc.index.astype(str), percents.values)
        ax.set_title("Top Countries by Project Share (%)")
        ax.set_xlabel("Country")
        ax.set_ylabel("Percent (%)")

        for i, v in enumerate(percents.values):
            ax.text(i, v, f"{v:.2f}%", ha="center", va="bottom", fontsize=9)

        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)
    else:
        st.warning("ไม่พบคอลัมน์ Country หรือไม่มีข้อมูลหลังกรอง")

    st.markdown("---")

    # ==========================================================
    # Insight 4: Backers vs Log(Pledged)
    # ใช้ log1p(Pledged) เหมือนกราฟที่คุณทำ
    # และโชว์ Correlation(Pledged, Backers)
    # ==========================================================
    st.markdown("### 🔍 Insight 4: จำนวน Backers สัมพันธ์สูงกับยอดเงินที่ได้")

    if all(c in dff.columns for c in ["Pledged", "Backers"]) and len(dff) > 0:
        tmp = dff.dropna(subset=["Pledged", "Backers"]).copy()
        tmp["log_pledged"] = np.log1p(tmp["Pledged"])

        corr_val = tmp[["Pledged", "Backers"]].corr().iloc[0, 1]

        st.write(f"**What:** Correlation ระหว่าง Pledged ↔ Backers = **{corr_val:.4f}** (สูงมาก)")
        st.write("**Why:** Backers คือแรงขับหลักของยอดเงิน และโปรเจกต์ที่ viral จะโตจาก backers ได้เร็ว")
        st.write("**So What:** Early backer growth ใช้เป็นสัญญาณทำนายความสำเร็จ ควรติดตาม growth rate ของ backers ให้ชัดตั้งแต่ต้นแคมเปญ")

        fig, ax = plt.subplots()
        ax.scatter(tmp["Backers"], tmp["log_pledged"], s=12)
        ax.set_title("Backers vs Log(Pledged)")
        ax.set_xlabel("Backers")
        ax.set_ylabel("pledged_log")
        st.pyplot(fig)
    else:
        st.warning("ไม่พบคอลัมน์ Pledged/Backers หรือไม่มีข้อมูลหลังกรอง")

    st.markdown("---")


