# app.py
# ==========================================================
# Crowdfunding Dashboard (Before vs After Cleaning)
# - Login + Demo Account button
# - Onboarding / Quick Guide popup after login
# - 4 Modules (Overview -> Detail)
# - Modern filters (search, multiselect, date range, sliders)
# - Light Blue + White theme, dark-gray text (high contrast)
# - Load CSV from Google Drive (NO local file dependency)
# - Plotly only (NO matplotlib)
# - Fix DuplicateElementId by unique keys
# ==========================================================

import re
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# -----------------------------
# Page Config (MUST be first Streamlit call)
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
# Theme / CSS
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
def _drive_id(url: str) -> str:
    m = re.search(r"/d/([^/]+)", url)
    return m.group(1) if m else url

@st.cache_data(show_spinner=False)
def load_drive_csv(url: str) -> pd.DataFrame:
    fid = _drive_id(url)
    # uc download
    csv_url = f"https://drive.google.com/uc?export=download&id={fid}"
    return pd.read_csv(csv_url)

def pct(x: float) -> str:
    return f"{x*100:.2f}%"

def money(x) -> str:
    try:
        if pd.isna(x):
            return "-"
        return f"{int(round(float(x))):,}"
    except Exception:
        return "-"

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

def plot(fig, key_prefix="plot"):
    """Plotly chart with auto-unique key to avoid StreamlitDuplicateElementId."""
    st.session_state.setdefault("_plot_i", 0)
    st.session_state["_plot_i"] += 1
    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_{st.session_state['_plot_i']}")

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


def prepare_after_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Requirement:
    - After file must NOT contain Launched_dt / Deadline_dt / DurationDays
    - Convert Launched, Deadline to datetime in-place (same column)
    - Use internal _duration_days for filters/EDA (not shown)
    """
    out = df.copy()

    # drop any accidental columns from prior scripts
    cols_to_drop = ["Launched_dt", "Deadline_dt", "DurationDays"]
    out = out.drop(columns=[c for c in cols_to_drop if c in out.columns], errors="ignore")

    # ensure datetime in same columns
    if "Launched" in out.columns:
        out["Launched"] = pd.to_datetime(out["Launched"], errors="coerce")
    if "Deadline" in out.columns:
        out["Deadline"] = pd.to_datetime(out["Deadline"], errors="coerce")

    # internal duration (for filter only)
    if "Launched" in out.columns and "Deadline" in out.columns:
        out["_duration_days"] = (out["Deadline"] - out["Launched"]).dt.days
    else:
        out["_duration_days"] = np.nan

    return out


def build_filters(df: pd.DataFrame, key_prefix: str = "main") -> pd.DataFrame:
    st.sidebar.markdown("## ตัวกรอง (Filters)")
    df2 = df.copy()

    q = st.sidebar.text_input(
        "ค้นหาชื่อโครงการ (Search Name)",
        value="",
        key=f"{key_prefix}_search_name",
    )

    # categorical
    cat_cols = ["State", "Category", "Subcategory", "Country"]
    selected = {}
    for col in cat_cols:
        if col in df2.columns:
            options = sorted(df2[col].dropna().unique().tolist())
            selected[col] = st.sidebar.multiselect(
                f"เลือก {col}",
                options=options,
                default=options,
                key=f"{key_prefix}_{col}_ms",
            )

    # date ranges (use Launched/Deadline directly)
    launched_range = None
    if "Launched" in df2.columns and pd.api.types.is_datetime64_any_dtype(df2["Launched"]) and df2["Launched"].notna().any():
        min_d = df2["Launched"].min().date()
        max_d = df2["Launched"].max().date()
        launched_range = st.sidebar.date_input(
            "ช่วงวันเริ่มระดมทุน (Launched range)",
            value=(min_d, max_d),
            key=f"{key_prefix}_launched_range",
        )

    deadline_range = None
    if "Deadline" in df2.columns and pd.api.types.is_datetime64_any_dtype(df2["Deadline"]) and df2["Deadline"].notna().any():
        min_dd = df2["Deadline"].min().date()
        max_dd = df2["Deadline"].max().date()
        deadline_range = st.sidebar.date_input(
            "ช่วงวันสิ้นสุด (Deadline range)",
            value=(min_dd, max_dd),
            key=f"{key_prefix}_deadline_range",
        )

    # numeric sliders
    num_cols = ["Goal", "Pledged", "Backers", "_duration_days"]
    ranges = {}
    for col in num_cols:
        if col in df2.columns and pd.api.types.is_numeric_dtype(df2[col]):
            valid = df2[col].dropna()
            if len(valid) > 0:
                label = "DurationDays" if col == "_duration_days" else col
                ranges[col] = st.sidebar.slider(
                    f"ช่วง {label}",
                    min_value=float(valid.min()),
                    max_value=float(valid.max()),
                    value=(float(valid.min()), float(valid.max())),
                    key=f"{key_prefix}_{col}_slider",
                )

    # apply filters
    mask = pd.Series(True, index=df2.index)

    if q.strip() and "Name" in df2.columns:
        mask &= df2["Name"].astype(str).str.contains(q.strip(), case=False, na=False)

    for col, vals in selected.items():
        if vals:
            mask &= df2[col].isin(vals)

    if launched_range and "Launched" in df2.columns:
        start, end = launched_range
        mask &= df2["Launched"].dt.date.between(start, end)

    if deadline_range and "Deadline" in df2.columns:
        start, end = deadline_range
        mask &= df2["Deadline"].dt.date.between(start, end)

    for col, (lo, hi) in ranges.items():
        mask &= df2[col].between(lo, hi)

    out = df2.loc[mask].copy()
    st.sidebar.markdown("---")
    st.sidebar.caption(f"ผลลัพธ์หลังกรอง: {len(out):,} แถว")
    return out


# -----------------------------
# Authentication
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
                    st.rerun()
                else:
                    st.error("ชื่อผู้ใช้/รหัสผ่านไม่ถูกต้อง (ลองใช้บัญชีทดลอง)")
        with col2:
            if st.button("บัญชีทดลอง (Demo)", key="btn_demo"):
                st.session_state["logged_in"] = True
                st.rerun()
        with col3:
            st.caption("บัญชีตัวอย่าง: admin/admin123 หรือ judge/judge123 (หรือกด Demo)")
        st.markdown("</div>", unsafe_allow_html=True)


if not st.session_state["logged_in"]:
    login_view()
    st.stop()


# -----------------------------
# Load data from Drive
# -----------------------------
with st.spinner("กำลังโหลดข้อมูลจาก Google Drive..."):
    df_before = load_drive_csv(BEFORE_URL)
    df_after_raw = load_drive_csv(AFTER_URL)

df_after = prepare_after_df(df_after_raw)

# -----------------------------
# Header
# -----------------------------
top = st.container()
with top:
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown("# 📊 Crowdfunding Dashboard")
        st.caption("Before vs After Cleaning (Google Drive Data) — พร้อมตัวกรอง, EDA, และ Insight แบบมีหลักฐาน")
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

# Build filters ONCE
filtered_df = build_filters(df_after, key_prefix="main")

# Tabs
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
    success_rate = dff["State"].eq("successful").mean() if "State" in dff.columns and total > 0 else 0

    med_goal = dff["Goal"].median() if "Goal" in dff.columns and total > 0 else np.nan
    med_pledged = dff["Pledged"].median() if "Pledged" in dff.columns and total > 0 else np.nan
    med_backers = dff["Backers"].median() if "Backers" in dff.columns and total > 0 else np.nan

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1: kpi_block("Projects", f"{total:,}", "หลังกรองด้วย Filters")
    with k2: kpi_block("Success", pct(success_rate), "State=successful")
    with k3: kpi_block("Median Goal", money(med_goal), "")
    with k4: kpi_block("Median Pledged", money(med_pledged), "")
    with k5: kpi_block("Median Backers", money(med_backers), "")

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
            plot(fig, "state_share")
        else:
            st.warning("ไม่พบคอลัมน์ State หรือไม่มีข้อมูลหลังกรอง")

    with c2:
        st.markdown("### Top Category (ตามจำนวนโครงการ)")
        if "Category" in dff.columns and total > 0:
            top_cat = dff["Category"].value_counts().head(10).reset_index()
            top_cat.columns = ["Category", "Count"]
            fig = px.bar(top_cat, x="Count", y="Category", orientation="h", text="Count")
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
            plot(fig, "top_cat")
        else:
            st.warning("ไม่พบคอลัมน์ Category หรือไม่มีข้อมูลหลังกรอง")

    st.markdown("### ตารางข้อมูล (หลังกรอง)")
    # hide internal col
    show_cols = [c for c in dff.columns if not c.startswith("_")]
    st.dataframe(dff[show_cols].head(200), use_container_width=True)

    csv = dff[show_cols].to_csv(index=False).encode("utf-8")
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
    st.caption("แสดงโครงสร้างข้อมูล + ตรวจ Goal ผิดตรรกะ (Goal<=0) และการแปลงวันเวลา Launched/Deadline เป็น datetime")

    b1, b2 = st.columns(2)

    with b1:
        st.markdown("### ก่อนทำความสะอาด (Before)")
        st.write(f"Shape: **{df_before.shape[0]:,} แถว × {df_before.shape[1]:,} คอลัมน์**")
        st.dataframe(
            df_before.dtypes.astype(str).reset_index().rename(columns={"index": "Feature", 0: "dtype"}),
            use_container_width=True,
        )
        if "Goal" in df_before.columns:
            invalid_goal = (pd.to_numeric(df_before["Goal"], errors="coerce") <= 0).sum()
            st.info(f"พบ Goal ≤ 0 (ผิดตรรกะธุรกิจ): **{invalid_goal:,} แถว**")

    with b2:
        st.markdown("### หลังทำความสะอาด (After)")
        show_cols_after = [c for c in df_after.columns if not c.startswith("_")]
        st.write(f"Shape: **{df_after.shape[0]:,} แถว × {len(show_cols_after):,} คอลัมน์ (ไม่นับคอลัมน์ภายใน)**")
        st.dataframe(
            df_after[show_cols_after].dtypes.astype(str).reset_index().rename(columns={"index": "Feature", 0: "dtype"}),
            use_container_width=True,
        )

        # confirm removed columns
        removed = [c for c in ["Launched_dt", "Deadline_dt", "DurationDays"] if c in df_after.columns]
        if len(removed) == 0:
            st.success("✅ ไม่มีคอลัมน์ Launched_dt / Deadline_dt / DurationDays ตามที่ต้องการ")
        else:
            st.warning(f"ยังพบคอลัมน์ที่ไม่ควรมี: {removed}")

    st.markdown("---")
    st.markdown("## ขั้นตอนการจัดการข้อมูล (Cleaning Steps)")
    with st.expander("ดูขั้นตอนแบบสรุป (ทำซ้ำได้)", expanded=True):
        st.markdown(
            """
**Step 1: ตรวจสอบคุณภาพข้อมูล (Data Quality Check)**  
- ตรวจจำนวนแถว/คอลัมน์, ชนิดข้อมูล, missing, ค่า invalid, ข้อมูลซ้ำ

**Step 2: แปลงวันเวลา (Datetime Parsing)**  
- แปลง `Launched`, `Deadline` จากข้อความ → `datetime` (อยู่คอลัมน์เดิม)

**Step 3: ตรวจ Goal ผิดตรรกะธุรกิจ (Business Invalid)**  
- ตรวจ `Goal <= 0` เพื่อชี้จุดผิดตรรกะ

**Step 4: EDA & Insight**  
- วิเคราะห์ State, Category, Country และความสัมพันธ์ Pledged ↔ Backers
            """
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
            fig = px.histogram(dff, x="Goal", nbins=60, title="Goal Distribution")
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
            plot(fig, "hist_goal")
        else:
            st.caption("ไม่พบคอลัมน์ Goal หรือไม่มีข้อมูล")

    with n2:
        if "Pledged" in dff.columns and len(dff) > 0:
            fig = px.histogram(dff, x="Pledged", nbins=60, title="Pledged Distribution")
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
            plot(fig, "hist_pledged")
        else:
            st.caption("ไม่พบคอลัมน์ Pledged หรือไม่มีข้อมูล")

    with n3:
        if "Backers" in dff.columns and len(dff) > 0:
            fig = px.histogram(dff, x="Backers", nbins=60, title="Backers Distribution")
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
            plot(fig, "hist_backers")
        else:
            st.caption("ไม่พบคอลัมน์ Backers หรือไม่มีข้อมูล")

    st.markdown("---")
    st.markdown("### ความสัมพันธ์ (Correlation) ระหว่างตัวแปรเชิงตัวเลข")

    num_cols = [c for c in ["Goal", "Pledged", "Backers", "_duration_days"] if c in dff.columns]
    if len(num_cols) >= 2 and len(dff) > 0:
        corr = dff[num_cols].corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Heatmap")
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
        plot(fig, "corr")

        if "Pledged" in dff.columns and "Backers" in dff.columns:
            tmp = dff.dropna(subset=["Pledged", "Backers"]).copy()
            tmp["pledged_log"] = np.log1p(tmp["Pledged"])
            fig = px.scatter(tmp, x="Backers", y="pledged_log", title="Backers vs Log(Pledged)")
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
            plot(fig, "scatter_backers_logpledged")
    else:
        st.warning("ตัวแปรเชิงตัวเลขไม่พอสำหรับ correlation หรือไม่มีข้อมูลหลังกรอง")

# ==========================================================
# TAB 4: Insights (What–Why–So What) [Plotly only]
# ==========================================================
with tab4:
    st.markdown("## Insights (มีหลักฐานด้วยกราฟ/ตัวเลข)")
    dff = filtered_df.copy()

    # --------------------------
    # Insight 1
    # --------------------------
    st.markdown("### 🔍 Insight 1: อัตราความสำเร็จโดยรวม (State Share)")
    if "State" in dff.columns and len(dff) > 0:
        s = dff["State"].astype(str).str.strip().str.lower().value_counts().reset_index()
        s.columns = ["State", "Count"]
        s["Percent"] = s["Count"] / s["Count"].sum() * 100

        sr = dff["State"].astype(str).str.lower().eq("successful").mean()
        st.write(f"**What:** Success ≈ **{sr*100:.2f}%**")
        st.write("**Why:** ตั้ง Goal สูง/การแข่งขันสูง/ช่วงต้นแคมเปญไม่ดึง Backers")
        st.write("**So What:** เทียบ Goal กับโปรเจกต์สำเร็จในหมวดเดียวกัน + ติดตาม backers ช่วงต้นเป็นสัญญาณเตือน")

        fig = px.bar(s, x="State", y="Count", text=s["Percent"].map(lambda x: f"{x:.2f}%"), title="Project Outcome (Count + %)")
        fig.update_traces(textposition="outside")
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))
        plot(fig, "ins1_state")
    else:
        st.warning("ไม่พบคอลัมน์ State หรือไม่มีข้อมูลหลังกรอง")

    st.markdown("---")

    # --------------------------
    # Insight 2
    # --------------------------
    st.markdown('### 🔍 Insight 2: หมวด “Dance” / “Theater” สำเร็จสูง (Category Success Rate)')
    if all(c in dff.columns for c in ["Category", "State"]) and len(dff) > 0:
        tmp = dff.copy()
        tmp["state_norm"] = tmp["State"].astype(str).str.strip().str.lower()
        tmp["is_success"] = (tmp["state_norm"] == "successful").astype(int)

        grp = tmp.groupby("Category", dropna=False).agg(
            total=("is_success", "size"),
            success=("is_success", "sum"),
        ).reset_index()
        grp["success_rate"] = (grp["success"] / grp["total"]) * 100
        top = grp.sort_values("success_rate", ascending=False).head(10)

        if len(top) >= 3:
            st.write(f"**What:** Top = {top.iloc[0]['Category']} ~ {top.iloc[0]['success_rate']:.2f}% | "
                     f"{top.iloc[1]['Category']} ~ {top.iloc[1]['success_rate']:.2f}% | "
                     f"{top.iloc[2]['Category']} ~ {top.iloc[2]['success_rate']:.2f}%")
        st.write("**Why:** Creative arts มี community support สูง และมักตั้ง Goal ไม่สูงมาก")
        st.write("**So What:** ใช้ Category เป็นตัวกรองเบื้องต้นเพื่อเพิ่มโอกาสสำเร็จ")

        fig = px.bar(top, x="success_rate", y="Category", orientation="h",
                     text=top["success_rate"].map(lambda x: f"{x:.2f}%"),
                     title="Top Categories by Success Rate")
        fig.update_traces(textposition="outside")
        fig.update_layout(height=450, margin=dict(l=10, r=10, t=60, b=10), xaxis_title="Percent (%)")
        plot(fig, "ins2_cat")
    else:
        st.warning("ไม่พบคอลัมน์ Category/State หรือไม่มีข้อมูลหลังกรอง")

    st.markdown("---")

    # --------------------------
    # Insight 3
    # --------------------------
    st.markdown("### 🔍 Insight 3: ประเทศมี Bias (Country Share)")
    if "Country" in dff.columns and len(dff) > 0:
        cc = dff["Country"].astype(str).str.strip().value_counts().head(10).reset_index()
        cc.columns = ["Country", "Count"]
        cc["Percent"] = cc["Count"] / cc["Count"].sum() * 100

        st.write(f"**What:** ประเทศอันดับ 1 มีสัดส่วนสูงมาก (ดูกราฟ)")
        st.write("**Why:** ตลาด/แพลตฟอร์มในบางประเทศใหญ่ ทำให้จำนวนโปรเจกต์มากกว่า")
        st.write("**So What:** เวลาเทียบภาพรวมควรแยกประเทศก่อน ไม่งั้นผลจะเอนตามประเทศข้อมูลเยอะ")

        fig = px.bar(cc, x="Country", y="Percent", text=cc["Percent"].map(lambda x: f"{x:.2f}%"), title="Top Countries by Project Share (%)")
        fig.update_traces(textposition="outside")
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))
        plot(fig, "ins3_country")
    else:
        st.warning("ไม่พบคอลัมน์ Country หรือไม่มีข้อมูลหลังกรอง")

    st.markdown("---")

    # --------------------------
    # Insight 4
    # --------------------------
    st.markdown("### 🔍 Insight 4: Backers สัมพันธ์กับยอดเงิน (Backers vs Log(Pledged))")
    if all(c in dff.columns for c in ["Pledged", "Backers"]) and len(dff) > 0:
        tmp = dff.dropna(subset=["Pledged", "Backers"]).copy()
        tmp["pledged_log"] = np.log1p(tmp["Pledged"])
        corr_val = tmp[["Pledged", "Backers"]].corr(numeric_only=True).iloc[0, 1]

        st.write(f"**What:** Correlation(Pledged, Backers) ≈ **{corr_val:.4f}**")
        st.write("**Why:** จำนวนผู้สนับสนุนคือแรงขับหลักของยอดเงิน และเกิดผลไวรัล/เครือข่าย")
        st.write("**So What:** ใช้ early backer growth เป็นสัญญาณทำนายความสำเร็จ + ปรับการตลาดให้ทัน")

        fig = px.scatter(tmp, x="Backers", y="pledged_log", title="Backers vs Log(Pledged)")
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))
        plot(fig, "ins4_scatter")
    else:
        st.warning("ไม่พบคอลัมน์ Pledged/Backers หรือไม่มีข้อมูลหลังกรอง")

    st.markdown("---")

    # --------------------------
    # Insight 5
    # --------------------------
    st.markdown("### 🔍 Insight 5: ระยะเวลาสั้น (≤15 วัน) มีโอกาสสำเร็จสูงกว่า")
    if "_duration_days" in dff.columns and "State" in dff.columns and len(dff) > 0:
        tmp = dff.dropna(subset=["_duration_days"]).copy()
        tmp["dur_bin"] = pd.cut(
            tmp["_duration_days"],
            bins=[-np.inf, 15, 30, 60, 180, np.inf],
            labels=["0–15", "15–30", "30–60", "60–180", "180+"],
        )
        tmp["is_success"] = tmp["State"].astype(str).str.lower().eq("successful").astype(int)
        by = tmp.groupby("dur_bin", as_index=False)["is_success"].mean()
        by["success_rate"] = by["is_success"] * 100

        st.write("**What:** กลุ่มระยะสั้นมี success rate สูงกว่า (ดูกราฟ)")
        st.write("**Why:** ความเร่งด่วน (urgency) ทำให้ตัดสินใจสนับสนุนเร็ว")
        st.write("**So What:** ช่วงที่เหมาะสมมักอยู่ 15–30 วัน ไม่จำเป็นต้องเปิดนาน")

        fig = px.bar(by, x="dur_bin", y="success_rate",
                     text=by["success_rate"].map(lambda x: f"{x:.2f}%"),
                     title="Success Rate by Campaign Duration")
        fig.update_traces(textposition="outside")
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10),
                          xaxis_title="Duration Group (days)", yaxis_title="Success Rate (%)")
        plot(fig, "ins5_duration")
    else:
        st.warning("ไม่พบข้อมูลระยะเวลา (Duration) หรือไม่มีคอลัมน์ State")
