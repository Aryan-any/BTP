import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests

DB_PATH = "scam_data.db"
API_URL = "http://127.0.0.1:8000/detect"

st.set_page_config(page_title="Aegis Crypto Intelligence", layout="wide", page_icon="🛡️", initial_sidebar_state="expanded")

# --- Vibrant, Modern CSS Overhaul ---
st.markdown("""
<style>
    .hero-banner {
        background: linear-gradient(135deg, #090e17 0%, #111827 50%, #1e3a8a 100%);
        color: white;
        padding: 40px 20px;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 10px 40px rgba(30, 58, 138, 0.25);
    }
    .hero-title {
        font-size: 3rem;
        font-weight: 900;
        margin: 0;
        letter-spacing: -1px;
    }
    .hero-subtitle {
        font-size: 1.2rem;
        font-weight: 500;
        margin-top: 10px;
        opacity: 0.95;
    }
    
    .metric-card {
        background: #ffffff;
        border: 1px solid #f0f0f0;
        padding: 24px;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }
    
    .metric-card-risk { border-top: 5px solid #EF4444; }
    .metric-card-warn { border-top: 5px solid #F59E0B; }
    .metric-card-safe { border-top: 5px solid #10B981; }
    .metric-card-info { border-top: 5px solid #3B82F6; }
    .metric-card-gray { border-top: 5px solid #64748b; }

    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.08);
    }
    .metric-title {
        font-size: 0.9rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 12px;
        font-weight: 700;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 900;
        margin: 0;
    }
    
    .alert-flag {
        background: #FEF2F2;
        border-left: 4px solid #EF4444;
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 12px;
        color: #991B1B;
        font-size: 1.05rem;
        font-weight: 500;
    }
    
    .safe-flag {
        background: #F0FDF4;
        border-left: 4px solid #22C55E;
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 12px;
        color: #166534;
        font-size: 1.05rem;
        font-weight: 500;
    }
    
    h2, h3, h4 {
        color: #1e293b;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# --- Hero Section ---
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">🛡️ Aegis Intelligence Platform</div>
    <div class="hero-subtitle">Real-Time Triple-Ensemble Fraud & Wash Trading Array</div>
</div>
""", unsafe_allow_html=True)

# --- Database Integration ---
def fetch_data(project: str):
    try:
        conn = sqlite3.connect(DB_PATH)
        query = "SELECT timestamp, risk, confidence FROM trends WHERE project = ? ORDER BY timestamp ASC"
        df = pd.read_sql_query(query, conn, params=(project.lower(),))
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()

# --- UI Controls Sidebar ---
with st.sidebar:
    st.header("🎯 Target Acquisition")
    search_query = st.text_input("Enter Token / Project:", value="Ethereum", help="Supports precise crypto assets like PEPE, SHIB, Ethereum.")
    st.markdown("---")
    st.markdown("### ⚙️ Ensembles Active")
    st.markdown("🟢 **On-Chain:** XGBoost + Dexter Bounds")
    st.markdown("🟢 **Topology:** PyTorch GNN Limits")
    st.markdown("🟢 **Social NLP:** FinBERT Transfomers")
    st.markdown("---")
    
    st.markdown("### 🖥️ Core Diagnostics")
    st.markdown("""
    <div style="font-size: 0.85rem; color: #475569; padding: 10px; background: #f8fafc; border-radius: 8px; border: 1px solid #e2e8f0;">
        <div>📡 <b>RPC Socket:</b> DexScreener Live API</div>
        <div>💾 <b>State Engine:</b> SQLite (WAL Mode)</div>
        <div>⚡ <b>Compute:</b> Torch CPU Optimized</div>
        <div style="margin-top: 8px; font-weight: 700; color: #10b981;">● SYSTEM ONLINE & NOMINAL</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    run_btn = st.button("Execute Deep Analysis", use_container_width=True, type="primary")

# --- Core App Logic ---
if run_btn:
    with st.status(f"Scanning target network for '{search_query}'...", expanded=True) as status:
        st.write("Initializing ML Triple-Ensemble...")
        st.write("Pulling dynamic Web3 bindings via DexScreener...")
        st.write("Scraping real-time social footprints via Reddit...")
        
        try:
            res = requests.get(f"{API_URL}?project={search_query}")
            if res.status_code == 200:
                data = res.json()
                status.update(label="Target Acquisition Complete!", state="complete", expanded=False)
                
                # --- RESULTS HEADER ---
                st.markdown(f"<h2 style='text-align: center; margin-top: 10px; color: #1e293b;'>Intelligence Profile: {search_query.upper()}</h2>", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                
                # --- METRICS ROW ---
                c1, c2, c3, c4 = st.columns(4)
                
                risk_val = data['final_risk'] * 100
                conf_val = data['confidence'] * 100
                
                with c1:
                    r_col, r_class = ("#EF4444", "metric-card-risk") if risk_val > 70 else ("#F59E0B", "metric-card-warn") if risk_val > 40 else ("#10B981", "metric-card-safe")
                    st.markdown(f"<div class='metric-card {r_class}'><div class='metric-title'>Composite Threat</div><div class='metric-value' style='color:{r_col}'>{risk_val:.1f}%</div></div>", unsafe_allow_html=True)
                
                with c2:
                    st.markdown(f"<div class='metric-card metric-card-info'><div class='metric-title'>Data Confidence</div><div class='metric-value' style='color:#3B82F6'>{conf_val:.1f}%</div></div>", unsafe_allow_html=True)
                
                with c3:
                    trend = data['trend']
                    t_col, t_class = ("#EF4444", "metric-card-risk") if trend == "RISING" else ("#10B981", "metric-card-safe") if trend == "FALLING" else ("#64748b", "metric-card-gray")
                    trend_icon = "📈" if trend == "RISING" else "📉" if trend == "FALLING" else "➖"
                    st.markdown(f"<div class='metric-card {t_class}'><div class='metric-title'>Trajectory</div><div class='metric-value' style='color:{t_col}'>{trend_icon} {trend}</div></div>", unsafe_allow_html=True)
                
                with c4:
                    sev = data['severity']
                    s_col, s_class = ("#EF4444", "metric-card-risk") if sev == "CRITICAL" else ("#F59E0B", "metric-card-warn") if sev == "HIGH" else ("#3B82F6", "metric-card-info") if sev == "MODERATE" else ("#10B981", "metric-card-safe")
                    st.markdown(f"<div class='metric-card {s_class}'><div class='metric-title'>Severity Level</div><div class='metric-value' style='color:{s_col}'>{sev}</div></div>", unsafe_allow_html=True)
                
                # --- BREAKDOWN ROW ---
                st.markdown("<br>", unsafe_allow_html=True)
                col_chart, col_flags = st.columns([1.2, 1])
                
                with col_chart:
                    st.markdown("#### 🔭 Dimensional Threat Signature")
                    
                    # Internal sub-columns for Radar vs Bars
                    rad_col, bar_col = st.columns([1.3, 1])
                    
                    with rad_col:
                        fig_radar = go.Figure(data=go.Scatterpolar(
                          r=[data['onchain_risk']*100, data['offchain_risk']*100, risk_val],
                          theta=['Web3 Topology (DL)', 'Social Sentiment (NLP)', 'Unified Core'],
                          fill='toself',
                          line_color='#2563EB',
                          fillcolor='rgba(37, 99, 235, 0.15)',
                          marker=dict(size=8)
                        ))
                        fig_radar.update_layout(
                            polar=dict(
                                radialaxis=dict(visible=True, range=[0, 100], gridcolor="#e2e8f0", tickfont=dict(size=10)),
                                bgcolor="#ffffff",
                                angularaxis=dict(gridcolor="#e2e8f0")
                            ),
                            showlegend=False,
                            paper_bgcolor="rgba(0,0,0,0)", 
                            plot_bgcolor="rgba(0,0,0,0)",
                            margin=dict(l=30, r=30, t=20, b=20),
                            font=dict(color="#475569", size=11),
                            height=250
                        )
                        st.plotly_chart(fig_radar, use_container_width=True)

                    with bar_col:
                        st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
                        st.markdown("**Web3 Protocol Footprint**")
                        st.progress(data['onchain_risk'])
                        st.markdown(f"<small style='color: #64748b;'>{data['onchain_risk']*100:.1f}% DL/ML Bounding Risk</small>", unsafe_allow_html=True)
                        
                        st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
                        st.markdown("**Language Semantic Panic**")
                        st.progress(data['offchain_risk'])
                        st.markdown(f"<small style='color: #64748b;'>{data['offchain_risk']*100:.1f}% FinBERT Distillation</small>", unsafe_allow_html=True)

                with col_flags:
                    st.markdown("#### ⚡ Autonomous Directives")
                    
                    action = data['action']
                    bg_col, b_col, t_col = ("#FEF2F2", "#FCA5A5", "#991B1B") if sev in ["CRITICAL", "HIGH"] else ("#F0FDF4", "#86EFAC", "#166534")
                    st.markdown(f"""
                    <div style="padding: 18px; border-radius: 12px; background: {bg_col}; border: 1px solid {b_col}; margin-bottom: 25px;">
                        <span style="font-size: 0.85rem; font-weight: 700; color: {t_col}; text-transform: uppercase;">Primary Recommendation</span>
                        <p style="margin: 5px 0 0 0; font-size: 1.15rem; color: {t_col}; font-weight: 600;">{action}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("#### 🔍 Detected Footprints")
                    expl = data['explanation']
                    if expl:
                        for e in expl:
                            st.markdown(f"<div class='alert-flag'>⚠️ {e}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div class='safe-flag'>✅ No critical anomaly footprints detected. Data appears organically derived.</div>", unsafe_allow_html=True)
                
                # --- Under The Hood Expander ---
                with st.expander("🛠️ View Under-The-Hood Diagnostics & JSON Matrix"):
                    st.json(data)
                        
            else:
                status.update(label="API execution failed.", state="error")
                st.error("API returned an error. Ensure backend is running.")
        except Exception as e:
            status.update(label="Connection constraints failure.", state="error")
            st.error(f"Failed to connect to backend: {e}")

# --- Historical UI Plotting ---
st.markdown("<br><hr style='border-color: #e2e8f0; margin-bottom: 30px;'>", unsafe_allow_html=True)
st.markdown(f"### 📈 Temporal Domain Tracker: {search_query.upper()}")

df = fetch_data(search_query)

if not df.empty:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    c1, c2 = st.columns(2)
    
    with c1:
        fig1 = px.area(df, x="timestamp", y="risk", color_discrete_sequence=["#EF4444"])
        fig1.update_traces(fillcolor='rgba(239, 68, 68, 0.15)', line=dict(width=3))
        fig1.update_layout(
            title="Risk Volatility Tracker Array",
            paper_bgcolor="rgba(0,0,0,0)", 
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=True, gridcolor="#f1f5f9", title=""),
            yaxis=dict(showgrid=True, gridcolor="#f1f5f9", range=[0, 1.0], title="Composite Threat Risk"),
            font=dict(color="#475569")
        )
        st.plotly_chart(fig1, use_container_width=True)
        
    with c2:
        fig2 = px.line(df, x="timestamp", y="confidence", color_discrete_sequence=["#3B82F6"])
        fig2.update_traces(line=dict(width=3))
        fig2.update_layout(
            title="Extraction Confidence Timeline",
            paper_bgcolor="rgba(0,0,0,0)", 
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=True, gridcolor="#f1f5f9", title=""),
            yaxis=dict(showgrid=True, gridcolor="#f1f5f9", range=[0, 1.0], title="Confidence Factor"),
            font=dict(color="#475569")
        )
        st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("No historical block signatures recorded. Execute a deep analysis to establish a baseline boundary.", icon="📊")
