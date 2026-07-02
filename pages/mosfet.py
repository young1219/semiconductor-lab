import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon as MplPolygon
import plotly.graph_objects as go
import requests

st.set_page_config(page_title="MOSFET SIMULATOR", page_icon="🔌", layout="wide")

st.markdown("""
<style>
    [data-testid="stSidebarUserContent"] { padding-top: 0rem !important; }
    [data-testid="stSidebarNav"] { display: none !important; }
    [data-testid="stSidebar"] .element-container,
    [data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p {
        margin-bottom: 0px !important; margin-top: 0px !important; }
    [data-testid="stSidebar"] h3 { font-size:0.95rem !important; margin-bottom:5px !important; margin-top:5px !important; }
    [data-testid="stSidebar"] hr { margin:6px 0 !important; }
    [data-testid="stSidebar"] .stSlider { margin-top:0px !important; padding-bottom:0px !important; margin-bottom:-10px !important; }
    [data-testid="stSidebar"] .stSelectbox { margin-top:-4px !important; margin-bottom:-4px !important; }
    [data-testid="stSidebar"] .stTextArea { margin-top:4px !important; margin-bottom:-4px !important; }
    [data-testid="stSidebar"] .stTextArea textarea { font-size:0.78rem !important; }
    .stat-card { background:#ffffff; border-radius:12px; padding:16px; border:1px solid #eaeaea; box-shadow:0px 4px 10px rgba(0,0,0,0.02); height:100%; }
    .stat-title { font-size:0.75rem; color:#64748b; font-weight:600; text-transform:uppercase; margin-bottom:4px; }
    .stat-label { font-size:0.7rem; color:#94a3b8; font-weight:600; margin-bottom:2px; }
    .stat-value { font-size:1.15rem; font-weight:700; color:#1e293b; }
    .section-header { font-size:1.25rem; font-weight:800; color:#334155; margin-top:0px; margin-bottom:12px; display:flex; align-items:center; gap:8px; }
    .block-container { padding-top:2.5rem !important; padding-bottom:1rem !important; }
</style>
""", unsafe_allow_html=True)

GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
GEMINI_URL = (f"https://generativelanguage.googleapis.com/v1beta/models/"
              f"gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}")

def call_gemini(prompt):
    # Gemini API 형식에 맞춰 1회성 구조(contents -> parts -> text)로 전달합니다.
    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
    try:
        resp = requests.post(GEMINI_URL, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            return "⏳ <b>[트래픽 초과]</b> 현재 동시에 많은 사용자가 이용 중이거나 무료 요청 한도를 초과했습니다. 잠시 후 다시 시도해 주세요."
        return f"❌ <b>[서버 오류]</b> API 요청 중 문제가 발생했습니다. (오류 코드: {e.response.status_code})"
    except Exception as e:
        return "❌ <b>[연결 오류]</b> AI 서버와 통신할 수 없습니다. 네트워크 상태를 확인해 주세요."

for key, default in [("vth_val", 1.0), ("vgs_val", 2.6), ("vds_val", 3.7)]:
    if key not in st.session_state:
        st.session_state[key] = default

with st.sidebar:
    if st.button("⬅ 홈으로 돌아가기", use_container_width=True):
        st.switch_page("app.py")
    st.markdown("### 🎛️ 제어 및 입력 패널")
    device = st.selectbox("소자 타입 선택", ["NMOS", "PMOS"])
    st.sidebar.divider()
    st.markdown("<span style='font-size:0.75rem;font-weight:700;color:#2c3e50;'>문턱 전압 |V_TH| (V)</span>", unsafe_allow_html=True)
    st.sidebar.write("")
    vth = st.slider("V_TH", 0.0, 2.0, value=float(st.session_state["vth_val"]), step=0.1, key="vth_slide", label_visibility="collapsed")
    st.session_state["vth_val"] = vth
    st.markdown("<span style='font-size:0.75rem;font-weight:700;color:#2c3e50;'>게이트 전압 V_GS (V)</span>", unsafe_allow_html=True)
    st.sidebar.write("")
    vgs = st.slider("V_GS", 0.0, 5.0, value=float(st.session_state["vgs_val"]), step=0.1, key="vgs_slide", label_visibility="collapsed")
    st.session_state["vgs_val"] = vgs
    st.markdown("<span style='font-size:0.75rem;font-weight:700;color:#2c3e50;'>드레인 전압 V_DS (V)</span>", unsafe_allow_html=True)
    st.sidebar.write("")
    vds = st.slider("V_DS", 0.0, 5.0, value=float(st.session_state["vds_val"]), step=0.1, key="vds_slide", label_visibility="collapsed")
    st.session_state["vds_val"] = vds
    st.sidebar.divider()
    st.markdown("<span style='font-size:0.8rem;font-weight:700;color:#1e293b;'>🤖 ASK AI</span>", unsafe_allow_html=True)
    user_question = st.text_area("", height=80, placeholder="e.g. 현재 전압 조건 상태에 대해 물리적으로 쉽게 설명해줘.", label_visibility="collapsed")
    
    # 레이아웃 정리를 위해 버튼 배치 조정 가능
    # 리셋 버튼을 없애고 AI 해설 보기 버튼이 가로를 모두 차지하도록 단일 버튼으로 변경합니다.
    ask_btn = st.button("🤖 AI 해설 보기", use_container_width=True, type="primary")

def calc_mosfet(device, vgs, vds, vth, Kn=1.0, Kp=1.0):
    if device == "NMOS":
        vgs_eff = vgs - vth
        vds_sat = max(vgs_eff, 0.0)
        if vgs_eff <= 0:
            region = "Cutoff";  id_mA = 0.0
        elif vds < vgs_eff:
            region = "Linear";  id_mA = round(Kn * (vgs_eff * vds - 0.5 * vds**2), 2)
        else:
            region = "Saturation"; id_mA = round(0.5 * Kn * vgs_eff**2, 2)
    else:
        vgs_real = -vgs; vds_real = -vds; vth_real = -vth
        vgs_eff  = vth_real - vgs_real
        vds_sat  = max(vgs_eff, 0.0)
        if vgs_real >= vth_real:
            region = "Cutoff";  id_mA = 0.0
        elif abs(vds_real) < vgs_eff:
            region = "Linear";  id_mA = round(Kp * (vgs_eff * abs(vds_real) - 0.5 * abs(vds_real)**2), 2)
        else:
            region = "Saturation"; id_mA = round(0.5 * Kp * vgs_eff**2, 2)
    return region, id_mA, vds_sat

region, id_mA, vds_sat = calc_mosfet(device, vgs, vds, vth)
region_kr   = {"Cutoff":"차단 영역","Linear":"선형 영역","Saturation":"포화 영역"}.get(region, region)
region_color= "#22c55e" if region=="Saturation" else "#f59e0b" if region=="Linear" else "#ef4444"
region_desc = {
    "Cutoff":     "V_GS < V_TH → 반전 채널 미형성 → 전류 차단 (OFF 스위치)",
    "Linear":     "V_DS < V_GS − V_TH → 채널이 저항처럼 동작 (트라이오드)",
    "Saturation": "V_DS ≥ V_GS − V_TH → 드레인 핀치오프 → 정전류원처럼 동작",
}.get(region, "")

st.markdown(f"""
<h1 style='text-align:left;font-size:2.2rem;font-weight:900;color:#1e293b;
           margin-top:0;padding-bottom:12px;border-bottom:1px solid #e2e8f0;margin-bottom:24px;'>
    🔌 {device} MOSFET SIMULATOR
</h1>""", unsafe_allow_html=True)

col_left, col_mid, col_right = st.columns([0.28, 0.46, 0.26], gap="medium")

with col_left:
    st.markdown("<div class='section-header'>📊 소자 상태</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class='stat-card' style='margin-bottom:24px;'>
        <div class='stat-title'>Operating Region</div>
        <div style='font-size:1.6rem;font-weight:800;color:{region_color};line-height:1.2;margin-bottom:4px;'>{region_kr}</div>
        <div style='font-size:0.9rem;color:{region_color};margin-bottom:18px;font-weight:600;'>({region})</div>
        <div style='display:grid;grid-template-columns:1fr 1fr;gap:16px;'>
            <div><div class='stat-label'>인가전압 |V_DS|</div><div class='stat-value'>{vds:.2f} V</div></div>
            <div><div class='stat-label'>드레인전류 |I_D|</div><div class='stat-value'>{id_mA:.2f} mA</div></div>
            <div><div class='stat-label'>게이트전압 |V_GS|</div><div class='stat-value'>{vgs:.2f} V</div></div>
            <div><div class='stat-label'>포화전압 V_DSAT</div><div class='stat-value'>{vds_sat:.2f} V</div></div>
        </div>
        <div style='margin-top:20px;padding:12px 14px;background:#f8fafc;
                    border-left:4px solid {region_color};border-radius:6px;
                    font-size:0.78rem;font-weight:700;color:#334155;line-height:1.45;'>
            <span style='color:{region_color}'>{region_desc}</span>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>📐 MOSFET 구조</div>", unsafe_allow_html=True)
    fig_struct, ax = plt.subplots(figsize=(5, 4.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 8.5); ax.axis("off")
    fig_struct.patch.set_facecolor('white')
    sub_color   = "#c8dff0" if device=="NMOS" else "#fce4d6"
    sub_edge    = "#5a8abf" if device=="NMOS" else "#e67e22"
    sub_text    = "p-Substrate" if device=="NMOS" else "n-Substrate"
    sub_tc      = "#2c5f8a" if device=="NMOS" else "#a04000"
    well_text  = "n+" if device=="NMOS" else "p+"
    well_color = "#4caf7d" if device=="NMOS" else "#9b59b6"
    well_edge  = "#2e7d52" if device=="NMOS" else "#8e44ad"
    carrier_color = "#e65100" if device=="NMOS" else "#8e44ad"
    ch_color   = "#66bb6a" if device=="NMOS" else "#d2b4de"
    ch_edge    = "#388e3c" if device=="NMOS" else "#af7ac5"
    ax.add_patch(patches.FancyBboxPatch((0.3,0.3),9.4,4.8,boxstyle="round,pad=0.1",fc=sub_color,ec=sub_edge,lw=1.5))
    ax.text(5,1.0,sub_text,ha="center",va="center",fontsize=10,color=sub_tc,fontstyle='italic')
    for (x0,label) in [(0.5,"S"),(7.2,"D")]:
        ax.add_patch(patches.FancyBboxPatch((x0,3.0),2.3,2.1,boxstyle="round,pad=0.05",fc=well_color,ec=well_edge,lw=1.5))
        cx=x0+1.15
        ax.text(cx,4.1,label,ha="center",va="center",fontsize=18,fontweight="bold",color="white")
        ax.text(cx,3.35,well_text,ha="center",va="center",fontsize=10,color="#ffffff")
    ax.add_patch(patches.Rectangle((2.8,5.1),4.4,0.4,fc="#e8e8e8",ec="#aaa",lw=1.0))
    ax.text(7.35,5.3,"SiO2",ha="left",va="center",fontsize=8,color="#9400D3")
    ax.add_patch(patches.FancyBboxPatch((2.8,5.5),4.4,0.75,boxstyle="round,pad=0.05",fc="#37474f",ec="#1a1a2e",lw=1.5))
    ax.text(5,5.88,"Gate (G)",ha="center",va="center",fontsize=10,fontweight="bold",color="white")
    GATE_X_START,GATE_X_END=2.8,7.2; GATE_LEN=GATE_X_END-GATE_X_START
    SIO2_BOTTOM,CH_THICK=5.1,0.5
    if region=="Saturation":
        ratio=float(np.clip(vds_sat/max(abs(vds),0.01),0.15,0.85))
        po_x=(GATE_X_START+GATE_LEN*ratio if device=="NMOS" else GATE_X_END-GATE_LEN*ratio)
        if device=="NMOS":
            tri_pts=np.array([[GATE_X_START,SIO2_BOTTOM],[po_x,SIO2_BOTTOM],[GATE_X_START,SIO2_BOTTOM-CH_THICK]])
            dep_rect=patches.Rectangle((po_x,SIO2_BOTTOM-CH_THICK),GATE_X_END-po_x,CH_THICK,fc="#dce8f5",ec="#5a8abf",linestyle='--',alpha=0.6)
            arr_start,arr_end=GATE_X_START+0.2,po_x-0.15
        else:
            tri_pts=np.array([[GATE_X_END,SIO2_BOTTOM],[po_x,SIO2_BOTTOM],[GATE_X_END,SIO2_BOTTOM-CH_THICK]])
            dep_rect=patches.Rectangle((GATE_X_START,SIO2_BOTTOM-CH_THICK),po_x-GATE_X_START,CH_THICK,fc="#fbeee6",ec="#e67e22",linestyle='--',alpha=0.6)
            arr_start,arr_end=GATE_X_END-0.2,po_x+0.15
        ax.add_patch(MplPolygon(tri_pts,closed=True,fc=ch_color,ec=ch_edge,lw=1.2,alpha=0.9,zorder=4))
        ax.add_patch(dep_rect)
        ax.plot(po_x,SIO2_BOTTOM,'ro',ms=7,zorder=10)
        ax.text(po_x,SIO2_BOTTOM-0.8,"Pinch-off",ha="center",fontsize=7,color="red",fontweight="bold")
        ax.annotate("",xy=(arr_end,SIO2_BOTTOM-CH_THICK*0.4),xytext=(arr_start,SIO2_BOTTOM-CH_THICK*0.4),
                    arrowprops=dict(arrowstyle='->',color=carrier_color,lw=1.4),zorder=6)
    elif region=="Linear":
        drain_thin=CH_THICK*(1.0-0.4*(vds/(vds_sat if vds_sat>0 else 1)))
        trap_pts=np.array([[GATE_X_START,SIO2_BOTTOM],[GATE_X_END,SIO2_BOTTOM],[GATE_X_END,SIO2_BOTTOM-drain_thin],[GATE_X_START,SIO2_BOTTOM-CH_THICK]])
        ax.add_patch(MplPolygon(trap_pts,closed=True,fc=ch_color,ec=ch_edge,lw=1.2,alpha=0.85,zorder=4))
    else:
        ax.text(5,SIO2_BOTTOM-0.35,"No Channel (Cutoff)",ha="center",va="top",fontsize=8,color="#dc3545",
                bbox=dict(boxstyle='round,pad=0.3',fc='#fff0f0',ec='#dc3545',alpha=0.85))
    ax.text(5,7.8,f"Applied: V_GS={vgs:.1f}V | V_DS={vds:.1f}V",ha="center",fontsize=8,color="#444")
    st.pyplot(fig_struct)
    plt.close(fig_struct)

with col_mid:
    st.markdown("<div class='section-header'>📈 특성 곡선 & 밴드 다이어그램</div>", unsafe_allow_html=True)

    v_ax = np.linspace(0, 5, 300)
    i_ax = [calc_mosfet(device, vgs, vd, vth)[1] for vd in v_ax]
    vgs_for_boundary = np.linspace(vth+0.01, 5.0, 300)
    sat_vds_pts, sat_id_pts = [], []
    for vg_ in vgs_for_boundary:
        vds_b = vg_ - vth
        id_b  = round(0.5*(vg_-vth)**2, 2)
        if 0 <= vds_b <= 5.0:
            sat_vds_pts.append(vds_b); sat_id_pts.append(id_b)
    fig_iv = go.Figure()
    fig_iv.add_trace(go.Scatter(x=sat_vds_pts,y=sat_id_pts,mode='lines',line=dict(color='#e74c3c',dash='dash',width=1.8),name="Saturation Boundary (V_DS = V_GS − V_TH)"))
    fig_iv.add_trace(go.Scatter(x=list(v_ax),y=i_ax,mode='lines',line=dict(color='#1a5276',width=2.5),name=f"V_GS = {vgs:.1f} V"))
    fig_iv.add_trace(go.Scatter(x=[vds],y=[id_mA],mode='markers',marker=dict(color='#e74c3c',size=11,line=dict(color='white',width=1.5)),name="Operating Point"))
    fig_iv.update_layout(
        height=320,margin=dict(l=10,r=10,t=40,b=10),
        xaxis_title="|V_DS| (V)" if device=="PMOS" else "V_DS (V)",
        yaxis_title="I_D (mA)",
        xaxis=dict(range=[0,5]),yaxis=dict(rangemode='tozero'),
        legend=dict(yanchor="top",y=0.99,xanchor="left",x=0.01,bgcolor="rgba(255,255,255,0.85)",bordercolor="rgba(128,128,128,0.3)",borderwidth=1,font=dict(size=9)),
        plot_bgcolor='white',
        title=dict(text="I-V Characteristic Curve",font=dict(size=12,color="#64748b"),x=0.5,y=0.95,xanchor='center'))
    fig_iv.update_xaxes(showgrid=True,gridcolor='#f1f5f9')
    fig_iv.update_yaxes(showgrid=True,gridcolor='#f1f5f9')
    st.plotly_chart(fig_iv,use_container_width=True,theme="streamlit")

    Eg       = 1.12
    SCALE    = 0.22     
    MAX_DROP = 1.10     
    DELTA    = 0.12     
    Y_OFF    = 2.2      

    abs_vds   = abs(vds)
    overdrive = abs(vgs) - abs(vth)                  
    V         = float(min(abs_vds * SCALE, MAX_DROP))

    if region == "Cutoff":
        h_barrier = float(np.clip(0.25 + 0.45 * max(-overdrive, 0.0), 0.25, 0.75))
    else:
        h_barrier = 0.0

    n_exp = 3.0 if region == "Saturation" else (1.0 if region == "Linear" else 6.0)

    x_src = np.linspace(0.0, 1.0, 50)
    x_ch  = np.linspace(1.0, 2.0, 140)
    x_drn = np.linspace(2.0, 3.0, 50)
    t     = np.linspace(0.0, 1.0, len(x_ch))
    drop  = t ** n_exp          
    if region == "Cutoff":
        drop = np.zeros_like(t)

    s         = np.clip(t / 0.35, 0.0, 1.0)
    barrier_p = s * s * (3 - 2 * s)        

    td      = np.linspace(0.0, 1.0, len(x_drn))
    sd      = td * td * (3 - 2 * td)        

    if device == "NMOS":
        EF_src, EF_drn = Y_OFF, Y_OFF - V            
        ec_src_lvl = EF_src + DELTA                  
        ec_drn_lvl = EF_drn + DELTA                  
        ec_ch  = ec_src_lvl + h_barrier * barrier_p - V * drop
        ec_drn = ec_ch[-1] + (ec_drn_lvl - ec_ch[-1]) * sd
        ec_all = np.concatenate([np.full_like(x_src, ec_src_lvl), ec_ch, ec_drn])
        ev_all = ec_all - Eg
        po_band = ec_all                              
    else:  
        EF_src, EF_drn = Y_OFF, Y_OFF + V            
        ev_src_lvl = EF_src - DELTA                  
        ev_drn_lvl = EF_drn - DELTA                  
        ev_ch  = ev_src_lvl - h_barrier * barrier_p + V * drop
        ev_drn = ev_ch[-1] + (ev_drn_lvl - ev_ch[-1]) * sd
        ev_all = np.concatenate([np.full_like(x_src, ev_src_lvl), ev_ch, ev_drn])
        ec_all = ev_all + Eg
        po_band = ev_all                              

    x_all = np.concatenate([x_src, x_ch, x_drn])
    ci    = len(x_src) + len(x_ch) // 2              
    ch_mid_y = float((ec_all[ci] + ev_all[ci]) / 2)

    fig_band = go.Figure()
    fig_band.add_trace(go.Scatter(x=list(x_all), y=list(ec_all), mode='lines',
                                  line=dict(color='#e74c3c', width=2.5), name="E<sub>c</sub>"))
    fig_band.add_trace(go.Scatter(x=list(x_all), y=list(ev_all), mode='lines',
                                  line=dict(color='#2980b9', width=2.5), name="E<sub>v</sub>"))
    fig_band.add_trace(go.Scatter(x=[0.0, 1.0], y=[EF_src, EF_src], mode='lines',
                                  line=dict(color='purple', width=1.8, dash='dot'),
                                  name="E<sub>F</sub> (Source)"))
    fig_band.add_trace(go.Scatter(x=[2.0, 3.0], y=[EF_drn, EF_drn], mode='lines',
                                  line=dict(color='purple', width=1.8, dash='dot'),
                                  name="E<sub>F</sub> (Drain)"))

    fig_band.add_annotation(x=0.15, y=ec_all[0], ay=ev_all[0], axref='x', ayref='y',
                            xref='x', yref='y', arrowhead=2, arrowsize=1,
                            arrowwidth=1.2, arrowcolor='gray', ax=0.15)
    fig_band.add_annotation(x=0.15, y=ev_all[0], ay=ec_all[0], axref='x', ayref='y',
                            xref='x', yref='y', arrowhead=2, arrowsize=1,
                            arrowwidth=1.2, arrowcolor='gray', ax=0.15)
    fig_band.add_annotation(x=0.22, y=(ec_all[0] + ev_all[0]) / 2, text=f"Eg={Eg}eV",
                            showarrow=False, font=dict(size=9, color='gray'), xanchor='left')

    if region == "Saturation":
        fig_band.add_annotation(x=1.5, y=ch_mid_y, text="Inversion Layer<br>(Saturation)",
                                showarrow=False, font=dict(size=9, color='#27ae60'),
                                bgcolor='#eafaf1', bordercolor='#27ae60', borderwidth=1)
        po_i = len(x_src) + int(0.85 * len(x_ch))
        fig_band.add_annotation(x=x_all[po_i], y=po_band[po_i], text="Pinch-off",
                                showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.2,
                                arrowcolor="#e74c3c", ax=16, ay=-26,
                                font=dict(size=8, color="#e74c3c"))
    elif region == "Linear":
        fig_band.add_annotation(x=1.5, y=ch_mid_y, text="Channel Formed<br>(Linear)",
                                showarrow=False, font=dict(size=9, color='#f39c12'),
                                bgcolor='#fef9e7', bordercolor='#f39c12', borderwidth=1)
    else:
        fig_band.add_annotation(x=1.5, y=ch_mid_y, text="No Channel<br>(Cutoff)",
                                showarrow=False, font=dict(size=9, color='#ef4444'),
                                bgcolor='#fff0f0', bordercolor='#ef4444', borderwidth=1)

    fig_band.update_layout(
        height=320, margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(tickvals=[0.5, 1.5, 2.5], ticktext=["Source", "Channel", "Drain"],
                   tickfont=dict(size=10), showgrid=True, gridcolor='#f1f5f9'),
        yaxis=dict(title="Energy (eV)", title_font=dict(size=10),
                   showgrid=True, gridcolor='#f1f5f9'),
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99,
                    bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="rgba(128,128,128,0.3)", borderwidth=1, font=dict(size=9)),
        plot_bgcolor='white',
        title=dict(text=f"Energy Band Diagram ({device})",
                   font=dict(size=12, color="#64748b"), x=0.5, y=0.95, xanchor='center'))
    st.plotly_chart(fig_band, use_container_width=True, theme="streamlit")


# 변경 포인트 3: AI 해설 영역 수정 (과거 대화 축소 및 누적)
with col_right:
    st.markdown("<div class='section-header'>🤖 AI 해설</div>", unsafe_allow_html=True)
    if "gemini_response" not in st.session_state:
        st.session_state.gemini_response = ""
        
    if ask_btn:
        question = (user_question.strip() if user_question.strip()
                    else f"현재 {device} MOSFET 조건에 대해 물리적으로 쉽게 설명해줘.")
        
        # 이번 턴에 주입할 개별 프롬프트 생성
        full_prompt = f"""
[역할]
당신은 전자정보공학부 학부생 전담 AI 튜터입니다.
청중: 물리전자, 반도체소자, 전자회로, 응용회로실험 등의 전공 과목을 듣는 대학생으로, MOSFET 동작 영역(Cutoff/Linear/Saturation)
용어는 배웠지만 '왜' 핀치오프가 일어나는지는 직관이 아직 부족한 상태입니다.
교재에 실린 일반 이론을 반복하는 것이 아니라,
지금 이 화면에 나타난 수치와 그래프를 출발점으로 삼아 이야기하세요.

[현재 시뮬레이터 상태]
- 소자 종류 : {device} MOSFET
- 인가 전압 : V_GS = {vgs:.2f}V / V_DS = {vds:.2f}V / V_TH = {vth:.2f}V
- 동작 영역 : {region}
- 드레인 전류 : I_D = {id_mA:.3f} mA
- 포화 기준 전압 : V_DS,sat = {vds_sat:.2f}V (= V_GS − V_TH)

[답변 작성 지침]
1. 첫 문장은 반드시 현재 수치를 직접 언급하며 시작할 것.
   ("지금 V_GS가 {vgs:.1f}V로 설정되어 있고..." 형태)
   "안녕하세요", "좋은 질문이에요" 같은 인사말 절대 금지.
2. 마크다운은 과하지 않게, 아래 4개 흐름을 **굵은 소제목**으로 구분해서 작성:
   → 왜 지금 동작 영역이 {region}으로 분류되는지
     (V_DS와 V_DS,sat의 대소관계, V_GS와 V_TH의 대소관계로 근거 제시)
   → 그 결과 채널(반전층)이 어떤 모양으로 형성되어 있는지, 또는 왜 없는지
     (왼쪽 MOSFET 구조도에서 지금 보이는 채널 모양과 연결해서 언급)
   → 전류 I_D = {id_mA:.3f}mA가 왜 이 값인지 캐리어 이동 관점에서 설명
   → 오른쪽 I-V 그래프에서 현재 동작점(파란 점)이 곡선의 어느 위치에 있는지 한 문장으로 짚기
3. 전체 4~6문장. 전체 답변이 장황해지지 않도록 주의.
4. 전공 용어는 처음 등장할 때만 괄호로 영문 병기 (예: 반전층(inversion layer)).
5. 마지막 줄은 아래 형식으로 능동 질문 1개와 정답 접기(Toggle)를 포함할 것:
   "🤔 직접 생각해보기: [슬라이더 조작 유도 질문]"
   
   그 바로 아래 줄에 아래 HTML 문법을 줄바꿈(Enter) 없이 '한 줄로 바짝 붙여서' 출력할 것 (태그 사이에 빈 줄이나 공백이 있으면 디자인이 깨집니다):
   <details><summary>💡 정답 및 해설 확인하기</summary><div style="padding:12px; background:#f8fafc; border-radius:6px; margin-top:4px; color:#334155;">[여기에 질문에 대한 명확한 정답과 이유를 1~2문장으로 작성]</div></details>
6. 한국어로만 답변하고 영어 문장, 프랑스어, 한자 등 다른 언어를 절대 섞지 말 것.
7. 숫자들이 서로 모순되지 않는지 속으로 검산한 뒤(검산 과정 자체는 출력하지 말 것), 결과만 자연스럽게 설명에 반영할 것.
8. 비유를 한 개 사용할 것. 수도꼭지, 도로 정체, 좁아지는 터널 등 학생이 즉시 그림을 그릴 수 있는 일상 비유여야 함.
   비유를 먼저 제시하고, 그 다음 실제 물리 현상으로 연결할 것.
9. 학생 질문에 우선적으로 대답할 것.

[학생 질문]
"{question}"
"""
        # 기록 보관 및 슬라이싱 코드를 전부 지우고, 단발성으로 prompt만 넘겨 결과를 받습니다.
        with st.spinner("AI가 분석 중입니다...")):
            response_text = call_gemini(full_prompt)
            st.session_state.gemini_response = response_text
            
    if st.session_state.gemini_response:
        st.markdown(f"""
        <div style='background:#ffffff;padding:16px;border-radius:10px;
                    border:1px solid #e2e8f0;font-size:0.85rem;color:#1e293b;
                    line-height:1.6;white-space:pre-wrap;min-height:140px;
                    box-shadow:0px 4px 6px rgba(0,0,0,0.02);'>{st.session_state.gemini_response}</div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background:#f0f9ff;padding:16px;border-radius:10px;
                    border:1px solid #bae6fd;font-size:0.88rem;font-weight:600;
                    color:#0369a1;display:flex;align-items:flex-start;gap:8px;'>
            <span>👉</span>
            <span>왼쪽 패널에서 설정을 마치고 [AI 실시간 해설 보기] 버튼을 눌러보세요.</span>
        </div>""", unsafe_allow_html=True)
