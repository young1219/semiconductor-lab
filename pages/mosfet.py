import streamlit as st
import google.generativeai as genai
import plotly.graph_objects as go
import numpy as np

# 1. 페이지 레이아웃 설정
st.set_page_config(layout="wide")
st.title("⚡ AI 반도체 설계 직관 보조 툴 (Gemini AX)")

# 2. API 키 보안 로드 및 엔진 주입
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
else:
    st.sidebar.warning("🔑 Gemini API 키가 Secrets에 등록되지 않아 AI 인사이트 기능이 제한됩니다.")

# 🧠 [메모리 기능 추가] 이전 대화 기록을 저장할 히스토리 세션 초기화
if "mosfet_chat_history" not in st.session_state:
    st.session_state.mosfet_history = []

# 3. 좌측 사이드바: MOSFET 컨트롤러 및 질문 입력창 배치
st.sidebar.header("🎛️ MOSFET 소자 및 전압 조절")
mosfet_type = st.sidebar.radio("소자 타입 선택 (Type)", ["NMOS", "PMOS"])

V_th_default = 0.7 if mosfet_type == "NMOS" else -0.7
V_gs_default = 3.5 if mosfet_type == "NMOS" else -3.5
V_ds_default = 2.2 if mosfet_type == "NMOS" else -2.2

if mosfet_type == "NMOS":
    V_th = st.sidebar.slider("문턱 전압 (V_th)", 0.0, 2.0, V_th_default, 0.1)
    V_gs = st.sidebar.slider("게이트 전압 (V_gs)", 0.0, 5.0, V_gs_default, 0.1)
    V_ds = st.sidebar.slider("드레인 전압 (V_ds)", 0.0, 5.0, V_ds_default, 0.1)
    K_n = 0.5
else:
    V_th = st.sidebar.slider("문턱 전압 (V_th)", -2.0, 0.0, V_th_default, 0.1)
    V_gs = st.sidebar.slider("게이트 전압 (V_gs)", -5.0, 0.0, V_gs_default, 0.1)
    V_ds = st.sidebar.slider("드레인 전압 (V_ds)", -5.0, 0.0, V_ds_default, 0.1)
    K_n = 0.5

st.sidebar.markdown("---")
st.sidebar.header("💬 AI에게 질문하기")

# 4. 백엔드 알고리즘: 동작 영역 판정 및 전류 계산
v_gs_abs = abs(V_gs)
v_th_abs = abs(V_th)
v_ds_abs = abs(V_ds)

if v_gs_abs < v_th_abs:
    region = "차단 영역 (Cut-off)"
    I_d = 0.0
    condition_text = f"{mosfet_type}: |V_gs| < |V_th| 상태입니다."
elif v_ds_abs < (v_gs_abs - v_th_abs):
    region = "선형 영역 (Linear / Triode)"
    I_d = K_n * (2 * (v_gs_abs - v_th_abs) * v_ds_abs - v_ds_abs**2)
    condition_text = f"{mosfet_type}: |V_gs| >= |V_th| 이고 |V_ds| < |V_gs| - |V_th| 상태입니다."
else:
    region = "포화 영역 (Saturation)"
    I_d = K_n * ((v_gs_abs - v_th_abs)**2)
    condition_text = f"{mosfet_type}: |V_gs| >= |V_th| 이고 |V_ds| >= |V_gs| - |V_th| 상태입니다."

# 5. 화면 분할
col1, col2 = st.columns([1.1, 0.9])

with col1:
    st.subheader("📊 실시간 소자 특성 시각화")
    st.info(f"**현재 판정 상태:** {mosfet_type} {region} (I_d = {I_d:.3f} mA)")
    
    fig_channel = go.Figure()
    if mosfet_type == "NMOS":
        sub_text = "<b>P-Substrate</b>"
        diff_text = "<b>n+</b>"
        channel_color = "rgba(0, 255, 255, 0.8)" 
        channel_label = "N-Channel (Electrons)"
    else:
        sub_text = "<b>N-Substrate</b>"
        diff_text = "<b>p+</b>"
        channel_color = "rgba(255, 0, 255, 0.8)" 
        channel_label = "P-Channel (Holes)"

    fig_channel.add_shape(type="rect", x0=0, y0=0, x1=10, y1=4, fillcolor="#D3D3D3", line=dict(color="gray")) 
    fig_channel.add_shape(type="rect", x0=0.5, y0=2.5, x1=3.0, y1=4.0, fillcolor="#FF7F50", line=dict(color="chocolate")) 
    fig_channel.add_shape(type="rect", x0=7.0, y0=2.5, x1=9.5, y1=4.0, fillcolor="#FF7F50", line=dict(color="chocolate")) 
    fig_channel.add_shape(type="rect", x0=3.0, y0=4.0, x1=7.0, y1=4.3, fillcolor="#FFD700", line=dict(color="goldenrod")) 
    fig_channel.add_shape(type="rect", x0=3.0, y0=4.3, x1=7.0, y1=4.8, fillcolor="#555555", line=dict(color="#333333")) 

    labels = [
        dict(x=1.75, y=3.25, text=diff_text, showarrow=False, font=dict(size=14, color="white")),
        dict(x=8.25, y=3.25, text=diff_text, showarrow=False, font=dict(size=14, color="white")),
        dict(x=1.75, y=4.4, text="<b>Source</b>", showarrow=False, font=dict(size=13, color="black")),
        dict(x=8.25, y=4.4, text="<b>Drain</b>", showarrow=False, font=dict(size=13, color="black")),
        dict(x=5.0, y=5.2, text=f"<b>Gate ({mosfet_type})</b>", showarrow=False, font=dict(size=13, color="black")),
        dict(x=5.0, y=4.15, text="Gate - Insulator", showarrow=False, font=dict(size=10, color="black")),
        dict(x=5.0, y=1.2, text=sub_text, showarrow=False, font=dict(size=14, color="black")),
    ]
    for lbl in labels: fig_channel.add_annotation(lbl)
        
    y_max = 4.0
    if region == "차단 영역 (Cut-off)":
        fig_channel.add_annotation(x=5.0, y=3.7, text=f"<i>No Channel Formed (|V_gs| < |V_th|)</i>", showarrow=False, font=dict(size=12, color="red"))
    elif region == "선형 영역 (Linear / Triode)":
        thick_source = (v_gs_abs - v_th_abs) * 0.15 + 0.05
        thick_drain = thick_source * (1 - (v_ds_abs / (v_gs_abs - v_th_abs)))
        if thick_drain < 0.05: thick_drain = 0.05
        
        fig_channel.add_trace(go.Scatter(
            x=[3.0, 7.0, 7.0, 3.0, 3.0], 
            y=[y_max, y_max, y_max - thick_drain, y_max - thick_source, y_max], 
            fill="toself", fillcolor=channel_color, mode='lines', line=dict(width=0), showlegend=False
        ))
    else:
        thick_source = (v_gs_abs - v_th_abs) * 0.15 + 0.05
        pinch_x = 7.0 - (v_ds_abs - (v_gs_abs - v_th_abs)) * 0.3
        if pinch_x < 5.0: pinch_x = 5.0
        
        fig_channel.add_trace(go.Scatter(
            x=[3.0, pinch_x, 3.0, 3.0], 
            y=[y_max, y_max, y_max - thick_source, y_max], 
            fill="toself", fillcolor=channel_color, mode='lines', line=dict(width=0), showlegend=False
        ))
        
        fig_channel.add_trace(go.Scatter(
            x=[pinch_x, 7.0], y=[y_max - 0.02, y_max - 0.02],
            mode="lines", line=dict(color="red" if mosfet_type=="NMOS" else "purple", width=3, dash="dot"),
            showlegend=False
        ))
        
        fig_channel.add_trace(go.Scatter(
            x=[pinch_x], y=[4.0], mode="markers", 
            marker=dict(color="red", size=18, symbol="circle-open", line=dict(width=2)), 
            name="Pinch-off"
        ))
        fig_channel.add_annotation(x=pinch_x, y=3.4, text="<b>Pinch-off</b>", showarrow=True, arrowhead=2, arrowcolor="red", font=dict(size=11, color="red"))

    fig_channel.update_layout(title=f"<b>🎨 {mosfet_type} 내부 물리 구조 및 채널 동적 매핑</b>", xaxis=dict(visible=False, range=[-0.5, 10.5]), yaxis=dict(visible=False, range=[-0.5, 6.0]), height=350, margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
    st.plotly_chart(fig_channel, use_container_width=True)

    # I-V 특성 곡선 패밀리 파트
    v_ds_mesh = np.linspace(0, 5, 100)
    i_d_curve = []
    for v_ds_each in v_ds_mesh:
        if v_gs_abs < v_th_abs: 
            i_d_curve.append(0.0)
        elif v_ds_each < (v_gs_abs - v_th_abs): 
            i_d_curve.append(K_n * (2 * (v_gs_abs - v_th_abs) * v_ds_each - v_ds_each**2))
        else: 
            i_d_curve.append(K_n * ((v_gs_abs - v_th_abs)**2))
            
    fig_iv = go.Figure()
    fig_iv.add_trace(go.Scatter(x=v_ds_mesh, y=i_d_curve, mode='lines', name='I_D', line=dict(color='blue' if mosfet_type=="NMOS" else 'purple', width=3)))
    fig_iv.add_trace(go.Scatter(x=[v_ds_abs], y=[I_d], mode='markers', name='Q-point', marker=dict(color='red', size=14, symbol='circle')))
    fig_iv.update_layout(title=f"📈 {mosfet_type} 드레인 특성 곡선 (I_D - |V_DS|)", xaxis_title="Absolute Drain-Source Voltage |V_DS| [V]", yaxis_title="Drain Current (I_D) [mA]", xaxis=dict(range=[0, 5.5]), yaxis=dict(range=[-0.5, 5]), height=300, margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
    st.plotly_chart(fig_iv, use_container_width=True)

# 🤖 우측 대화형 AI 분석창 파트 (기록 기억 기능 탑재 완료)
with col2:
    st.subheader("💬 AI 반도체 엔지니어 아키텍트 채팅")
    
    # 🧼 대화 초기화 버튼 배치
    if st.button("🗑️ 대화 기록 초기화"):
        st.session_state.mosfet_history = []
        st.rerun()
        
    st.markdown("---")

    # 📥 챗봇 스타일 전용 컨테이너 생성 및 기존 기록 출력
    chat_container = st.container(height=420)
    with chat_container:
        if len(st.session_state.mosfet_history) == 0:
            st.caption(f"현재 {mosfet_type} 동작 조건에 대해 궁금한 점을 아래에 입력해 대화를 시작해 보세요!")
        for role, text in st.session_state.mosfet_history:
            if role == "user":
                st.chat_message("user").write(text)
            else:
                st.chat_message("assistant").write(text)

    # ⌨️ 스트림릿 최신 챗 인터페이스 바인딩
    if user_input := st.chat_input("질문을 입력하고 엔터를 누르세요 (예: 방금 말한 핀치오프 현상에 대해 더 설명해줘)"):
        if "GEMINI_API_KEY" in st.secrets:
            # 1. 화면에 유저 질문 즉시 출력 및 세션 저장
            chat_container.chat_message("user").write(user_input)
            st.session_state.mosfet_history.append(("user", user_input))
            
            # 2. 페르소나 및 하드웨어 연동 시스템 콘텍스트 정의
            system_context = (
                f"[현재 하드웨어 스펙 및 바이어스 세팅 정보]\n"
                f"- 소자 종류: {mosfet_type} MOSFET\n"
                f"- 인가 조건: V_GS={V_gs:.2f}V, V_DS={V_ds:.2f}V, V_TH={V_th:.2f}V\n"
                f"- 판정 모드: {region}\n"
                f"- 드레인 전류: I_D={I_d:.3f}mA\n\n"
                f"[미션]\n"
                f"너는 세계 최고 수준의 AI 반도체 및 회로 설계 엔지니어 교수다. "
                f"인사말 절대 없이 다이렉트로 연산 결론부터 대답해라. "
                f"유저가 이전 대화에 이어 질문하면 히스토리를 연동해서 맥락에 맞게 '한국어 경어체'로 친절하게 마크다운 설명해라.\n\n"
            )
            
            # 3. 이전 대화 기록 전체 묶기
            full_prompt = system_context + "--- [이전 대화 기록 시작] ---\n"
            for role, text in st.session_state.mosfet_history[:-1]:
                full_prompt += f"{'학생' if role=='user' else '교수'}: {text}\n"
            full_prompt += f"--- [이전 대화 기록 끝] ---\n\n마지막 학생 질문: {user_input}"
            
            # 4. 연산 및 답변 출력
            with st.spinner("교수님이 답변을 연산 중입니다..."):
                try:
                    model = genai.GenerativeModel('gemini-2.5-flash')
                    response = model.generate_content(full_prompt)
                    
                    # 화면 출력 및 히스토리 기록 추가 후 새로고침
                    chat_container.chat_message("assistant").write(response.text)
                    st.session_state.mosfet_history.append(("assistant", response.text))
                    st.rerun()
                except Exception as e:
                    st.error(f"연산 엔진 내부 오류 발생: {e}")
        else:
            st.error("Secrets 금고에 GEMINI_API_KEY가 세팅되어 있지 않습니다.")
