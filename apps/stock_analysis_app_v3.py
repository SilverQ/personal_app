import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
import FinanceDataReader as fdr

# --- Gemini API Configuration ---
# 실제 배포 시에는 st.secrets 또는 환경변수 사용을 권장합니다.
# GEMINI_API_key = os.environ.get("GEMINI_API_KEY")
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY" # 여기에 실제 API 키를 입력하세요.
genai.configure(api_key=GEMINI_API_KEY)

@st.cache_data(ttl=3600) # 1시간 동안 캐시하여 불필요한 API 호출 방지
def get_stock_info(ticker, date_str):
    """FinanceDataReader를 사용하여 특정 날짜의 주식 정보를 가져오는 함수"""
    try:
        # 최신 가격 정보 조회 (오늘 날짜 기준)
        df_price = fdr.DataReader(ticker, date_str)
        if df_price.empty:
            # 만약 오늘 데이터가 없다면(주말, 공휴일 등) 가장 최신 데이터 1개만 가져옴
            df_price = fdr.DataReader(ticker)
            if df_price.empty:
                st.error(f"'{ticker}'에 대한 가격 정보를 조회할 수 없습니다.")
                return None, None
        
        price = df_price.iloc[-1]['Close']

        # 상장주식수 조회하여 시가총액 계산
        df_listing = fdr.StockListing('KRX')
        listing_info = df_listing[df_listing['Code'] == ticker]
        
        if not listing_info.empty:
            listed_shares = listing_info['Stocks'].iloc[0]
            market_cap = listed_shares * price
        else:
            st.warning(f"상장 정보를 찾을 수 없어 시가총액을 계산할 수 없습니다: {ticker}")
            market_cap = None

        return price, market_cap
    except Exception as e:
        st.error(f"주식 정보 조회 중 오류 발생: {e}")
        return None, None

def generate_gemini_content(prompt, system_instruction):
    """Gemini API를 호출하여 콘텐츠를 생성하는 함수 (시스템 프롬프트 적용)"""
    try:
        model = genai.GenerativeModel(
            model_name='gemini-1.5-flash',
            system_instruction=system_instruction
        )
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        # 오류 내용을 Streamlit 화면에 직접 자세히 표시합니다.
        st.error("Gemini API 호출 중 오류가 발생했습니다. 자세한 내용은 아래와 같습니다.")
        st.exception(e)
        return "오류로 인해 분석 내용을 생성할 수 없습니다."

def main():
    st.set_page_config(layout="wide")
    
    today_str = pd.Timestamp.now().strftime('%Y-%m-%d')

    # --- Initialize session state ---
    if 'gemini_analysis' not in st.session_state:
        st.session_state.gemini_analysis = "사이드바의 'Gemini 최신 정보 분석' 버튼을 클릭하여 AI 분석을 시작하세요."
    if 'current_price' not in st.session_state:
        st.session_state.current_price = 230000
    if 'market_cap' not in st.session_state:
        st.session_state.market_cap = "정보 조회 필요"

    # --- Title Area ---
    title_col, info_col = st.columns([3, 1])
    with title_col:
        st.title("AI 기반 투자 분석 리포트")
    with info_col:
        info_html = f"""
        <div style='text-align: right;'>
        <b>조회 기준일:</b> {today_str}<br>
        <b>애널리스트:</b> Gemini 1.5 Flash
        </div>
        """
        st.markdown(info_html, unsafe_allow_html=True)

    st.divider()

    # --- Sidebar Inputs ---
    st.sidebar.header("분석 대상 기업")
    company_name = st.sidebar.text_input("기업명", "SK하이닉스")
    stock_code = st.sidebar.text_input("종목코드", "000660")

    if st.sidebar.button("📈 최신 시세 조회", help="네이버 금융에서 최신 종가와 시가총액 정보를 가져옵니다."):
        with st.spinner("최신 시세 정보를 조회 중입니다..."):
            price, market_cap = get_stock_info(stock_code, today_str)
            if price is not None:
                st.session_state.current_price = price
                if market_cap is not None:
                    st.session_state.market_cap = f"{market_cap / 1000000000000:,.1f}조 원"
                st.sidebar.success("시세 조회가 완료되었습니다.")
            else:
                st.sidebar.error("정보 조회에 실패했습니다.")

    st.sidebar.divider()
    
    if st.sidebar.button("✨ Gemini 최신 정보 분석", help="최신 뉴스와 데이터를 바탕으로 투자 포인트와 리스크 요인을 새로 분석합니다."):
        with st.spinner('Gemini가 최신 정보를 분석 중입니다... 잠시만 기다려주세요.'):
            system_prompt = "당신은 15년 경력의 유능한 대한민국 주식 전문 애널리스트입니다. 객관적인 데이터와 최신 정보에 기반하여 명확하고 간결하게 핵심을 전달합니다."
            user_prompt = f"**기업 분석 요청**\n- **분석 대상:** {company_name}({stock_code})\n- **요청 사항:** 최근 6개월간의 정보를 종합하여, 아래 형식에 맞춰 '긍정적 투자 포인트' 2가지와 '잠재적 리스크 요인' 2가지를 구체적인 근거와 함께 도출해주세요.\n\n**[결과 출력 형식]**\n### 긍정적 투자 포인트\n**1. [제목]**\n- [근거1]\n- [근거2]\n**2. [제목]**\n- [근거1]\n- [근거2]\n\n### 잠재적 리스크 요인\n**1. [제목]**\n- [근거1]\n- [근거2]\n**2. [제목]**\n- [근거1]\n- [근거2]"
            st.session_state.gemini_analysis = generate_gemini_content(user_prompt, system_prompt)

    st.sidebar.divider()

    # --- 1. 요약 (Executive Summary) ---
    st.header("1. 요약 (Executive Summary)")
    
    target_price = 283000  # Placeholder
    upside_potential = ((target_price / st.session_state.current_price) - 1) * 100 if st.session_state.current_price > 0 else 0

    summary_cols = st.columns(4)
    summary_cols[0].metric("투자의견", "BUY")
    summary_cols[1].metric("현재주가", f"{st.session_state.current_price:,.0f} 원")
    summary_cols[2].metric("목표주가", f"{target_price:,.0f} 원")
    summary_cols[3].metric("상승여력", f"{upside_potential:.2f} %")

    st.info(f"**핵심 투자 요약:**\n\n> **{company_name}**는 AI 시장 성장에 따른 HBM 수요 증가의 핵심 수혜주이며, 메모리 업황 개선에 따른 실적 턴어라운드가 기대됩니다.")
    st.divider()

    # --- Main Content in 2 Columns ---
    main_col1, main_col2 = st.columns(2)

    with main_col1:
        st.subheader("2. 기업 개요")
        st.text_input("회사명", company_name, disabled=True, key="company_name_display")
        st.text_input("티커", stock_code, disabled=True, key="stock_code_display")
        st.text_area("주요 사업", "메모리 반도체(DRAM, NAND Flash) 및 시스템 반도체(CIS 등) 제조 및 판매", disabled=True)
        st.text_input("시가총액", st.session_state.market_cap, disabled=True)

    with main_col2:
        st.subheader("3. Gemini 종합 분석")
        with st.container(border=True):
            st.markdown(st.session_state.gemini_analysis)

    st.divider()

    # --- 4. 실적 전망 (Earnings Forecast) ---
    st.header("4. 실적 전망 (Earnings Forecast)")
    data = {
        '(단위: 십억원)': ['매출액', '영업이익', '순이익', 'EPS (원)', 'BPS (원)', 'ROE (%)'],
        '2023A': [36229, -7730, -9138, -12544, 83836, -14.96],
        '2024E': [62345, 15678, 12456, 17100, 101000, 16.93],
        '2025E': [85789, 28910, 22500, 30890, 125000, 24.71]
    }
    df_forecast = pd.DataFrame(data).set_index('(단위: 십억원)')
    st.dataframe(df_forecast, use_container_width=True)
    st.caption("> 2024년, 2025년 실적은 시장 컨센서스를 바탕으로 한 추정치입니다.")
    st.divider()

    # --- 5. 가치평가 (Valuation) ---
    st.header("5. 가치평가 (Valuation)")
    st.write("본 리포트는 **PBR-ROE 모델**을 기반으로 목표주가를 산출했습니다.")

    st.sidebar.subheader("PBR-ROE 모델 변수")
    est_roe = st.sidebar.slider("예상 ROE (%)", 0.0, 50.0, 24.71, 0.1)
    cost_of_equity = st.sidebar.slider("자기자본비용 (Ke, %)", 5.0, 15.0, 9.0, 0.1)
    terminal_growth = st.sidebar.slider("영구성장률 (g, %)", 0.0, 5.0, 3.0, 0.1)
    st.sidebar.subheader("목표주가 산출 변수")
    est_bps = st.sidebar.number_input("예상 BPS (원)", value=125000)

    target_pbr = (est_roe - terminal_growth) / (cost_of_equity - terminal_growth) if (cost_of_equity - terminal_growth) != 0 else 0
    calculated_target_price = target_pbr * est_bps

    st.subheader("5.1. 목표 PBR 산출")
    val_col1, val_col2 = st.columns(2)
    with val_col1:
        st.markdown(f"- **(A) 예상 ROE:** `{est_roe:.2f} %`")
        st.markdown(f"- **(B) 자기자본비용 (Ke):** `{cost_of_equity:.2f} %`")
        st.markdown(f"- **(C) 영구성장률 (g):** `{terminal_growth:.2f} %`")
    with val_col2:
        st.success(f"**목표 PBR (배):** `{target_pbr:.2f}` 배")

    st.subheader("5.2. 목표주가 산출")
    val2_col1, val2_col2 = st.columns(2)
    with val2_col1:
        st.markdown(f"- **(D) 목표 PBR:** `{target_pbr:.2f}` 배")
        st.markdown(f"- **(E) 예상 BPS:** `{est_bps:,.0f}` 원")
    with val2_col2:
        st.success(f"**목표주가 (원):** `{calculated_target_price:,.0f}` 원")
    st.divider()    
    st.write("*본 보고서는 외부 출처로부터 얻은 정보에 기반하며, 정확성을 보장하지 않습니다. 투자 결정에 대한 최종 책임은 투자자 본인에게 있습니다.*")

if __name__ == "__main__":
    main()