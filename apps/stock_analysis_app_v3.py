import streamlit as st
import pandas as pd
from google import genai
import os
import FinanceDataReader as fdr
import configparser
import re

# --- Gemini API Client Initialization ---
# 1. 환경 변수에서 API 키를 먼저 시도합니다 (GEMINI_API_KEY 또는 GOOGLE_API_KEY).
api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

# 2. 환경 변수에 키가 없으면 config.ini 파일에서 읽기를 시도합니다.
if not api_key:
    try:
        config = configparser.ConfigParser()
        # 스크립트 파일의 위치를 기준으로 config.ini 경로를 절대 경로로 계산합니다.
        APP_DIR = os.path.dirname(os.path.abspath(__file__))
        ROOT_DIR = os.path.dirname(APP_DIR)
        config_path = os.path.join(ROOT_DIR, 'config.ini')
        config.read(config_path)
        api_key = config.get('GEMINI_API_KEY', 'key', fallback=None)
    except Exception:
        # config.ini 파일이 없거나 읽기 오류가 발생해도 앱 실행은 계속됩니다.
        pass

# 3. 최종적으로 얻은 키로 클라이언트를 초기화합니다.
client = None
if api_key:
    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        st.error(f"Gemini API 클라이언트 초기화 중 오류가 발생했습니다: {e}")
else:
    # 키를 찾지 못했을 경우, 앱 실행 시 경고 메시지를 한 번만 표시합니다.
    st.warning("Gemini API 키를 찾을 수 없어 AI 분석 기능이 제한됩니다. 환경 변수 또는 config.ini 파일을 확인해주세요.")

@st.cache_data(ttl=86400) # 24시간 동안 캐시
def get_stock_list():
    """apps/stock_list.csv 파일에서 전체 상장 종목 리스트를 가져오는 함수"""
    try:
        # 스크립트 파일의 위치를 기준으로 stock_list.csv 경로를 절대 경로로 계산합니다.
        APP_DIR = os.path.dirname(os.path.abspath(__file__))
        stock_list_path = os.path.join(APP_DIR, 'stock_list.csv')
        df_listing = pd.read_csv(stock_list_path, dtype={'code': str, 'name': str})
        # 필요한 컬럼이 모두 있는지 확인
        if 'name' not in df_listing.columns or 'code' not in df_listing.columns:
            st.error("'apps/stock_list.csv' 파일에 'name' 또는 'code' 컬럼이 없습니다.")
            return pd.DataFrame()
    except FileNotFoundError:
        st.error("'apps/stock_list.csv' 파일을 찾을 수 없습니다.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"종목 리스트 로딩 중 오류 발생: {e}")
        return pd.DataFrame()
    return df_listing

@st.cache_data(ttl=86400) # 24시간 동안 캐시
def get_fdr_stock_listing():
    """FinanceDataReader를 사용하여 KRX 전체 상장 종목 리스트를 가져오는 함수 (시가총액 계산용)"""
    # 이 함수는 캐시되어 앱 실행 중 한 번만 호출됩니다.
    return fdr.StockListing('KRX')

@st.cache_data(ttl=3600) # 1시간 동안 캐시하여 불필요한 API 호출 방지
def get_stock_info(ticker, date_str):
    """FinanceDataReader를 사용하여 특정 날짜의 주식 정보와 시가총액을 가져오는 함수"""
    try:
        # 1. 최신 가격 정보 조회
        df_price = fdr.DataReader(ticker, date_str)
        if df_price.empty:
            df_price = fdr.DataReader(ticker)
            if df_price.empty:
                st.error(f"'{ticker}'에 대한 가격 정보를 조회할 수 없습니다.")
                return None, None
        
        price = df_price.iloc[-1]['Close']

        # 2. 상장 주식 수 조회하여 시가총액 계산
        df_listing = get_fdr_stock_listing()
        # FinanceDataReader의 'Code' 컬럼을 사용하여 ticker를 찾습니다.
        listing_info = df_listing[df_listing['Code'] == ticker]
        
        if not listing_info.empty:
            # 'Stocks' 컬럼에서 상장 주식 수를 가져옵니다.
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
    if client is None:
        return "오류: Gemini 클라이언트가 초기화되지 않았습니다. API 키 설정을 확인하세요."
    try:
        # 구버전 라이브러리와의 호환성을 위해 시스템 프롬프트와 사용자 프롬프트를 결합합니다.
        combined_prompt = f"{system_instruction}\n\n{prompt}"
        
        response = client.models.generate_content(
            model='models/gemini-1.5-flash',
            contents=combined_prompt
        )
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
        st.session_state.gemini_analysis = "상단 설정에서 기업 정보를 입력하고 'Gemini 최신 정보 분석' 버튼을 클릭하여 AI 분석을 시작하세요."
    if 'main_business' not in st.session_state:
        st.session_state.main_business = "-"
    if 'investment_summary' not in st.session_state:
        st.session_state.investment_summary = "-"
    if 'current_price' not in st.session_state:
        st.session_state.current_price = None
    if 'market_cap' not in st.session_state:
        st.session_state.market_cap = 0
    if 'df_forecast' not in st.session_state:
        # 실적 전망 테이블 초기화 (SK하이닉스 예시 데이터)
        data = {
            '(단위: 십억원)': ['매출액', '영업이익', '순이익', 'EPS (원)', 'BPS (원)', 'ROE (%)'],
            '2023A': [32766, -7730, -9138, -7437, 119897, -6.0],
            '2024E': [107461, 21843, 15635, 12724, 135321, 9.4],
            '2025E': [129765, 28742, 20567, 16738, 152059, 11.0]
        }
        st.session_state.df_forecast = pd.DataFrame(data).set_index('(단위: 십억원)')

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

    # --- Inputs Expander ---
    with st.expander("⚙️ 분석 설정 (기업, 모델 변수 등)", expanded=True):
        input_col1, input_col2 = st.columns(2)

        with input_col1:
            st.subheader("분석 대상 기업")
            
            df_listing = get_stock_list()
            
            # 'name'과 'code' 컬럼이 있는지, 데이터가 비어있지 않은지 확인
            if not df_listing.empty:
                # "기업명 (종목코드)" 형식의 리스트 생성
                df_listing['display'] = df_listing['name'] + ' (' + df_listing['code'] + ')'
                stock_options = df_listing['display'].tolist()
                
                # 기본 선택값 설정 (예: SK하이닉스)
                default_selection = "SK하이닉스 (000660)"
                try:
                    default_index = stock_options.index(default_selection)
                except ValueError:
                    default_index = 0
                
                selected_stock = st.selectbox(
                    "기업 선택",
                    stock_options,
                    index=default_index,
                    help="분석할 기업을 선택하세요."
                )

                # 선택된 문자열에서 기업명과 종목코드 분리
                match = re.match(r"(.+) \((.+)\)", selected_stock)
                if match:
                    company_name, stock_code = match.groups()
                else:
                    company_name, stock_code = "", "" # Fallback
            else:
                # df_listing이 비어있거나 필요한 컬럼이 없는 경우, 기존의 text_input 방식을 fallback으로 사용
                st.warning("종목 리스트를 불러오지 못했습니다. 기업명과 종목코드를 직접 입력해주세요.")
                company_name = st.text_input("기업명", "SK하이닉스")
                stock_code = st.text_input("종목코드", "000660")

            if st.button("📈 최신 시세 조회", help="최신 종가와 시가총액 정보를 가져옵니다."):
                with st.spinner("최신 시세 정보를 조회 중입니다..."):
                    price, market_cap = get_stock_info(stock_code, today_str)
                    st.session_state.current_price = price if price is not None else None
                    st.session_state.market_cap = market_cap if market_cap is not None else 0
                    if price is not None:
                        st.success("시세 조회가 완료되었습니다.")
                    else:
                        st.error("정보 조회에 실패했습니다.")

            if st.button("✨ Gemini 최신 정보 분석", help="최신 뉴스와 데이터를 바탕으로 투자 포인트와 리스크 요인을 새로 분석합니다."):
                with st.spinner('Gemini가 최신 정보를 분석 중입니다... 잠시만 기다려주세요.'):
                    system_prompt = "당신은 15년 경력의 유능한 대한민국 주식 전문 애널리스트입니다. 객관적인 데이터와 최신 정보에 기반하여 명확하고 간결하게 핵심을 전달합니다. 요청받은 모든 항목에 대해 반드시 한국어로 답변해야 합니다."
                    user_prompt = f'''**기업 분석 요청**
- **분석 대상:** {company_name}({stock_code})
- **요청 사항:**
  1. 이 기업의 **주요 사업**에 대해 한국어로 2-3문장으로 요약해주세요.
  2. 이 기업에 대한 **핵심 투자 요약**을 강점과 약점을 포함하여 한국어로 3줄 이내로 작성해주세요.
  3. 최근 6개월간의 정보를 종합하여, 아래 형식에 맞춰 '긍정적 투자 포인트' 2가지와 '잠재적 리스크 요인' 2가지를 구체적인 근거와 함께 한국어로 도출해주세요.

**[결과 출력 형식]**
### 주요 사업
[여기에 주요 사업 내용 작성]

### 핵심 투자 요약
[여기에 핵심 투자 요약 내용 작성]

### 긍정적 투자 포인트
**1. [제목]**
- [근거1]
- [근거2]
**2. [제목]**
- [근거1]
- [근거2]

### 잠재적 리스크 요인
**1. [제목]**
- [근거1]
- [근거2]
**2. [제목]**
- [근거1]
- [근거2]'''
                    
                    full_response = generate_gemini_content(user_prompt, system_prompt)
                    
                    # 응답 파싱 로직
                    try:
                        # '###'를 기준으로 응답을 분리합니다.
                        parts = full_response.split('###')
                        
                        # 각 섹션의 내용을 추출합니다. strip()으로 공백을 제거합니다.
                        st.session_state.main_business = parts[1].replace('주요 사업', '').strip()
                        st.session_state.investment_summary = parts[2].replace('핵심 투자 요약', '').strip()
                        
                        # '긍정적 투자 포인트'와 '잠재적 리스크 요인' 부분을 합쳐서 저장합니다.
                        st.session_state.gemini_analysis = "###" + "###".join(parts[3:])

                    except Exception as e:
                        # 파싱 실패 시, 전체 응답을 보여주고 오류 메시지를 기록합니다.
                        st.session_state.main_business = "-"
                        st.session_state.investment_summary = "-"
                        st.session_state.gemini_analysis = f"**오류: Gemini 응답을 처리하는 중 문제가 발생했습니다.**\n\n{full_response}"

        with input_col2:
            st.subheader("PBR-ROE 모델 변수")
            
            # 실적 전망 테이블의 2025년 추정치를 기본값으로 사용
            try:
                default_roe = float(st.session_state.df_forecast.loc['ROE (%)', '2025E'])
                default_bps = int(st.session_state.df_forecast.loc['BPS (원)', '2025E'])
            except (ValueError, KeyError):
                # 테이블 로드 실패 시 사용할 안전한 기본값
                default_roe = 10.0
                default_bps = 150000

            est_roe = st.slider("예상 ROE (%)", 0.0, 50.0, default_roe, 0.1, help="실적 전망 테이블의 2025년 ROE(%)가 기본값으로 설정됩니다.")
            cost_of_equity = st.slider("자기자본비용 (Ke, %)", 5.0, 15.0, 9.0, 0.1)
            terminal_growth = st.slider("영구성장률 (g, %)", 0.0, 5.0, 3.0, 0.1)
            
            st.subheader("목표주가 산출 변수")
            est_bps = st.number_input("예상 BPS (원)", value=default_bps, help="실적 전망 테이블의 2025년 BPS(원)가 기본값으로 설정됩니다.")

    # --- Calculation ---
    target_pbr = (est_roe - terminal_growth) / (cost_of_equity - terminal_growth) if (cost_of_equity - terminal_growth) != 0 else 0
    calculated_target_price = target_pbr * est_bps

    st.divider()

    # --- 1. 요약 (Executive Summary) ---
    st.header("1. 요약 (Executive Summary)")
    
    upside_potential = 0.0
    # 현재가가 유효한 경우에만 상승여력 계산
    if st.session_state.current_price and st.session_state.current_price > 0 and calculated_target_price > 0:
        upside_potential = ((calculated_target_price / st.session_state.current_price) - 1) * 100
    
    # 투자의견 결정 로직
    if upside_potential > 15:
        investment_opinion = "매수 (Buy)"
    elif upside_potential > -5:
        investment_opinion = "중립 (Neutral)"
    else:
        investment_opinion = "매도 (Sell)"
    
    # 현재가가 없으면 투자의견 보류
    if not st.session_state.current_price or st.session_state.current_price == 0:
        investment_opinion = "-"
        upside_potential = 0.0

    summary_cols = st.columns(4)
    summary_cols[0].metric("투자의견", investment_opinion)
    summary_cols[1].metric("현재주가", f"{st.session_state.current_price:,.0f} 원" if st.session_state.current_price else "N/A")
    summary_cols[2].metric("목표주가", f"{calculated_target_price:,.0f} 원")
    summary_cols[3].metric("상승여력", f"{upside_potential:.2f} %")

    st.info(f"**핵심 투자 요약:**\n\n> {st.session_state.investment_summary}")
    st.divider()

    # --- Main Content in 2 Columns ---
    main_col1, main_col2 = st.columns(2)

    with main_col1:
        st.subheader("2. 기업 개요")
        st.text_input("회사명", company_name, disabled=True, key="company_name_display")
        st.text_input("티커", stock_code, disabled=True, key="stock_code_display")
        st.text_area("주요 사업", "-", disabled=True)
        market_cap_display = f"{st.session_state.market_cap / 1000000000000:,.1f}조 원" if st.session_state.market_cap > 0 else "0 원"
        st.text_input("시가총액", market_cap_display, disabled=True)

    with main_col2:
        st.subheader("3. Gemini 종합 분석")
        with st.container(border=True):
            st.markdown(st.session_state.gemini_analysis)

    st.divider()

    # --- 4. 실적 전망 (Earnings Forecast) ---
    st.header("4. 실적 전망 (Earnings Forecast)")
    st.caption("아래 표의 데이터를 직접 수정하여 목표주가 계산에 실시간으로 반영할 수 있습니다.")
    
    # st.data_editor를 사용하여 사용자가 데이터를 직접 수정할 수 있도록 함
    edited_df = st.data_editor(st.session_state.df_forecast, use_container_width=True)
    st.session_state.df_forecast = edited_df # 수정된 내용을 다시 세션 상태에 저장
    
    st.caption("> 2024년, 2025년 실적은 시장 컨센서스 또는 사용자 추정치를 바탕으로 합니다.")
    st.divider()

    # --- 5. 가치평가 (Valuation) ---
    st.header("5. 가치평가 (Valuation)")
    st.write("본 리포트는 **PBR-ROE 모델**을 기반으로 목표주가를 산출했습니다.")

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