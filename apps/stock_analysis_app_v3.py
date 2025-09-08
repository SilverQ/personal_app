import streamlit as st
import pandas as pd
from google import genai
import os
import configparser
import re
import requests
import json
import time

# --- API Client Initialization ---
# 스크립트 파일의 위치를 기준으로 config.ini 경로를 절대 경로로 계산합니다.
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)
config_path = os.path.join(ROOT_DIR, 'config.ini')

# Config 파서 초기화
config = configparser.ConfigParser()
config.read(config_path)

# Gemini API 키 설정
gemini_api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if not gemini_api_key:
    try:
        gemini_api_key = config.get('GEMINI_API_KEY', 'key', fallback=None)
    except (configparser.NoSectionError, configparser.NoOptionError):
        gemini_api_key = None

client = None
if gemini_api_key:
    try:
        client = genai.Client(api_key=gemini_api_key)
    except Exception as e:
        st.error(f"Gemini API 클라이언트 초기화 중 오류가 발생했습니다: {e}")
else:
    st.warning("Gemini API 키를 찾을 수 없어 AI 분석 기능이 제한됩니다. 환경 변수 또는 config.ini 파일을 확인해주세요.")

# Kiwoom API 키 설정
try:
    KIWOOM_APP_KEY = config.get('KIWOOM_API', 'appkey')
    KIWOOM_APP_SECRET = config.get('KIWOOM_API', 'secretkey')
    KIWOOM_API_MODE = config.get('KIWOOM_API', 'mode', fallback='mock')
except (configparser.NoSectionError, configparser.NoOptionError):
    KIWOOM_APP_KEY, KIWOOM_APP_SECRET, KIWOOM_API_MODE = None, None, 'mock'
    st.warning("Kiwoom API 키를 찾을 수 없어 시세 조회가 제한됩니다. config.ini 파일을 확인해주세요.")

# Kiwoom API URL 설정
if KIWOOM_API_MODE == 'real':
    KIWOOM_BASE_URL = "https://api.kiwoom.com"
else:
    KIWOOM_BASE_URL = "https://mockapi.kiwoom.com"


@st.cache_data(ttl=86400) # 24시간 동안 캐시
def get_stock_list():
    """apps/stock_list.csv 파일에서 전체 상장 종목 리스트를 가져오는 함수"""
    try:
        stock_list_path = os.path.join(APP_DIR, 'stock_list.csv')
        df_listing = pd.read_csv(stock_list_path, dtype={'code': str, 'name': str})
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

def get_kiwoom_token():
    """키움 API 접근 토큰을 발급받고 세션에 저장하는 함수"""
    if not KIWOOM_APP_KEY or not KIWOOM_APP_SECRET:
        st.error("키움 API 키가 설정되지 않았습니다. config.ini 파일을 확인하세요.")
        return None

    if 'kiwoom_token' in st.session_state and st.session_state.kiwoom_token_expires_at > time.time():
        return st.session_state.kiwoom_token

    url = f"{KIWOOM_BASE_URL}/oauth2/token"
    headers = {"Content-Type": "application/json"}
    data = {
        "grant_type": "client_credentials",
        "appkey": KIWOOM_APP_KEY,
        "secretkey": KIWOOM_APP_SECRET
    }
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        token_data = response.json()

        # 키움 API는 'token' 이라는 키로 토큰을 반환합니다.
        if 'token' not in token_data:
            st.error("키움 API 토큰 발급 실패: 응답에 'token'이 없습니다.")
            st.json(token_data)
            return None

        st.session_state.kiwoom_token = token_data['token']
        
        # 만료 시간 처리 (expires_dt 또는 expires_in 사용)
        if 'expires_dt' in token_data:
            try:
                expires_ts = pd.to_datetime(token_data['expires_dt'], format='%Y%m%d%H%M%S').timestamp()
                st.session_state.kiwoom_token_expires_at = expires_ts - 60 # 1분 버퍼
            except ValueError:
                # 파싱 실패 시 기본값으로 1시간 설정
                st.session_state.kiwoom_token_expires_at = time.time() + 3600 - 60
        else:
            expires_in = int(token_data.get('expires_in', 3600)) # 기본 1시간
            st.session_state.kiwoom_token_expires_at = time.time() + expires_in - 60
            
        return st.session_state.kiwoom_token
    except requests.exceptions.RequestException as e:
        st.error(f"키움 API 토큰 발급 중 오류 발생: {e}")
        if e.response:
            st.error(f"응답 내용: {e.response.text}")
        return None
    except json.JSONDecodeError:
        st.error(f"키움 API 토큰 응답이 JSON 형식이 아닙니다. 응답: {response.text}")
        return None

def get_kiwoom_stock_info(stock_code):
    """키움 API를 사용하여 주식의 상세 정보를 가져오는 함수"""
    token = get_kiwoom_token()
    if not token:
        return {}

    url = f"{KIWOOM_BASE_URL}/api/dostk/stkinfo"
    headers = {
        "Content-Type": "application/json;charset=UTF-8",
        "authorization": f"Bearer {token}",
        "appkey": KIWOOM_APP_KEY,
        "appsecret": KIWOOM_APP_SECRET,
        "api-id": "ka10001" # 주식기본정보조회
    }
    params = {"stk_cd": stock_code}

    try:
        response = requests.post(url, headers=headers, json=params)
        response.raise_for_status()
        data = response.json()

        # 키움 API는 return_code가 0이면 성공입니다.
        if data.get('return_code') != 0:
            st.error(f"키움 API 오류: {data.get('return_msg', '상세 메시지 없음')}")
            st.json(data)
            return {}

        # +, - 기호를 제거하고 숫자 타입으로 변환하는 헬퍼 함수
        def clean_value(value_str):
            if isinstance(value_str, str) and value_str:
                try:
                    return float(value_str.replace('+', '').replace('-', ''))
                except ValueError:
                    return 0.0
            elif isinstance(value_str, (int, float)):
                return float(value_str)
            return 0.0

        # 성공 응답에서 직접 데이터를 파싱합니다.
        info = {
            'price': clean_value(data.get('cur_prc', 0)),
            'market_cap': int(clean_value(data.get('mac', 0)) * 100000000), # 'mac'은 억원 단위
            'per': clean_value(data.get('per', 0)),
            'pbr': clean_value(data.get('pbr', 0)),
            'eps': clean_value(data.get('eps', 0)),
            'bps': clean_value(data.get('bps', 0)),
            'roe': clean_value(data.get('roe', 0)),
            'high_52w': clean_value(data.get('250hgst', 0)),
            'low_52w': clean_value(data.get('250lwst', 0)),
        }
        return info
    except requests.exceptions.RequestException as e:
        st.error(f"키움 API (stkinfo) 호출 중 오류 발생: {e}")
        if e.response:
            st.error(f"응답 내용: {e.response.text}")
        return {}
    except (ValueError, TypeError, json.JSONDecodeError) as e:
        st.error(f"API 응답 데이터 처리 중 오류 발생: {e}")
        return {}

def generate_gemini_content(prompt, system_instruction):
    """Gemini API를 호출하여 콘텐츠를 생성하는 함수"""
    if client is None:
        return "오류: Gemini 클라이언트가 초기화되지 않았습니다."
    try:
        combined_prompt = f"{system_instruction}\n\n{prompt}"
        response = client.models.generate_content(model='models/gemini-1.5-flash', contents=combined_prompt)
        return response.text
    except Exception as e:
        st.error("Gemini API 호출 중 오류가 발생했습니다.")
        st.exception(e)
        return "오류로 인해 분석 내용을 생성할 수 없습니다."

def get_empty_forecast_df():
    """실적 전망 테이블을 위한 빈 데이터프레임을 생성하는 함수"""
    data = {
        '(단위: 십억원)': ['매출액', '영업이익', '순이익', 'EPS (원)', 'BPS (원)', 'ROE (%)'],
        '2023A': [0.0] * 6, '2024E': [0.0] * 6, '2025E': [0.0] * 6
    }
    return pd.DataFrame(data).set_index('(단위: 십억원)').astype(float)

def reset_states_on_stock_change():
    """사용자가 새 기업을 선택했을 때 관련 세션 상태를 초기화하는 콜백 함수"""
    st.session_state.gemini_analysis = "상단 설정에서 기업 정보를 입력하고 'Gemini 최신 정보 분석' 버튼을 클릭하여 AI 분석을 시작하세요."
    st.session_state.main_business = "-"
    st.session_state.investment_summary = "-"
    st.session_state.kiwoom_data = {}
    st.session_state.df_forecast = get_empty_forecast_df()

def main():
    st.set_page_config(layout="wide")
    today_str = pd.Timestamp.now().strftime('%Y-%m-%d')

    # --- Robust Session State Initialization ---
    # 각 세션 상태 변수가 존재하는지 확인하고, 없으면 초기화합니다.
    if "gemini_analysis" not in st.session_state:
        st.session_state.gemini_analysis = "상단 설정에서 기업 정보를 입력하고 'Gemini 최신 정보 분석' 버튼을 클릭하여 AI 분석을 시작하세요."
    if "main_business" not in st.session_state:
        st.session_state.main_business = "-"
    if "investment_summary" not in st.session_state:
        st.session_state.investment_summary = "-"
    if "kiwoom_data" not in st.session_state:
        st.session_state.kiwoom_data = {}
    if "df_forecast" not in st.session_state:
        st.session_state.df_forecast = get_empty_forecast_df()
    if "kiwoom_token" not in st.session_state:
        st.session_state.kiwoom_token = None
    if "kiwoom_token_expires_at" not in st.session_state:
        st.session_state.kiwoom_token_expires_at = 0

    title_col, info_col = st.columns([3, 1])
    with title_col:
        st.title("AI 기반 투자 분석 리포트")
    with info_col:
        st.markdown(f"<div style='text-align: right;'><b>조회 기준일:</b> {today_str}<br><b>애널리스트:</b> Gemini 1.5 Flash</div>", unsafe_allow_html=True)

    st.divider()

    with st.expander("⚙️ 분석 설정 (기업, 모델 변수 등)", expanded=True):
        col1, col2, col3 = st.columns([2, 2, 1])

        with col1:
            st.markdown("**분석 대상**")
            df_listing = get_stock_list()
            
            if not df_listing.empty:
                df_listing['display'] = df_listing['name'] + ' (' + df_listing['code'] + ')'
                stock_options = df_listing['display'].tolist()
                default_index = stock_options.index("SK하이닉스 (000660)") if "SK하이닉스 (000660)" in stock_options else 0
                
                selected_stock = st.selectbox("기업 선택", stock_options, index=default_index, help="기업 변경 시 분석 내용은 초기화됩니다.", on_change=reset_states_on_stock_change, key='selected_stock', label_visibility="collapsed")
                match = re.match(r"(.+) \((.+)\)", selected_stock)
                company_name, stock_code = match.groups() if match else ("", "")
            else:
                st.warning("종목 리스트를 불러오지 못했습니다.")
                company_name = st.text_input("기업명", "SK하이닉스")
                stock_code = st.text_input("종목코드", "000660")

            btn_cols = st.columns(2)
            if btn_cols[0].button("📈 정보 조회", help="키움 API를 통해 최신 시세와 주요 투자 지표를 가져옵니다.", use_container_width=True):
                with st.spinner("키움 API에서 최신 정보를 조회 중입니다..."):
                    kiwoom_data = get_kiwoom_stock_info(stock_code)
                    if kiwoom_data:
                        st.session_state.kiwoom_data = kiwoom_data
                        df_new = st.session_state.df_forecast.copy()
                        df_new.loc['EPS (원)', '2023A'] = kiwoom_data.get('eps', 0)
                        df_new.loc['BPS (원)', '2023A'] = kiwoom_data.get('bps', 0)
                        df_new.loc['ROE (%)', '2023A'] = kiwoom_data.get('roe', 0)
                        st.session_state.df_forecast = df_new
                        st.success("정보 조회가 완료되었습니다.")
                    else:
                        st.error("정보 조회에 실패했습니다.")

            if btn_cols[1].button("✨ AI 분석", help="최신 뉴스와 데이터를 바탕으로 투자 포인트와 리스크 요인을 새로 분석합니다.", use_container_width=True):
                with st.spinner('Gemini가 최신 정보를 분석 중입니다...'):
                    system_prompt = "당신은 15년 경력의 유능한 대한민국 주식 전문 애널리스트입니다. 객관적인 데이터와 최신 정보에 기반하여 명확하고 간결하게 핵심을 전달합니다."
                    user_prompt = f'''**기업 분석 요청**
- **분석 대상:** {company_name}({stock_code})
- **요청 사항:**
  1. 이 기업의 **주요 사업**에 대해 한국어로 2-3문장으로 요약해주세요.
  2. 이 기업에 대한 **핵심 투자 요약**을 강점과 약점을 포함하여 한국어로 3줄 이내로 작성해주세요.
  3. 최근 6개월간의 정보를 종합하여, 아래 형식에 맞춰 '긍정적 투자 포인트' 2가지와 '잠재적 리스크 요인' 2가지를 구체적인 근거와 함께 한국어로 도출해주세요.

**[결과 출력 형식]**
### 주요 사업
[내용]

### 핵심 투자 요약
[내용]

### 긍정적 투자 포인트
**1. [제목]**
- [근거]
**2. [제목]**
- [근거]

### 잠재적 리스크 요인
**1. [제목]**
- [근거]
**2. [제목]**
- [근거]'''
                    full_response = generate_gemini_content(user_prompt, system_prompt)
                    try:
                        parts = full_response.split('###')
                        st.session_state.main_business = parts[1].replace('주요 사업', '').strip()
                        st.session_state.investment_summary = parts[2].replace('핵심 투자 요약', '').strip()
                        st.session_state.gemini_analysis = "###" + "###".join(parts[3:])
                    except Exception:
                        st.session_state.main_business = "-"
                        st.session_state.investment_summary = "-"
                        st.session_state.gemini_analysis = f"**오류: Gemini 응답 처리 중 문제가 발생했습니다.**\n\n{full_response}"

        with col2:
            st.markdown("**가치평가 모델**")
            default_roe = st.session_state.kiwoom_data.get('roe', 10.0)
            est_roe = st.slider("예상 ROE (%)", 0.0, 50.0, default_roe, 0.1, help="'최신 정보 조회' 시점의 ROE가 기본값으로 설정됩니다.")
            cost_of_equity = st.slider("자기자본비용 (Ke, %)", 5.0, 15.0, 9.0, 0.1)
            terminal_growth = st.slider("영구성장률 (g, %)", 0.0, 5.0, 3.0, 0.1)

        with col3:
            st.markdown("**목표주가 변수**")
            default_bps = st.session_state.kiwoom_data.get('bps', 150000)
            est_bps = st.number_input("예상 BPS (원)", value=int(default_bps), help="'최신 정보 조회' 시점의 BPS가 기본값으로 설정됩니다.")

    target_pbr = (est_roe - terminal_growth) / (cost_of_equity - terminal_growth) if (cost_of_equity - terminal_growth) != 0 else 0
    calculated_target_price = target_pbr * est_bps

    st.divider()

    st.header("1. 요약 (Executive Summary)")
    current_price = st.session_state.kiwoom_data.get('price', 0)
    upside_potential = ((calculated_target_price / current_price) - 1) * 100 if current_price > 0 else 0.0
    
    if upside_potential > 15: investment_opinion = "매수 (Buy)"
    elif upside_potential > -5: investment_opinion = "중립 (Neutral)"
    else: investment_opinion = "매도 (Sell)"
    if current_price == 0: investment_opinion = "-"

    summary_cols = st.columns(4)
    summary_cols[0].metric("투자의견", investment_opinion)
    summary_cols[1].metric("현재주가", f"{current_price:,.0f} 원" if current_price else "N/A")
    summary_cols[2].metric("목표주가", f"{calculated_target_price:,.0f} 원")
    summary_cols[3].metric("상승여력", f"{upside_potential:.2f} %")

    st.info(f"**핵심 투자 요약:**\n\n> {st.session_state.investment_summary}")
    st.divider()

    main_col1, main_col2 = st.columns(2)
    with main_col1:
        st.subheader("2. 기업 개요")
        kiwoom_data = st.session_state.kiwoom_data
        market_cap = kiwoom_data.get('market_cap', 0)
        market_cap_display = f"{market_cap / 100000000:,.0f} 억원" if market_cap > 0 else "N/A"
        
        st.text_input("회사명", company_name, disabled=True)
        st.text_input("티커", stock_code, disabled=True)
        st.text_area("주요 사업", st.session_state.main_business, disabled=True)
        st.text_input("시가총액", market_cap_display, disabled=True)
        
        overview_cols = st.columns(2)
        overview_cols[0].metric("PER", f"{kiwoom_data.get('per', 0):.2f} 배")
        overview_cols[1].metric("PBR", f"{kiwoom_data.get('pbr', 0):.2f} 배")
        overview_cols[0].metric("52주 최고", f"{kiwoom_data.get('high_52w', 0):,.0f} 원")
        overview_cols[1].metric("52주 최저", f"{kiwoom_data.get('low_52w', 0):,.0f} 원")

    with main_col2:
        st.subheader("3. Gemini 종합 분석")
        with st.container(border=True):
            st.markdown(st.session_state.gemini_analysis)

    st.divider()

    st.header("4. 실적 전망 (Earnings Forecast)")
    st.caption("아래 표의 데이터를 직접 수정하여 목표주가 계산에 실시간으로 반영할 수 있습니다. '최신 정보 조회' 시 2023A 데이터가 업데이트됩니다.")
    edited_df = st.data_editor(st.session_state.df_forecast, use_container_width=True)
    st.session_state.df_forecast = edited_df
    
    st.caption("> 2024년, 2025년 실적은 시장 컨센서스 또는 사용자 추정치를 바탕으로 합니다.")
    st.divider()

    st.header("5. 가치평가 (Valuation)")
    st.write("본 리포트는 **PBR-ROE 모델**을 기반으로 목표주가를 산출했습니다.")
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
