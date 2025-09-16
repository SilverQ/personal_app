import streamlit as st
import pandas as pd
from google import genai
import os
import configparser
import re
import requests
import json
import time
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Constants and Paths ---
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)

# --- Object-Oriented Handlers ---

class ConfigManager:
    """Manages application configuration from config.ini."""
    def __init__(self, root_dir):
        self.config_path = os.path.join(root_dir, 'config.ini')
        self.config = configparser.ConfigParser()
        self.config.read(self.config_path)

    def get_gemini_keys(self):
        """Retrieves a list of Gemini API keys from config or environment variables."""
        # Try environment variables first (comma-separated)
        keys_str = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if keys_str:
            return [key.strip() for key in keys_str.split(',') if key.strip()]

        # If not in env vars, try config.ini
        try:
            # Use items() to get (key, value) pairs directly
            keys = [value for _, value in self.config.items('GEMINI_API_KEY')]
            # Filter out any empty values
            return [key for key in keys if key]
        except configparser.NoSectionError:
            return []

    def get_kiwoom_config(self):
        """Retrieves Kiwoom API configuration."""
        try:
            app_key = self.config.get('KIWOOM_API', 'appkey')
            app_secret = self.config.get('KIWOOM_API', 'secretkey')
            mode = self.config.get('KIWOOM_API', 'mode', fallback='mock')
            base_url = "https://api.kiwoom.com" if mode == 'real' else "https://mockapi.kiwoom.com"
        except (configparser.NoSectionError, configparser.NoOptionError):
            st.warning("Kiwoom API 키를 찾을 수 없어 시세 조회가 제한됩니다. config.ini 파일을 확인해주세요.")
            return None, None, 'mock', 'https://mockapi.kiwoom.com'
        return app_key, app_secret, mode, base_url

class GeminiAPIHandler:
    """Handles interactions with the Gemini API with key rotation and model fallback."""
    def __init__(self, api_keys):
        self.api_keys = api_keys
        self.current_key_index = 0
        self.client = None
        self.primary_model = 'gemini-1.5-pro'
        self.fallback_model = 'gemini-1.5-flash'

        if not self.api_keys:
            st.warning("Gemini API 키를 찾을 수 없어 AI 분석 기능이 제한됩니다. 환경 변수 또는 config.ini 파일을 확인해주세요.")
        else:
            self._initialize_client()

    def _initialize_client(self):
        """Initializes the Gemini client with the current API key."""
        if self.current_key_index < len(self.api_keys):
            current_key = self.api_keys[self.current_key_index]
            try:
                self.client = genai.Client(api_key=current_key)
                # st.info(f"Gemini 클라이언트를 API Key #{self.current_key_index + 1}로 초기화했습니다.")
                return True
            except Exception as e:
                st.error(f"API Key #{self.current_key_index + 1}로 Gemini 클라이언트 초기화 중 오류: {e}")
                self.client = None
                return self.try_next_key() # Try next key if initialization fails
        else:
            # This case is hit when all keys have failed initialization
            if len(self.api_keys) > 0:
                st.error("사용 가능한 모든 Gemini API 키로 클라이언트를 초기화하는 데 실패했습니다.")
            self.client = None
            return False

    def try_next_key(self):
        """Switches to the next API key and re-initializes the client."""
        self.current_key_index += 1
        if self.current_key_index < len(self.api_keys):
            st.warning(f"API Key #{self.current_key_index}의 사용량이 소진되었거나 오류가 발생했습니다. 다음 키(#{self.current_key_index + 1})로 전환합니다.")
            return self._initialize_client()
        else:
            self.client = None # No more keys
            return False

    def generate_content(self, prompt, system_instruction):
        """
        Generates content by trying available API keys in sequence.
        For each key, it tries the primary model first, then the fallback model on ResourceExhausted errors.
        """
        if not self.api_keys:
            return "오류: Gemini API 키가 설정되지 않았습니다."

        from google.api_core.exceptions import ResourceExhausted
        from google.genai.types import GenerateContentConfig, Tool

        full_prompt = f"{system_instruction}\n\n{prompt}"
        config = GenerateContentConfig(tools=[Tool(google_search_retrieval={})])

        # Store the starting key index for this request
        start_index = self.current_key_index

        while self.current_key_index < len(self.api_keys):
            if not self.client and not self._initialize_client():
                # If we can't even initialize a client with the current key, try the next one.
                if not self.try_next_key():
                    break # No more keys to try

            # 1. Try with the primary model (Pro)
            try:
                st.session_state.analyst_model = "Gemini 1.5 Pro"
                st.info(f"API Key #{self.current_key_index + 1}을(를) 사용하여 '{st.session_state.analyst_model}' 모델로 분석을 시도합니다.")
                response = self.client.models.generate_content(
                    model=f'models/{self.primary_model}',
                    contents=full_prompt,
                    config=config
                )
                return response # Success
            except ResourceExhausted:
                st.warning(f"Key #{self.current_key_index + 1}의 '{self.primary_model}' 모델 사용량이 소진되었습니다. 폴백 모델로 전환합니다.")
                # 2. If exhausted, try with the fallback model (Flash) with the SAME key
                try:
                    st.session_state.analyst_model = "Gemini 1.5 Flash"
                    st.info(f"동일한 API Key #{self.current_key_index + 1}을(를) 사용하여 '{st.session_state.analyst_model}' 모델로 재시도합니다.")
                    response = self.client.models.generate_content(
                        model=f'models/{self.fallback_model}',
                        contents=full_prompt,
                        config=config
                    )
                    return response # Success with fallback
                except ResourceExhausted:
                    # Both models are exhausted for the current key, try the next key.
                    st.warning(f"Key #{self.current_key_index + 1}은(는) 두 모델 모두 사용량이 소진되었습니다.")
                    if not self.try_next_key():
                        break # No more keys left, exit the while loop
                except Exception as e_fallback:
                    error_message = f"**폴백 모델('{self.fallback_model}') 호출 중 오류 발생 (Key #{self.current_key_index + 1}):**\n\n**오류:**\n`{str(e_fallback)}`"
                    st.error(error_message)
                    if not self.try_next_key():
                        # Restore original index before failing
                        self.current_key_index = start_index
                        self._initialize_client()
                        return error_message # Return last error if no more keys
            except Exception as e:
                st.error(f"API Key #{self.current_key_index + 1} 사용 중 오류 발생: {type(e).__name__}")
                if not self.try_next_key():
                    # Restore original index before failing
                    self.current_key_index = start_index
                    self._initialize_client()
                    error_message = f"**API 호출 중 오류 발생 ('{self.primary_model}' 모델, Key #{self.current_key_index}):**\n\n**오류:**\n`{str(e)}`"
                    return error_message
        
        # If we've exited the loop, it means all keys from the start_index onwards have failed.
        # Reset to the starting key for the next attempt.
        self.current_key_index = start_index
        self._initialize_client()
        final_error_message = "**API 호출 최종 실패:** 사용 가능한 모든 API 키의 사용량 한도를 초과했거나, 지속적인 오류가 발생했습니다."
        st.error(final_error_message)
        return final_error_message



class KiwoomAPIHandler:
    """Handles interactions with the Kiwoom REST API."""
    def __init__(self, app_key, app_secret, base_url):
        self.app_key = app_key
        self.app_secret = app_secret
        self.base_url = base_url

    def get_token(self):
        """Fetches and caches the Kiwoom API token."""
        if not self.app_key or not self.app_secret:
            st.error("키움 API 키가 설정되지 않았습니다. config.ini 파일을 확인하세요.")
            return None

        if 'kiwoom_token' in st.session_state and st.session_state.kiwoom_token_expires_at > time.time():
            return st.session_state.kiwoom_token

        url = f"{self.base_url}/oauth2/token"
        headers = {"Content-Type": "application/json"}
        data = {
            "grant_type": "client_credentials",
            "appkey": self.app_key,
            "secretkey": self.app_secret
        }
        response = None
        try:
            response = requests.post(url, headers=headers, data=json.dumps(data))
            response.raise_for_status()
            token_data = response.json()

            if 'token' not in token_data and 'access_token' in token_data:
                token_data['token'] = token_data['access_token']

            if 'token' not in token_data:
                st.error("키움 API 토큰 발급 실패: 응답에 'token' 또는 'access_token'이 없습니다.")
                st.json(token_data)
                return None

            st.session_state.kiwoom_token = token_data['token']

            if 'expires_dt' in token_data:
                try:
                    expires_ts = pd.to_datetime(token_data['expires_dt'], format='%Y%m%d%H%M%S').timestamp()
                    st.session_state.kiwoom_token_expires_at = expires_ts - 60
                except ValueError:
                    st.session_state.kiwoom_token_expires_at = time.time() + 3600 - 60
            else:
                expires_in = int(token_data.get('expires_in', 3600))
                st.session_state.kiwoom_token_expires_at = time.time() + expires_in - 60

            return st.session_state.kiwoom_token
        except requests.exceptions.RequestException as e:
            st.error(f"키움 API 토큰 발급 중 오류 발생: {e}")
            if e.response:
                st.error(f"응답 내용: {e.response.text}")
            return None
        except json.JSONDecodeError:
            st.error(f"키움 API 토큰 응답이 JSON 형식이 아닙니다. 응답: {response.text if response else 'N/A'}")
            return None

    def get_stock_info(self, stock_code):
        """Fetches detailed stock information."""
        token = self.get_token()
        if not token:
            return {}

        url = f"{self.base_url}/api/dostk/stkinfo"
        headers = {
            "Content-Type": "application/json;charset=UTF-8",
            "authorization": f"Bearer {token}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "api-id": "ka10001"
        }
        params = {"stk_cd": stock_code}

        try:
            response = requests.post(url, headers=headers, json=params)
            response.raise_for_status()
            data = response.json()

            if data.get('return_code') != 0:
                st.error(f"키움 API 오류: {data.get('return_msg', '상세 메시지 없음')}")
                st.json(data)
                return {}

            def clean_value(value_str):
                if isinstance(value_str, str) and value_str:
                    try:
                        return float(value_str.replace('+', '').replace('-', ''))
                    except ValueError:
                        return 0.0
                elif isinstance(value_str, (int, float)):
                    return float(value_str)
                return 0.0

            info = {
                'price': clean_value(data.get('cur_prc', 0)),
                'market_cap': int(clean_value(data.get('mac', 0)) * 100000000),
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

    def _fetch_chart_data(self, token, api_id, endpoint, params):
        url = f'{self.base_url}{endpoint}'
        headers = {
            'Content-Type': 'application/json;charset=UTF-8',
            'authorization': f'Bearer {token}',
            'appkey': self.app_key,
            'appsecret': self.app_secret,
            'api-id': api_id,
        }
        try:
            response = requests.post(url, headers=headers, json=params)
            response.raise_for_status()
            response_data = response.json()

            if response_data.get('return_code') != 0:
                st.error(f"API Error for {api_id}: {response_data.get('return_msg')}")
                return []

            for key, value in response_data.items():
                if isinstance(value, list) and value:
                    return value
            
            # Handle non-list successful response by wrapping it in a list
            if response_data.get('return_code') == 0:
                return [response_data]

            return []
        except requests.exceptions.RequestException:
            return []
        except (ValueError, TypeError, json.JSONDecodeError):
            return []

    def _process_chart_dataframe(self, data_list):
        if not data_list:
            return pd.DataFrame()

        df = pd.DataFrame(data_list)

        date_col_map = {'stck_bsop_date': 'dt', 'date': 'dt', 'base_dt': 'dt', 'stck_dt': 'dt'}
        df.rename(columns=date_col_map, inplace=True)
        if 'dt' not in df.columns:
            # If no date column, cannot proceed with time series analysis
            return pd.DataFrame()
        df['dt'] = pd.to_datetime(df['dt'])
        df = df.set_index('dt').sort_index()

        def clean_and_convert_to_numeric(series):
            series_str = series.astype(str)
            cleaned_series = series_str.str.replace('[+,]', '', regex=True).str.replace('--', '-', regex=False)
            return pd.to_numeric(cleaned_series, errors='coerce').fillna(0)

        column_map = {
            'open': ['stck_oprc', 'open_pric'],
            'high': ['stck_hgpr', 'high_pric'],
            'low': ['stck_lwpr', 'low_pric'],
            'close': ['stck_clpr', 'cur_prc', 'clpr', 'stck_prpr', 'close_pric'],
            'volume': ['acml_vol', 'trde_qty', 'acc_trde_qty'],
            'for_rt': ['for_rt'],
            'for_netprps': ['for_netprps', 'frgnr_invsr'],
            'orgn_netprps': ['orgn_netprps', 'orgn'],
            'ind_netprps': ['ind_netprps', 'ind_invsr']
        }

        for standard_name, possible_names in column_map.items():
            for name in possible_names:
                if name in df.columns:
                    df[standard_name] = clean_and_convert_to_numeric(df[name])
                    break
        
        return df

    def fetch_all_chart_data(self, stock_code):
        token = self.get_token()
        if not token:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        now = pd.Timestamp.now()
        now_str = now.strftime('%Y%m%d')

        daily_params = {'stk_cd': stock_code, 'base_dt': now_str, 'period_cls': 'D', 'upd_stkpc_tp': '1'}
        daily_list = self._fetch_chart_data(token, 'ka10081', '/api/dostk/chart', daily_params)
        df_daily = self._process_chart_dataframe(daily_list)

        weekly_params = {'stk_cd': stock_code, 'base_dt': now_str, 'period_cls': 'W', 'upd_stkpc_tp': '1'}
        weekly_list = self._fetch_chart_data(token, 'ka10082', '/api/dostk/chart', weekly_params)
        df_weekly = self._process_chart_dataframe(weekly_list)

        monthly_params = {'stk_cd': stock_code, 'base_dt': now_str, 'period_cls': 'M', 'upd_stkpc_tp': '1'}
        monthly_list = self._fetch_chart_data(token, 'ka10083', '/api/dostk/chart', monthly_params)
        df_monthly = self._process_chart_dataframe(monthly_list)

        return df_daily, df_weekly, df_monthly

    def fetch_daily_fallback_data(self, stock_code):
        """Fetches daily data (closing price) as a fallback for line charts."""
        token = self.get_token()
        if not token:
            return pd.DataFrame()
        now = pd.Timestamp.now()
        now_str = now.strftime('%Y%m%d')
        params = {'stk_cd': stock_code, 'qry_dt': now_str, 'indc_tp': '1'}
        data_list = self._fetch_chart_data(token, 'ka10086', '/api/dostk/mrkcond', params)
        return self._process_chart_dataframe(data_list)

    def fetch_investor_data(self, stock_code):
        """Fetches investor-specific trading data."""
        token = self.get_token()
        if not token:
            return pd.DataFrame()
        
        now = pd.Timestamp.now()
        now_str = now.strftime('%Y%m%d')
        
        params = {'stk_cd': stock_code, 'dt': now_str, 'amt_qty_tp': '1', 'trde_tp': '0', 'unit_tp': '1000'}
        data_list = self._fetch_chart_data(token, 'ka10059', '/api/dostk/stkinfo', params)
        return self._process_chart_dataframe(data_list)

class ValuationCalculator:
    @staticmethod
    def calculate_target_pbr(roe, cost_of_equity, terminal_growth):
        roe_pct, ke_pct, g_pct = roe / 100, cost_of_equity / 100, terminal_growth / 100
        return (roe_pct - g_pct) / (ke_pct - g_pct) if (ke_pct - g_pct) != 0 else 0

    @staticmethod
    def calculate_target_price(target_pbr, bps):
        return target_pbr * bps

    @staticmethod
    def get_investment_opinion(current_price, target_price):
        if current_price <= 0 or target_price <= 0:
            return "-", 0.0
        upside = ((target_price / current_price) - 1) * 100
        if upside > 15: return "매수 (Buy)", upside
        if upside > -5: return "중립 (Neutral)", upside
        return "매도 (Sell)", upside

# --- Wrapper Functions ---

def get_kiwoom_token():
    if 'kiwoom_handler' not in st.session_state: return None
    return st.session_state.kiwoom_handler.get_token()

def get_kiwoom_stock_info(stock_code):
    if 'kiwoom_handler' not in st.session_state: return {}
    return st.session_state.kiwoom_handler.get_stock_info(stock_code)

def generate_gemini_content(prompt, system_instruction):
    if 'gemini_handler' not in st.session_state: return "오류: Gemini 핸들러가 초기화되지 않았습니다."
    return st.session_state.gemini_handler.generate_content(prompt, system_instruction)

# --- UI and Data Functions ---

@st.cache_data(ttl=86400)
def get_stock_list():
    try:
        df = pd.read_csv(os.path.join(APP_DIR, 'stock_list.csv'), dtype={'code': str, 'name': str})
        if 'name' not in df.columns or 'code' not in df.columns: return pd.DataFrame()
        return df
    except Exception: return pd.DataFrame()

def get_empty_forecast_df():
    data = {'(단위: 십억원)': ['매출액', '영업이익', '순이익', 'EPS (원)', 'BPS (원)', 'ROE (%)'],
            '2023A': [0.0]*6, '2024E': [0.0]*6, '2025E': [0.0]*6}
    return pd.DataFrame(data).set_index('(단위: 십억원)').astype(float)

def reset_states_on_stock_change():
    st.session_state.gemini_analysis = "상단 설정에서 기업 정보를 입력하고 'Gemini 최신 정보 분석' 버튼을 클릭하여 AI 분석을 시작하세요."
    st.session_state.main_business = "-"
    st.session_state.investment_summary = "-"
    st.session_state.kiwoom_data = {}
    st.session_state.df_forecast = get_empty_forecast_df()

# --- Plotting Functions ---

def display_candlestick_chart(stock_code, company_name):
    if 'kiwoom_handler' not in st.session_state:
        st.error("차트를 생성하려면 Kiwoom 핸들러가 필요합니다.")
        return

    with st.spinner("주가 및 투자자 동향 데이터를 조회하고 차트를 생성 중입니다..."):
        rangebreaks = [dict(bounds=["sat", "mon"]) # 주말 제외
        ]
        df_daily, df_weekly, df_monthly = st.session_state.kiwoom_handler.fetch_all_chart_data(stock_code)

        # --- Daily Chart with Investor Data ---
        df_daily_filtered = df_daily[df_daily.index >= (pd.Timestamp.now() - pd.DateOffset(months=3))]
        has_ohlc = not df_daily_filtered.empty and all(col in df_daily_filtered.columns for col in ['open', 'high', 'low', 'close'])

        if not has_ohlc:
            st.warning("API에서 일봉 OHLC 데이터를 제공하지 않아, 종가 기준 꺾은선 차트를 표시합니다.")
            df_daily_fallback = st.session_state.kiwoom_handler.fetch_daily_fallback_data(stock_code)
            df_daily_filtered = df_daily_fallback[df_daily_fallback.index >= (pd.Timestamp.now() - pd.DateOffset(months=3))]

        if not df_daily_filtered.empty:
            daily_title = f'{company_name} 일봉 (3개월)'
            fig = None
            if has_ohlc:
                # --- Fetch investor data ---
                df_investor = st.session_state.kiwoom_handler.fetch_investor_data(stock_code)
                
                # --- Create subplots (add row for investor data) ---
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, 
                                    subplot_titles=(daily_title, '거래량', '투자자별 매매동향 (순매수, 백만원)'), 
                                    row_heights=[0.5, 0.2, 0.3])

                # Candlestick Trace
                fig.add_trace(go.Candlestick(x=df_daily_filtered.index, open=df_daily_filtered['open'], high=df_daily_filtered['high'], low=df_daily_filtered['low'], close=df_daily_filtered['close'], name='캔들'), row=1, col=1)
                
                # Volume Trace
                if 'volume' in df_daily_filtered.columns:
                    colors = ['red' if c < o else 'green' for o, c in zip(df_daily_filtered['open'], df_daily_filtered['close'])]
                    fig.add_trace(go.Bar(x=df_daily_filtered.index, y=df_daily_filtered['volume'], name='거래량', marker_color=colors), row=2, col=1)

                # Investor Traces
                if not df_investor.empty and all(c in df_investor.columns for c in ['ind_netprps', 'for_netprps', 'orgn_netprps']):
                    df_investor_filtered = df_investor[df_investor.index.isin(df_daily_filtered.index)] # Align dates
                    fig.add_trace(go.Bar(x=df_investor_filtered.index, y=df_investor_filtered['ind_netprps'], name='개인'), row=3, col=1)
                    fig.add_trace(go.Bar(x=df_investor_filtered.index, y=df_investor_filtered['for_netprps'], name='외국인'), row=3, col=1)
                    fig.add_trace(go.Bar(x=df_investor_filtered.index, y=df_investor_filtered['orgn_netprps'], name='기관'), row=3, col=1)
                    fig.update_layout(barmode='group', yaxis3_title_text='순매수 금액')
                else:
                    st.warning("투자자별 매매동향 데이터를 차트에 표시할 수 없습니다. API 응답을 확인하세요.")

            else: # Fallback line chart
                fig = make_subplots(rows=1, cols=1, subplot_titles=(daily_title,))
                if 'close' in df_daily_filtered.columns:
                    fig.add_trace(go.Scatter(x=df_daily_filtered.index, y=df_daily_filtered['close'], mode='lines', name='종가'))
                else:
                    st.error(f"일봉 대체 데이터에 'close' 컬럼이 없습니다. 사용 가능한 컬럼: {df_daily_filtered.columns.tolist()}")
                    fig = None
            
            if fig:
                fig.update_xaxes(rangebreaks=rangebreaks)
                fig.update_layout(xaxis_rangeslider_visible=False, showlegend=True, height=700, margin=dict(l=10, r=10, b=10, t=40))
                st.plotly_chart(fig, use_container_width=True)

        # --- Weekly & Monthly Charts (Unchanged) ---
        chart_data = {
            '주봉 (1년)': df_weekly[df_weekly.index >= (pd.Timestamp.now() - pd.DateOffset(years=1))],
            '월봉 (3년)': df_monthly[df_monthly.index >= (pd.Timestamp.now() - pd.DateOffset(years=3))]
        }

        for title, df in chart_data.items():
            if df.empty or not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                st.warning(f"{title} 데이터를 표시할 수 없습니다. (필수 데이터 부족)")
                continue

            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, subplot_titles=(f'{company_name} {title}', '거래량'), row_heights=[0.7, 0.3])
            fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='캔들'), row=1, col=1)

            if 'volume' in df.columns:
                colors = ['red' if c < o else 'green' for o, c in zip(df['open'], df['close'])]
                fig.add_trace(go.Bar(x=df.index, y=df['volume'], name='거래량', marker_color=colors), row=2, col=1)

            if len(df) >= 5:
                df['ma5'] = df['close'].rolling(window=5).mean()
                fig.add_trace(go.Scatter(x=df.index, y=df['ma5'], name='MA 5', line=dict(color='orange', width=1)), row=1, col=1)
            if len(df) >= 20:
                df['ma20'] = df['close'].rolling(window=20).mean()
                fig.add_trace(go.Scatter(x=df.index, y=df['ma20'], name='MA 20', line=dict(color='purple', width=1)), row=1, col=1)

            fig.update_xaxes(rangebreaks=rangebreaks)
            fig.update_layout(xaxis_rangeslider_visible=False, showlegend=True, height=500, margin=dict(l=10, r=10, b=10, t=40))
            st.plotly_chart(fig, use_container_width=True)

# --- Main Application ---

def main():
    st.set_page_config(layout="wide")
    today_str = pd.Timestamp.now().strftime('%Y-%m-%d')

    if 'config_manager' not in st.session_state: st.session_state.config_manager = ConfigManager(ROOT_DIR)
    if 'gemini_handler' not in st.session_state: st.session_state.gemini_handler = GeminiAPIHandler(st.session_state.config_manager.get_gemini_keys())
    if 'kiwoom_handler' not in st.session_state:
        k_key, k_secret, _, k_url = st.session_state.config_manager.get_kiwoom_config()
        st.session_state.kiwoom_handler = KiwoomAPIHandler(k_key, k_secret, k_url)

    states_to_init = {
        "gemini_analysis": "상단 설정에서 기업 정보를 입력하고 'Gemini 최신 정보 분석' 버튼을 클릭하여 AI 분석을 시작하세요.",
        "main_business": "-", "investment_summary": "-", "kiwoom_data": {},
        "df_forecast": get_empty_forecast_df(), "gemini_api_calls": 0,
        "kiwoom_token": None, "kiwoom_token_expires_at": 0,
        "analyst_model": "Gemini 1.5 Pro" # Add default model name
    }
    for key, value in states_to_init.items():
        if key not in st.session_state: st.session_state[key] = value

    title_col, info_col = st.columns([3, 1])
    with title_col: st.title("AI 기반 투자 분석 리포트")
    with info_col: st.markdown(f"<div style='text-align: right;'><b>조회 기준일:</b> {today_str}<br><b>애널리스트:</b> {st.session_state.analyst_model}</div>", unsafe_allow_html=True)
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
                selected_stock = st.selectbox("기업 선택", stock_options, index=default_index, on_change=reset_states_on_stock_change, key='selected_stock', label_visibility="collapsed")
                company_name, stock_code = re.match(r"(.+) \((.+)\)", selected_stock).groups() if selected_stock else ("", "")
            else:
                company_name, stock_code = st.text_input("기업명", "SK하이닉스"), st.text_input("종목코드", "000660")

            btn_cols = st.columns(2)
            if btn_cols[0].button("📈 정보 조회", use_container_width=True):
                with st.spinner("키움 API에서 최신 정보를 조회 중입니다..."):
                    kiwoom_data = get_kiwoom_stock_info(stock_code)
                    if kiwoom_data:
                        st.session_state.kiwoom_data = kiwoom_data
                        df_new = st.session_state.df_forecast.copy()
                        df_new.loc['EPS (원)', '2023A'], df_new.loc['BPS (원)', '2023A'], df_new.loc['ROE (%)', '2023A'] = kiwoom_data.get('eps', 0), kiwoom_data.get('bps', 0), kiwoom_data.get('roe', 0)
                        st.session_state.df_forecast = df_new
                        st.success("정보 조회가 완료되었습니다.")
                    else: st.error("정보 조회에 실패했습니다.")

            if btn_cols[1].button("✨ AI 분석", use_container_width=True):
                st.session_state.gemini_api_calls += 1
                with st.spinner('Gemini가 최신 정보를 분석하고 출처를 확인 중입니다...'):
                    system_prompt = "당신은 15년 경력의 유능한 대한민국 주식 전문 애널리스트입니다. 웹 검색 기능을 활용하여 가장 최신 정보와 객관적인 데이터를 찾아 분석에 반영해야 합니다. 주장의 근거가 되는 부분에는 반드시 출처를 `[숫자]` 형식으로 명시해야 합니다. 명확하고 간결하게 핵심을 전달합니다."
                    user_prompt = f'''**기업 분석 요청**
- **분석 대상:** {company_name}({stock_code})
- **요청 사항:**
  1. **(최신 정보 기반)** 이 기업의 **주요 사업**에 대해 한국어로 2-3문장으로 요약해주세요.
  2. **(최신 정보 기반)** 이 기업에 대한 **핵심 투자 요약**을 강점과 약점을 포함하여 한국어로 3줄 이내로 작성해주세요.
  3. **(최신 정보 기반)** 최근 6개월간의 정보를 종합하여, 아래 형식에 맞춰 '긍정적 투자 포인트' 2가지와 '잠재적 리스크 요인' 2가지를 구체적인 근거와 함께 한국어로 도출해주세요.

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
                    response_or_error = generate_gemini_content(user_prompt, system_prompt)

                    # Check if the response is successful by checking for the .text attribute
                    if hasattr(response_or_error, 'text'):
                        full_response_obj = response_or_error
                        response_text = full_response_obj.text
                        citations = ""
                        if hasattr(full_response_obj, 'citation_metadata') and full_response_obj.citation_metadata:
                            citation_sources = full_response_obj.citation_metadata.citation_sources
                            if citation_sources:
                                citations = "\n\n---\n\n**출처:**\n"
                                for i, source in enumerate(citation_sources):
                                    citations += f"{i+1}. {source.uri}\n"

                        try:
                            parts = response_text.split('###')
                            st.session_state.main_business = parts[1].replace('주요 사업', '').strip()
                            st.session_state.investment_summary = parts[2].replace('핵심 투자 요약', '').strip()
                            analysis_content = "###" + "###".join(parts[3:])
                            st.session_state.gemini_analysis = analysis_content + citations
                        except Exception:
                            st.session_state.main_business, st.session_state.investment_summary = "-", "-"
                            st.session_state.gemini_analysis = f"**오류: Gemini 응답 처리 중 문제가 발생했습니다.**\n\n{response_text}{citations}"
                    # If it's not a response object, it must be the error string
                    elif isinstance(response_or_error, str):
                        st.session_state.gemini_analysis = response_or_error
                    else:
                        st.session_state.gemini_analysis = "AI 분석에 실패했습니다. API 호출 중 오류가 발생했거나 응답이 없습니다."
            st.caption(f"AI 분석 호출 (세션): {st.session_state.gemini_api_calls}")

        with col2:
            st.markdown("**가치평가 모델**")
            est_roe = st.slider("예상 ROE (%)", 0.0, 50.0, st.session_state.kiwoom_data.get('roe', 10.0), 0.1)
            cost_of_equity = st.slider("자기자본비용 (Ke, %)", 5.0, 15.0, 9.0, 0.1)
            terminal_growth = st.slider("영구성장률 (g, %)", 0.0, 5.0, 3.0, 0.1)
        with col3:
            st.markdown("**목표주가 변수**")
            est_bps = st.number_input("예상 BPS (원)", value=int(st.session_state.kiwoom_data.get('bps', 150000)))

    target_pbr = ValuationCalculator.calculate_target_pbr(est_roe, cost_of_equity, terminal_growth)
    calculated_target_price = ValuationCalculator.calculate_target_price(target_pbr, est_bps)
    current_price = st.session_state.kiwoom_data.get('price', 0)
    investment_opinion, upside_potential = ValuationCalculator.get_investment_opinion(current_price, calculated_target_price)
    st.divider()

    st.header("1. 요약 (Executive Summary)")
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
        market_cap = st.session_state.kiwoom_data.get('market_cap', 0)
        st.text_input("회사명", company_name, disabled=True)
        st.text_input("티커", stock_code, disabled=True)
        st.text_area("주요 사업", st.session_state.main_business, disabled=True)
        st.text_input("시가총액", f"{market_cap / 100000000:,.0f} 억원" if market_cap > 0 else "N/A", disabled=True)
        overview_cols = st.columns(2)
        overview_cols[0].metric("PER", f"{st.session_state.kiwoom_data.get('per', 0):.2f} 배")
        overview_cols[1].metric("PBR", f"{st.session_state.kiwoom_data.get('pbr', 0):.2f} 배")
        overview_cols[0].metric("52주 최고", f"{st.session_state.kiwoom_data.get('high_52w', 0):,.0f} 원")
        overview_cols[1].metric("52주 최저", f"{st.session_state.kiwoom_data.get('low_52w', 0):,.0f} 원")
    with main_col2:
        st.subheader("3. Gemini 종합 분석")
        with st.container(border=True): st.markdown(st.session_state.gemini_analysis)
    st.divider()

    st.header("4. 실적 전망 (Earnings Forecast)")
    st.caption("아래 표의 데이터를 직접 수정하여 목표주가 계산에 실시간으로 반영할 수 있습니다.")
    st.session_state.df_forecast = st.data_editor(st.session_state.df_forecast, use_container_width=True)
    st.divider()

    st.header("5. 가치평가 (Valuation)")
    val_col1, val_col2 = st.columns(2)
    with val_col1:
        st.markdown(f"- **(A) 예상 ROE:** `{est_roe:.2f} %`")
        st.markdown(f"- **(B) 자기자본비용 (Ke):** `{cost_of_equity:.2f} %`")
        st.markdown(f"- **(C) 영구성장률 (g):** `{terminal_growth:.2f} %`")
    with val_col2: st.success(f"**목표 PBR (배):** `{target_pbr:.2f}` 배")
    st.subheader("5.2. 목표주가 산출")
    val2_col1, val2_col2 = st.columns(2)
    with val2_col1:
        st.markdown(f"- **(D) 목표 PBR:** `{target_pbr:.2f}` 배")
        st.markdown(f"- **(E) 예상 BPS:** `{est_bps:,.0f}` 원")
    with val2_col2: st.success(f"**목표주가 (원):** `{calculated_target_price:,.0f}` 원")
    st.divider()

    st.header("6. 주가 차트 (Stock Chart)")
    if st.button("📊 주가 & 투자자 동향 차트 생성", help="키움 API를 통해 일봉, 주봉, 월봉 및 투자자별 동향 차트를 조회합니다.", use_container_width=True):
        display_candlestick_chart(stock_code, company_name)
    st.divider()
    st.write("*본 보고서는 외부 출처로부터 얻은 정보에 기반하며, 정확성을 보장하지 않습니다. 투자 결정에 대한 최종 책임은 투자자 본인에게 있습니다.*")

if __name__ == "__main__":
    main()