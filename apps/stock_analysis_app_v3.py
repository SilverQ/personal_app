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

    def _fetch_holdings_data(self, token, api_id, stock_code):
        url = f'{self.base_url}/api/dostk/frgnistt'
        headers = {
            'Content-Type': 'application/json;charset=UTF-8',
            'authorization': f'Bearer {token}',
            'appkey': self.app_key,
            'appsecret': self.app_secret,
            'api-id': api_id,
        }
        params = {'stk_cd': stock_code}
        try:
            response = requests.post(url, headers=headers, json=params)
            response.raise_for_status()
            response_data = response.json()

            if response_data.get('return_code') != 0:
                st.error(f"API Error for {api_id}: {response_data.get('return_msg')}")
                return []
            
            # The user's sample code suggests 'stk_frgnr' as the key for both.
            # We'll assume this is correct for now, but be aware it might need adjustment.
            if 'stk_frgnr' in response_data and isinstance(response_data['stk_frgnr'], list):
                return response_data['stk_frgnr']
            
            return []
        except requests.exceptions.RequestException as e:
            st.error(f"키움 API ({api_id}) 호출 중 오류 발생: {e}")
            if e.response:
                st.error(f"응답 내용: {e.response.text}")
            return []
        except (ValueError, TypeError, json.JSONDecodeError) as e:
            st.error(f"API 응답 데이터 처리 중 오류 발생: {e}")
            return []

    def fetch_investor_holdings(self, stock_code):
        token = self.get_token()
        if not token:
            return pd.DataFrame()

        stock_info = self.get_stock_info(stock_code)
        current_price = stock_info.get('price', 0)
        market_cap = stock_info.get('market_cap', 0) # Already in KRW

        if current_price <= 0 or market_cap <= 0:
            st.warning("현재가 또는 시가총액 정보를 가져올 수 없어 총 상장 주식 수를 계산할 수 없습니다.")
            return pd.DataFrame()

        # Calculate total shares (assuming market_cap is in KRW and price is in KRW)
        total_shares = market_cap / current_price
        
        foreign_data_list = self._fetch_holdings_data(token, 'ka10008', stock_code)
        institution_data_list = self._fetch_holdings_data(token, 'ka10009', stock_code)

        df_foreign = pd.DataFrame(foreign_data_list)
        df_institution = pd.DataFrame(institution_data_list)

        # Process and merge data
        if df_foreign.empty and df_institution.empty:
            return pd.DataFrame()

        # Standardize column names and convert types
        def process_holdings_df(df, investor_type):
            if df.empty:
                return pd.DataFrame()
            df.rename(columns={'dt': 'date', 'poss_stkcnt': f'{investor_type}_holdings'}, inplace=True)
            df['date'] = pd.to_datetime(df['date'])
            df[f'{investor_type}_holdings'] = pd.to_numeric(df[f'{investor_type}_holdings'], errors='coerce').fillna(0)
            return df.set_index('date')

        df_foreign_processed = process_holdings_df(df_foreign, 'foreign')
        df_institution_processed = process_holdings_df(df_institution, 'institution')

        # Merge on date
        df_holdings = pd.merge(df_foreign_processed, df_institution_processed, left_index=True, right_index=True, how='outer')
        df_holdings.fillna(0, inplace=True)

        # Calculate individual holdings
        if not df_holdings.empty:
            df_holdings['total_shares'] = total_shares # Add total shares as a column for context
            df_holdings['individual_holdings'] = df_holdings['total_shares'] - df_holdings['foreign_holdings'] - df_holdings['institution_holdings']
            # Ensure individual holdings don't go negative due to data discrepancies or rounding
            df_holdings['individual_holdings'] = df_holdings['individual_holdings'].clip(lower=0)

        return df_holdings

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
    st.session_state.master_analysis_results = []
    st.session_state.full_gemini_prompt = ""

# --- Plotting Functions ---

def display_candlestick_chart(stock_code, company_name):
    if 'kiwoom_handler' not in st.session_state:
        st.error("차트를 생성하려면 Kiwoom 핸들러가 필요합니다.")
        return

    with st.spinner("주가 및 투자자 동향 데이터를 조회하고 차트를 생성 중입니다..."):
        rangebreaks = [dict(bounds=["sat", "mon"])]  # 주말 제외
        df_daily, df_weekly, df_monthly = st.session_state.kiwoom_handler.fetch_all_chart_data(stock_code)

        col1, col2 = st.columns(2)

        with col1:
            # --- Daily Chart with Investor Data ---
            st.subheader("일봉 & 투자자 동향")
            df_daily_filtered = df_daily[df_daily.index >= (pd.Timestamp.now() - pd.DateOffset(months=3))]
            has_ohlc = not df_daily_filtered.empty and all(col in df_daily_filtered.columns for col in ['open', 'high', 'low', 'close'])

            if not df_daily.empty and not has_ohlc:
                st.warning("API에서 일봉 OHLC 데이터를 제공하지 않아, 종가 기준 꺾은선 차트를 표시합니다.")
                df_daily_fallback = st.session_state.kiwoom_handler.fetch_daily_fallback_data(stock_code)
                df_daily_filtered = df_daily_fallback[df_daily_fallback.index >= (pd.Timestamp.now() - pd.DateOffset(months=3))]
            elif df_daily.empty:
                st.warning("일봉 데이터를 가져올 수 없습니다.")
                df_daily_filtered = pd.DataFrame() # Ensure it's an empty dataframe

            if not df_daily_filtered.empty:
                daily_title = f'{company_name} 일봉 (3개월)'
                fig_daily = None
                if has_ohlc:
                    df_investor = st.session_state.kiwoom_handler.fetch_investor_data(stock_code)
                    fig_daily = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                                              subplot_titles=(daily_title, '거래량', '투자자별 누적 순매수 동향 (백만원)'),
                                              row_heights=[0.5, 0.2, 0.3])
                    fig_daily.add_trace(go.Candlestick(x=df_daily_filtered.index, open=df_daily_filtered['open'], high=df_daily_filtered['high'], low=df_daily_filtered['low'], close=df_daily_filtered['close'], name='캔들'), row=1, col=1)
                    if 'volume' in df_daily_filtered.columns:
                        colors = ['red' if c < o else 'green' for o, c in zip(df_daily_filtered['open'], df_daily_filtered['close'])]
                        fig_daily.add_trace(go.Bar(x=df_daily_filtered.index, y=df_daily_filtered['volume'], name='거래량', marker_color=colors), row=2, col=1)
                    if not df_investor.empty and all(c in df_investor.columns for c in ['ind_netprps', 'for_netprps', 'orgn_netprps']):
                        df_investor_filtered = df_investor[df_investor.index.isin(df_daily_filtered.index)].copy()
                        # Calculate cumulative sum
                        df_investor_filtered['ind_cumulative'] = df_investor_filtered['ind_netprps'].cumsum()
                        df_investor_filtered['for_cumulative'] = df_investor_filtered['for_netprps'].cumsum()
                        df_investor_filtered['orgn_cumulative'] = df_investor_filtered['orgn_netprps'].cumsum()

                        # Plot as line chart
                        fig_daily.add_trace(go.Scatter(x=df_investor_filtered.index, y=df_investor_filtered['ind_cumulative'], mode='lines', name='개인(누적)'), row=3, col=1)
                        fig_daily.add_trace(go.Scatter(x=df_investor_filtered.index, y=df_investor_filtered['for_cumulative'], mode='lines', name='외국인(누적)'), row=3, col=1)
                        fig_daily.add_trace(go.Scatter(x=df_investor_filtered.index, y=df_investor_filtered['orgn_cumulative'], mode='lines', name='기관(누적)'), row=3, col=1)
                        fig_daily.update_layout(yaxis3_title_text='누적 순매수 금액')
                    else:
                        st.warning("투자자별 매매동향 데이터를 차트에 표시할 수 없습니다.")
                else: # Fallback line chart
                    fig_daily = make_subplots(rows=1, cols=1, subplot_titles=(daily_title,))
                    if 'close' in df_daily_filtered.columns:
                        fig_daily.add_trace(go.Scatter(x=df_daily_filtered.index, y=df_daily_filtered['close'], mode='lines', name='종가'))
                    else:
                        st.error(f"일봉 대체 데이터에 'close' 컬럼이 없습니다.")
                        fig_daily = None
                
                if fig_daily:
                    fig_daily.update_xaxes(rangebreaks=rangebreaks)
                    fig_daily.update_layout(xaxis_rangeslider_visible=False, showlegend=True, height=600, margin=dict(l=10, r=10, b=10, t=40))
                    st.plotly_chart(fig_daily, use_container_width=True)

        with col2:
            st.subheader("주봉 & 월봉")
            # --- Weekly & Monthly Charts ---
            chart_data = {
                '주봉 (1년)': (df_weekly[df_weekly.index >= (pd.Timestamp.now() - pd.DateOffset(years=1))], 300),
                '월봉 (3년)': (df_monthly[df_monthly.index >= (pd.Timestamp.now() - pd.DateOffset(years=3))], 300)
            }

            for title, (df, height) in chart_data.items():
                if df.empty or not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                    st.warning(f"{title} 데이터를 표시할 수 없습니다. (필수 데이터 부족)")
                    continue

                fig_period = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, subplot_titles=(f'{company_name} {title}', '거래량'), row_heights=[0.7, 0.3])
                fig_period.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='캔들'), row=1, col=1)

                if 'volume' in df.columns:
                    colors = ['red' if c < o else 'green' for o, c in zip(df['open'], df['close'])]
                    fig_period.add_trace(go.Bar(x=df.index, y=df['volume'], name='거래량', marker_color=colors), row=2, col=1)

                if len(df) >= 5:
                    df['ma5'] = df['close'].rolling(window=5).mean()
                    fig_period.add_trace(go.Scatter(x=df.index, y=df['ma5'], name='MA 5', line=dict(color='orange', width=1)), row=1, col=1)
                if len(df) >= 20:
                    df['ma20'] = df['close'].rolling(window=20).mean()
                    fig_period.add_trace(go.Scatter(x=df.index, y=df['ma20'], name='MA 20', line=dict(color='purple', width=1)), row=1, col=1)

                fig_period.update_xaxes(rangebreaks=rangebreaks)
                fig_period.update_layout(xaxis_rangeslider_visible=False, showlegend=True, height=height, margin=dict(l=10, r=10, b=10, t=40))
                st.plotly_chart(fig_period, use_container_width=True)

# --- Main Application ---

def main():
    st.set_page_config(layout="wide")
    today_str = pd.Timestamp.now().strftime('%Y-%m-%d')

    # --- CSS Injection for Compact Layout ---
    st.markdown("""
        <style>
            .st-emotion-cache-16txtl3 {padding-top: 1rem;}
            h1, h2, h3, h4, h5 {
                margin-top: 0.2rem !important;
                margin-bottom: 0.2rem !important;
            }
            .stApp [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
                gap: 0.5rem;
            }
            .st-expander-header {
                padding-top: 0.5rem !important;
                padding-bottom: 0.5rem !important;
            }
            [data-testid="stMetricValue"] {
                font-size: 1.2rem;
            }
            [data-testid="stMetricLabel"] {
                 font-size: 0.9rem;
            }
        </style>
    """, unsafe_allow_html=True)


    if 'config_manager' not in st.session_state: st.session_state.config_manager = ConfigManager(ROOT_DIR)
    if 'gemini_handler' not in st.session_state: st.session_state.gemini_handler = GeminiAPIHandler(st.session_state.config_manager.get_gemini_keys())
    if 'kiwoom_handler' not in st.session_state:
        k_key, k_secret, _, k_url = st.session_state.config_manager.get_kiwoom_config()
        st.session_state.kiwoom_handler = KiwoomAPIHandler(k_key, k_secret, k_url)

    states_to_init = {
        "gemini_analysis": "좌측 설정에서 기업 정보를 입력하고 'AI 분석' 버튼을 클릭하여 분석을 시작하세요.",
        "main_business": "-", "investment_summary": "-", "kiwoom_data": {},
        "df_forecast": get_empty_forecast_df(), "gemini_api_calls": 0,
        "kiwoom_token": None, "kiwoom_token_expires_at": 0,
        "analyst_model": "Gemini 1.5 Pro",
        "master_analysis_results": [], # 대가별 분석 결과를 저장할 리스트
        "full_gemini_prompt": "" # 전송된 전체 프롬프트를 저장
    }
    for key, value in states_to_init.items():
        if key not in st.session_state: st.session_state[key] = value

    st.title("AI 기반 투자 분석 리포트")
    st.markdown(f"<div style='text-align: right; font-size: 0.9rem;'><b>조회 기준일:</b> {today_str} | <b>애널리스트:</b> {st.session_state.analyst_model}</div>", unsafe_allow_html=True)
    st.divider()

    # --- Main 4-Column Layout ---
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1.5])

    # --- Column 1: Overview & Settings ---
    with col1:
        st.markdown("##### 기업 선택 및 조회")
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
            with st.spinner("API 정보 조회 중..."):
                kiwoom_data = get_kiwoom_stock_info(stock_code)
                if kiwoom_data:
                    st.session_state.kiwoom_data = kiwoom_data
                    df_new = st.session_state.df_forecast.copy()
                    df_new.loc['EPS (원)', '2023A'], df_new.loc['BPS (원)', '2023A'], df_new.loc['ROE (%)', '2023A'] = kiwoom_data.get('eps', 0), kiwoom_data.get('bps', 0), kiwoom_data.get('roe', 0)
                    st.session_state.df_forecast = df_new
                else: st.error("정보 조회 실패")

        if btn_cols[1].button("✨ AI 분석", use_container_width=True):
            st.session_state.gemini_api_calls += 1
            with st.spinner('Gemini가 분석 중입니다...'):
                system_prompt = "당신은 15년 경력의 유능한 대한민국 주식 전문 애널리스트입니다. 웹 검색 기능을 활용하여 가장 최신 정보와 객관적인 데이터를 찾아 분석에 반영해야 합니다. 주장의 근거가 되는 부분에는 반드시 출처를 `[숫자]` 형식으로 명시해야 합니다. 명확하고 간결하게 핵심을 전달합니다."
                # 대가별 심층 분석 질문들을 포함한 확장된 프롬프트
                user_prompt = f'''**기업 분석 요청**
- **분석 대상:** {company_name}({stock_code})
- **요청 사항:**
  1. **(최신 정보 기반)** 이 기업의 **주요 사업**에 대해 한국어로 2-3문장으로 요약해주세요.
  2. **(최신 정보 기반)** 이 기업에 대한 **핵심 투자 요약**을 강점과 약점을 포함하여 한국어로 3줄 이내로 작성해주세요.
  3. **(최신 정보 기반)** 최근 6개월간의 정보를 종합하여, 아래 형식에 맞춰 '긍정적 투자 포인트' 2가지와 '잠재적 리스크 요인' 2가지를 구체적인 근거와 함께 한국어로 도출해주세요.
  4. **(워런 버핏 - 경제적 해자)** 분석 대상 기업의 **경제적 해자**는 무엇이며, 경쟁사 대비 얼마나 강력합니까?
  5. **(워런 버핏 - 경영진 평가)** 현재 **경영진의 능력과 평판**은 어떻습니까? 최근 주요 의사결정은 무엇이었나요?
  6. **(윌리엄 오닐 - 산업 내 리더십)** 이 기업은 **산업 내 주도주**입니까, 아니면 후발주자입니까? 시장 점유율과 기술력 측면에서 비교해주세요.
  7. **(조지 소로스/레이 달리오 - 거시 경제 영향)** 현재 **금리, 환율 등 거시 경제 상황**이 이 기업에 미칠 긍정적/부정적 영향은 무엇입니까?

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
- [근거]

### 대가별 심층 분석 시작
#### 워런 버핏 - 경제적 해자
**질문:** 분석 대상 기업의 경제적 해자는 무엇이며, 경쟁사 대비 얼마나 강력합니까?
**답변:** [AI 분석 결과]

#### 워런 버핏 - 경영진 평가
**질문:** 현재 경영진의 능력과 평판은 어떻습니까? 최근 주요 의사결정은 무엇이었나요?
**답변:** [AI 분석 결과]

#### 윌리엄 오닐 - 산업 내 리더십
**질문:** 이 기업은 산업 내 주도주입니까, 아니면 후발주자입니까? 시장 점유율과 기술력 측면에서 비교해주세요.
**답변:** [AI 분석 결과]

#### 조지 소로스/레이 달리오 - 거시 경제 영향
**질문:** 현재 금리, 환율 등 거시 경제 상황이 이 기업에 미칠 긍정적/부정적 영향은 무엇입니까?
**답변:** [AI 분석 결과]
'''
                st.session_state.full_gemini_prompt = user_prompt  # 전체 프롬프트 저장

                response_or_error = generate_gemini_content(user_prompt, system_prompt)
                if hasattr(response_or_error, 'text'):
                    full_response_obj = response_or_error
                    response_text = full_response_obj.text
                    citations = ""
                    if hasattr(full_response_obj, 'citation_metadata') and full_response_obj.citation_metadata and full_response_obj.citation_metadata.citation_sources:
                        citations = "\n\n---\n\n**출처:**\n" + "\n".join(f"{i+1}. {source.uri}" for i, source in enumerate(full_response_obj.citation_metadata.citation_sources))
                    try:
                        # 주요 사업, 핵심 투자 요약 파싱
                        main_business_match = re.search(r'### 주요 사업\n(.*?)(?=\n###)', response_text, re.DOTALL)
                        investment_summary_match = re.search(r'### 핵심 투자 요약\n(.*?)(?=\n###)', response_text, re.DOTALL)

                        st.session_state.main_business = main_business_match.group(
                            1).strip() if main_business_match else "-"
                        st.session_state.investment_summary = investment_summary_match.group(
                            1).strip() if investment_summary_match else "-"

                        # 긍정적 투자 포인트, 잠재적 리스크 요인 파싱 (일반 분석)
                        general_analysis_match = re.search(r'(### 긍정적 투자 포인트.*?)(?=\n### 대가별 심층 분석 시작)', response_text,
                                                           re.DOTALL)
                        st.session_state.gemini_analysis = general_analysis_match.group(
                            1).strip() + citations if general_analysis_match else "일반 분석 파싱 실패" + citations

                        # 대가별 심층 분석 파싱
                        master_analysis_section_match = re.search(r'### 대가별 심층 분석 시작\n(.*)', response_text, re.DOTALL)
                        if master_analysis_section_match:
                            master_analysis_text = master_analysis_section_match.group(1)
                            # 각 대가별 분석 블록을 정규식으로 찾기
                            master_blocks = re.findall(r'#### (.+?)\n\*\*질문:\*\*(.+?)\n\*\*답변:\*\*(.+?)(?=\n####|\Z)',
                                                       master_analysis_text, re.DOTALL)

                            st.session_state.master_analysis_results = []
                            for block in master_blocks:
                                master_concept = block[0].strip()
                                question = block[1].strip()
                                answer = block[2].strip()
                                st.session_state.master_analysis_results.append({
                                    'master_concept': master_concept,
                                    'question': question,
                                    'answer': answer
                                })
                        else:
                            st.session_state.master_analysis_results = []
                            st.error("대가별 심층 분석 섹션을 찾을 수 없습니다.")

                    except Exception as e:
                        st.session_state.main_business, st.session_state.investment_summary = "-", "-"
                        st.session_state.gemini_analysis = f"**오류: Gemini 응답 처리 중 문제가 발생했습니다.**\n\n{response_text}{citations}\n\n**파싱 오류:** {e}"
                        st.session_state.master_analysis_results = []
                elif isinstance(response_or_error, str):
                    st.session_state.gemini_analysis = response_or_error
                    st.session_state.master_analysis_results = []
                else:
                    st.session_state.gemini_analysis = "AI 분석에 실패했습니다."
                    st.session_state.master_analysis_results = []

        st.caption(f"AI 분석 호출 (세션): {st.session_state.gemini_api_calls}")
        
        with st.container(border=True):
            st.markdown("##### 기업 개요")
            market_cap = st.session_state.kiwoom_data.get('market_cap', 0)
            st.markdown(f"""
            - **시가총액**: {market_cap / 100000000:,.0f} 억원
            - **PER / PBR**: {st.session_state.kiwoom_data.get('per', 0):.2f} 배 / {st.session_state.kiwoom_data.get('pbr', 0):.2f} 배
            - **52주 최고/최저**: {st.session_state.kiwoom_data.get('high_52w', 0):,.0f} / {st.session_state.kiwoom_data.get('low_52w', 0):,.0f} 원
            """)

    # --- Column 2: Core AI Analysis & Opinion ---
    with col2:
        st.markdown("##### AI 요약 및 투자의견")
        current_price = st.session_state.kiwoom_data.get('price', 0)
        # Read slider values from session_state to ensure immediate update
        est_roe = st.session_state.get('est_roe_slider', st.session_state.kiwoom_data.get('roe', 10.0))
        cost_of_equity = st.session_state.get('cost_of_equity_slider', 9.0)
        terminal_growth = st.session_state.get('terminal_growth_slider', 3.0)
        est_bps = st.session_state.get('est_bps_input', int(st.session_state.kiwoom_data.get('bps', 150000)))
        
        target_pbr = ValuationCalculator.calculate_target_pbr(est_roe, cost_of_equity, terminal_growth)
        calculated_target_price = ValuationCalculator.calculate_target_price(target_pbr, est_bps)
        investment_opinion, upside_potential = ValuationCalculator.get_investment_opinion(current_price, calculated_target_price)

        m_cols = st.columns(2)
        m_cols[0].metric("투자의견", investment_opinion)
        m_cols[1].metric("상승여력", f"{upside_potential:.2f} %")
        m_cols[0].metric("현재주가", f"{current_price:,.0f} 원" if current_price else "N/A")
        m_cols[1].metric("목표주가", f"{calculated_target_price:,.0f} 원")

        with st.container(border=True, height=300):
            st.markdown("##### 주요 사업")
            st.markdown(st.session_state.main_business)
            st.markdown("---")
            st.markdown("##### 핵심 투자 요약")
            st.markdown(st.session_state.investment_summary)

    # --- Column 3: Valuation & Earnings ---
    with col3:
        st.markdown("##### 가치평가 및 실적")
        with st.container(border=True):
            st.slider("예상 ROE (%)", 0.0, 50.0, st.session_state.kiwoom_data.get('roe', 10.0), 0.1, key="est_roe_slider")
            st.slider("자기자본비용 (Ke, %)", 5.0, 15.0, 9.0, 0.1, key="cost_of_equity_slider")
            st.slider("영구성장률 (g, %)", 0.0, 5.0, 3.0, 0.1, key="terminal_growth_slider")
            st.number_input("예상 BPS (원)", value=int(st.session_state.kiwoom_data.get('bps', 150000)), key="est_bps_input")
            st.success(f"**목표 PBR:** `{target_pbr:.2f}` 배 | **목표주가:** `{calculated_target_price:,.0f}` 원")

        st.markdown("##### 실적 전망 (단위: 십억원)")
        st.session_state.df_forecast = st.data_editor(st.session_state.df_forecast, use_container_width=True, height=240)

    # --- Column 4: Detailed AI Analysis ---
    with col4:
        st.markdown("##### Gemini 세부 분석")
        with st.container(border=True, height=550):
            st.markdown(st.session_state.gemini_analysis)  # 일반 분석 결과 표시

            if st.session_state.master_analysis_results:
                st.markdown("---")
                st.markdown("##### 대가별 심층 분석")
                for item in st.session_state.master_analysis_results:
                    with st.expander(f"**{item['master_concept']}**"):
                        st.markdown(f"**질문:** {item['question']}")
                        st.markdown(f"**답변:** {item['answer']}")

            # 전체 프롬프트 표시
            if st.session_state.full_gemini_prompt:
                st.markdown("---")
                with st.expander("Gemini 프롬프트 확인"):
                    st.code(st.session_state.full_gemini_prompt, language='markdown')

    st.divider()

    # --- Bottom Chart Area ---
    st.header("6. 주가 차트 (Stock Chart)")
    if st.button("📊 주가 & 투자자 동향 차트 생성", help="키움 API를 통해 일봉, 주봉, 월봉 및 투자자별 동향 차트를 조회합니다.", use_container_width=True):
        display_candlestick_chart(stock_code, company_name)
    
    st.divider()
    st.write("*본 보고서는 외부 출처로부터 얻은 정보에 기반하며, 정확성을 보장하지 않습니다. 투자 결정에 대한 최종 책임은 투자자 본인에게 있습니다.*")

if __name__ == "__main__":
    main()