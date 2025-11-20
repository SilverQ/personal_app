# -*- coding: utf-8 -*-
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
import plotly.io as pio
from PIL import Image
from io import BytesIO
from typing import Any, Dict, List

# --- Constants and Paths ---
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)

# --- JSON Store Helpers ---

def _instrument_id_from_code(code: str, market: str = 'KRX') -> str:
    try:
        c = str(code).strip()
    except Exception:
        c = str(code)
    m = (market or 'KRX').strip() if isinstance(market, str) else 'KRX'
    return f"{m}:{c}"

def _data_base_dir() -> str:
    return os.path.join(ROOT_DIR, '@data')

def _timeseries_dir(code: str) -> str:
    return os.path.join(_data_base_dir(), 'timeseries', code)

def _instruments_dir() -> str:
    return os.path.join(_data_base_dir(), 'instruments')

def _manifests_dir() -> str:
    return os.path.join(_data_base_dir(), 'manifests')

def _ensure_dirs(*paths: str):
    for p in paths:
        try:
            os.makedirs(p, exist_ok=True)
        except Exception:
            pass

def _year_from_date(d: pd.Timestamp) -> int:
    return int(pd.Timestamp(d).year)

def _load_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}

def _save_json_atomic(path: str, obj: Dict[str, Any]):
    tmp = f"{path}.tmp"
    try:
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        # Best-effort write; fall back to direct write
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

def _candles_year_path(code: str, year: int) -> str:
    return os.path.join(_timeseries_dir(code), f'candles_daily-{year}.json')

def _investor_year_path(code: str, year: int) -> str:
    return os.path.join(_timeseries_dir(code), f'investor_flows-{year}.json')

def _init_candles_header(code: str, year: int) -> Dict[str, Any]:
    return {
        'schema_version': '1.0.0',
        'resource_type': 'timeseries/candles',
        'instrument_id': _instrument_id_from_code(code),
        'interval': 'daily',
        'year': int(year),
        'meta': {'currency': 'KRW', 'timezone': 'Asia/Seoul', 'units': {'volume': 'shares'}},
        'source': {'name': 'Kiwoom', 'endpoint': '/api/dostk/chart', 'api_id': 'ka10081'},
        'retrieved_at': pd.Timestamp.now().isoformat(),
        'data': {}
    }

def _init_investor_header(code: str, year: int) -> Dict[str, Any]:
    return {
        'schema_version': '1.0.0',
        'resource_type': 'timeseries/investor_flows',
        'instrument_id': _instrument_id_from_code(code),
        'year': int(year),
        'meta': {'currency': 'KRW', 'unit': 'KRW', 'scale': 1.0},
        'source': {'name': 'Kiwoom', 'endpoint': '/api/dostk/stkinfo', 'api_id': 'ka10059'},
        'retrieved_at': pd.Timestamp.now().isoformat(),
        'data': {}
    }

def _upsert_manifest(code: str, dataset: str, year: int, dates: List[str]):
    _ensure_dirs(_manifests_dir())
    mpath = os.path.join(_manifests_dir(), f'{code}.json')
    manifest = _load_json(mpath) or {
        'schema_version': '1.0.0',
        'instrument_id': _instrument_id_from_code(code),
        'datasets': {}
    }
    ds_list = manifest.get('datasets', {}).get(dataset, [])
    # Find year entry
    entry = None
    for e in ds_list:
        if int(e.get('year', -1)) == int(year):
            entry = e
            break
    if not dates:
        # nothing to update
        _save_json_atomic(mpath, manifest)
        return
    start_d = min(dates)
    end_d = max(dates)
    now_iso = pd.Timestamp.now().isoformat()
    if entry is None:
        entry = {'year': int(year), 'start': start_d, 'end': end_d, 'last_updated': now_iso}
        ds_list.append(entry)
    else:
        entry['start'] = min(entry.get('start', start_d), start_d)
        entry['end'] = max(entry.get('end', end_d), end_d)
        entry['last_updated'] = now_iso
    manifest.setdefault('datasets', {})[dataset] = ds_list
    _save_json_atomic(mpath, manifest)

def upsert_instrument_metadata(code: str, name: str = None, market: str = None, currency: str = 'KRW'):
    """Create or update instrument metadata file @data/instruments/{code}.json"""
    _ensure_dirs(_instruments_dir())
    ipath = os.path.join(_instruments_dir(), f'{code}.json')
    obj = _load_json(ipath) or {
        'schema_version': '1.0.0',
        'resource_type': 'instrument',
        'instrument_id': _instrument_id_from_code(code),
        'code': code,
    }
    if name:
        obj['name'] = name
    if market:
        obj['market'] = market
    if currency:
        obj['currency'] = currency
    obj['retrieved_at'] = pd.Timestamp.now().isoformat()
    _save_json_atomic(ipath, obj)

def upsert_daily_candles_store(code: str, df_daily: pd.DataFrame):
    if df_daily is None or df_daily.empty:
        return
    _ensure_dirs(_timeseries_dir(code))
    # Expect df_daily index as datetime-like
    dfi = df_daily.copy()
    try:
        dfi.index = pd.to_datetime(dfi.index)
    except Exception:
        dfi.index = pd.to_datetime(dfi.index, errors='coerce')
    dates_group = {}
    for d, row in dfi.iterrows():
        if pd.isna(d):
            continue
        y = _year_from_date(d)
        dates_group.setdefault(y, []).append((pd.Timestamp(d).strftime('%Y-%m-%d'), row))
    for year, items in dates_group.items():
        path = _candles_year_path(code, year)
        obj = _load_json(path)
        if not obj:
            obj = _init_candles_header(code, year)
        data = obj.setdefault('data', {})
        written_dates = []
        for day_str, row in items:
            try:
                data[day_str] = {
                    'o': float(row.get('open', float('nan'))),
                    'h': float(row.get('high', float('nan'))),
                    'l': float(row.get('low', float('nan'))),
                    'c': float(row.get('close', float('nan'))),
                    'v': float(row.get('volume', float('nan'))) if 'volume' in row else None,
                    'as_of': pd.Timestamp.now().isoformat()
                }
                written_dates.append(day_str)
            except Exception:
                continue
        obj['retrieved_at'] = pd.Timestamp.now().isoformat()
        _save_json_atomic(path, obj)
        _upsert_manifest(code, 'candles_daily', year, written_dates)

def upsert_investor_flows_store(code: str, df_inv: pd.DataFrame):
    if df_inv is None or df_inv.empty:
        return
    _ensure_dirs(_timeseries_dir(code))
    dfi = df_inv.copy()
    try:
        dfi.index = pd.to_datetime(dfi.index)
    except Exception:
        dfi.index = pd.to_datetime(dfi.index, errors='coerce')
    dates_group = {}
    for d, row in dfi.iterrows():
        if pd.isna(d):
            continue
        y = _year_from_date(d)
        dates_group.setdefault(y, []).append((pd.Timestamp(d).strftime('%Y-%m-%d'), row))
    for year, items in dates_group.items():
        path = _investor_year_path(code, year)
        obj = _load_json(path)
        if not obj:
            obj = _init_investor_header(code, year)
        data = obj.setdefault('data', {})
        written_dates = []
        for day_str, row in items:
            try:
                data[day_str] = {
                    'individual': float(row.get('ind_netprps', 0.0)),
                    'foreign': float(row.get('for_netprps', 0.0)),
                    'institution': float(row.get('orgn_netprps', 0.0)),
                    'as_of': pd.Timestamp.now().isoformat()
                }
                written_dates.append(day_str)
            except Exception:
                continue
        obj['retrieved_at'] = pd.Timestamp.now().isoformat()
        _save_json_atomic(path, obj)
        _upsert_manifest(code, 'investor_flows', year, written_dates)

def load_daily_candles_from_store(code: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    try:
        start_date = pd.Timestamp(start_date).normalize()
        end_date = pd.Timestamp(end_date).normalize()
    except Exception:
        return pd.DataFrame()
    years = list(range(start_date.year, end_date.year + 1))
    records = []
    for y in years:
        path = _candles_year_path(code, y)
        obj = _load_json(path)
        if not obj or 'data' not in obj:
            continue
        for d, v in obj['data'].items():
            try:
                ts = pd.to_datetime(d)
            except Exception:
                continue
            if ts < start_date or ts > end_date:
                continue
            records.append({'date': ts, 'open': v.get('o'), 'high': v.get('h'), 'low': v.get('l'), 'close': v.get('c'), 'volume': v.get('v')})
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame.from_records(records).set_index('date').sort_index()
    return df

def load_investor_flows_from_store(code: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    try:
        start_date = pd.Timestamp(start_date).normalize()
        end_date = pd.Timestamp(end_date).normalize()
    except Exception:
        return pd.DataFrame()
    years = list(range(start_date.year, end_date.year + 1))
    records = []
    for y in years:
        path = _investor_year_path(code, y)
        obj = _load_json(path)
        if not obj or 'data' not in obj:
            continue
        for d, v in obj['data'].items():
            try:
                ts = pd.to_datetime(d)
            except Exception:
                continue
            if ts < start_date or ts > end_date:
                continue
            records.append({'date': ts, 'ind_netprps': v.get('individual', 0.0), 'for_netprps': v.get('foreign', 0.0), 'orgn_netprps': v.get('institution', 0.0)})
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame.from_records(records).set_index('date').sort_index()
    return df

# --- Derivation helpers (reuse store to build weekly/monthly) ---
def _has_coverage(df: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp, tolerance_days: int = 3) -> bool:
    if df is None or df.empty:
        return False
    try:
        s = pd.Timestamp(start_date).normalize()
        e = pd.Timestamp(end_date).normalize()
        dmin = pd.Timestamp(df.index.min()).normalize()
        dmax = pd.Timestamp(df.index.max()).normalize()
        return (dmin <= s + pd.Timedelta(days=tolerance_days)) and (dmax >= e - pd.Timedelta(days=1))
    except Exception:
        return False

def derive_weekly_monthly_from_daily(df_daily: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    if df_daily is None or df_daily.empty:
        return pd.DataFrame(), pd.DataFrame()
    dfd = df_daily.copy()
    try:
        dfd.index = pd.to_datetime(dfd.index)
    except Exception:
        dfd.index = pd.to_datetime(dfd.index, errors='coerce')
    dfd = dfd.sort_index()
    agg = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }
    if 'volume' in dfd.columns:
        agg['volume'] = 'sum'
    try:
        weekly = dfd.resample('W-FRI').agg(agg).dropna(how='all')
    except Exception:
        weekly = pd.DataFrame()
    try:
        monthly = dfd.resample('M').agg(agg).dropna(how='all')
    except Exception:
        monthly = pd.DataFrame()
    return weekly, monthly

# --- Helpers to validate/repair period frequency ---
def ensure_monthly_frequency(df_monthly: pd.DataFrame,
                             df_daily_full: pd.DataFrame = None,
                             df_weekly: pd.DataFrame = None) -> pd.DataFrame:
    """Ensure we have approximately one candlestick per calendar month.
    If the provided monthly DataFrame looks quarterly/sparse, try to rebuild
    from a wider daily dataset; if not available, rebuild from weekly.
    """
    try:
        if df_monthly is None or df_monthly.empty:
            # Try rebuild
            if isinstance(df_daily_full, pd.DataFrame) and not df_daily_full.empty:
                _, monthly = derive_weekly_monthly_from_daily(df_daily_full)
                return monthly
            if isinstance(df_weekly, pd.DataFrame) and not df_weekly.empty:
                dfw = df_weekly.copy()
                dfw.index = pd.to_datetime(dfw.index)
                agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}
                if 'volume' in dfw.columns:
                    agg['volume'] = 'sum'
                return dfw.resample('M').agg(agg).dropna(how='all')

        idx = pd.to_datetime(df_monthly.index)
        idx = idx.sort_values()
        if len(idx) < 6:
            # Too few points, try rebuild
            return ensure_monthly_frequency(pd.DataFrame(), df_daily_full, df_weekly)
        # Compute typical gap in days
        gaps = np.diff(idx.values).astype('timedelta64[D]').astype(int)
        median_gap = np.median(gaps) if len(gaps) else 30
        # If median gap > 50 days, it's likely quarterly or sparser
        if median_gap > 50:
            return ensure_monthly_frequency(pd.DataFrame(), df_daily_full, df_weekly)
        return df_monthly
    except Exception:
        # Best-effort: return as-is on failure
        return df_monthly

# --- Utility: Save charts as JPG ---
def _safe_filename(name: str) -> str:
    try:
        s = str(name)
    except Exception:
        s = "chart"
    # Remove characters illegal on Windows and normalize spaces
    s = re.sub(r'[\\/:*?"<>|]', '', s)
    s = s.strip().replace(' ', '_')
    return s or 'chart'

def save_plotly_fig_as_jpg(fig, out_path, width=1200, height=800, scale=2):
    try:
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        # Requires kaleido
        pio.write_image(fig, out_path, format='jpg', engine='kaleido', width=width, height=height, scale=scale)
        return True, out_path
    except Exception as e:
        # English fallback messages to avoid encoding issues
        st.error(f"Combined chart export failed: {e}")
        st.info("Image export requires 'kaleido' and 'pillow'. Install: pip install -U kaleido pillow")
        return False, str(e)
        # English fallback messages to avoid encoding issues
        st.error(f"Chart export failed: {e}")
        st.info("Image export requires 'kaleido'. Install: pip install -U kaleido")
        return False, str(e)
        st.error(f"차트 저장 실패: {e}")
        st.info("이미지 저장을 위해 'kaleido' 패키지가 필요합니다. 설치: pip install -U kaleido")
        return False, str(e)

def save_combined_charts_as_jpg(fig_daily, fig_weekly, fig_monthly, out_path,
                                size=1400, top_ratio=0.62,
                                scale=1, bg_color=(255,255,255)):
    try:
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        # Target square canvas
        size = int(size)
        top_ratio = float(top_ratio)
        daily_h = max(300, int(size * top_ratio))
        bottom_h = max(300, size - daily_h)
        left_w = size
        right_w_each = size // 2
        # Make safe copies for export and adjust margins to avoid clipping
        fd = go.Figure(fig_daily)
        fw = go.Figure(fig_weekly)
        fm = go.Figure(fig_monthly)
        export_margin = dict(l=60, r=30, b=50, t=60)
        fd.update_layout(margin=export_margin)
        fw.update_layout(margin=export_margin)
        fm.update_layout(margin=export_margin)
        
        def _render_bytes(fig, w, h, sc):
            # Try JPG first, then fallback to PNG
            try:
                return pio.to_image(fig, format='jpg', engine='kaleido', width=w, height=h, scale=sc)
            except Exception:
                return pio.to_image(fig, format='png', engine='kaleido', width=w, height=h, scale=max(1, sc))

        # Render figures to images (bytes) using kaleido, sized to fit square layout
        try:
            img_daily_b = _render_bytes(fd, left_w, daily_h, scale)
            img_weekly_b = _render_bytes(fw, right_w_each, bottom_h, scale)
            img_monthly_b = _render_bytes(fm, right_w_each, bottom_h, scale)
        except Exception:
            # Fallback: reduce size/scale and retry once
            img_daily_b = _render_bytes(fd, left_w//2, daily_h//2, 1)
            img_weekly_b = _render_bytes(fw, right_w_each//2, bottom_h//2, 1)
            img_monthly_b = _render_bytes(fm, right_w_each//2, bottom_h//2, 1)

        img_daily = Image.open(BytesIO(img_daily_b)).convert('RGB')
        img_weekly = Image.open(BytesIO(img_weekly_b)).convert('RGB')
        img_monthly = Image.open(BytesIO(img_monthly_b)).convert('RGB')

        # Square canvas (W x W)
        canvas = Image.new('RGB', (size, size), color=bg_color)
        # Paste: daily spans full top row; weekly bottom-left; monthly bottom-right
        canvas.paste(img_daily, (0, 0))
        canvas.paste(img_weekly, (0, daily_h))
        canvas.paste(img_monthly, (right_w_each, daily_h))

        canvas.save(out_path, format='JPEG', quality=90)
        return True, out_path
    except Exception as e:
        st.error(f"통합 차트 저장 실패: {e}")
        st.info("이미지 저장을 위해 'kaleido'와 'pillow' 패키지가 필요합니다. 설치: pip install -U kaleido pillow")
        return False, str(e)

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
        # Prefer broadly available v1 models by default
        self.primary_model = 'gemini-2.0-flash'
        self.fallback_model = 'gemini-2.5-flash'

        if not self.api_keys:
            st.warning("Gemini API 키를 찾을 수 없어 AI 분석 기능이 제한됩니다. 환경 변수 또는 config.ini 파일을 확인해주세요.")
        else:
            self._initialize_client()

    def _initialize_client(self):
        """Initializes the Gemini client with the current API key."""
        if self.current_key_index < len(self.api_keys):
            current_key = self.api_keys[self.current_key_index]
            try:
                self.client = genai.Client(api_key=current_key, http_options={"api_version": "v1"})
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
        from google.genai.types import GenerateContentConfig

        full_prompt = f"{system_instruction}\n\n{prompt}"
        # Minimal config: remove tool usage to avoid INVALID_ARGUMENT on v1
        config = None

        # Store the starting key index for this request
        start_index = self.current_key_index

        while self.current_key_index < len(self.api_keys):
            if not self.client and not self._initialize_client():
                # If we can't even initialize a client with the current key, try the next one.
                if not self.try_next_key():
                    break # No more keys to try

            # 1. Try with the primary model (Pro)
            try:
                st.session_state.analyst_model = self.primary_model
                st.info(f"API Key #{self.current_key_index + 1}을(를) 사용하여 '{st.session_state.analyst_model}' 모델로 분석을 시도합니다.")
                response = self.client.models.generate_content(
                    model=self.primary_model,
                    contents=full_prompt,
                    config=config
                )
                return response # Success
            except ResourceExhausted:
                st.warning(f"Key #{self.current_key_index + 1}의 '{self.primary_model}' 모델 사용량이 소진되었습니다. 폴백 모델로 전환합니다.")
                # 2. If exhausted, try with the fallback model (Flash) with the SAME key
                try:
                    st.session_state.analyst_model = self.fallback_model
                    st.info(f"동일한 API Key #{self.current_key_index + 1}을(를) 사용하여 '{st.session_state.analyst_model}' 모델로 재시도합니다.")
                    response = self.client.models.generate_content(
                        model=self.fallback_model,
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
                """Convert API numeric string to float safely.
                - Preserve '-' sign, strip leading '+'
                - Remove thousand separators and percent symbols
                - Fallback to 0.0 if conversion fails
                """
                try:
                    s = str(value_str).strip()
                    if not s:
                        return 0.0
                    s = s.replace(',', '').replace('%', '')
                    if s.startswith('+'):
                        s = s[1:]
                    return float(s)
                except Exception:
                    try:
                        return float(value_str)
                    except Exception:
                        return 0.0

            # Try to infer market classification if provided
            market_raw = data.get('mkt_cls') or data.get('isu_mkt_cls') or data.get('market') or data.get('mkt_type') or data.get('mkt_div')
            market = str(market_raw).upper() if market_raw is not None else ''
            if 'KOSPI' in market or '코스피' in market:
                market_std = '코스피'
            elif 'KOSDAQ' in market or '코스닥' in market:
                market_std = '코스닥'
            else:
                market_std = 'UNKNOWN'

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
                'market': market_std,
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
        """Justified PBR with guards: PBR = (ROE - g) / (Ke - g).
        If Ke <= g, or result < 0, return 0.0 to avoid nonsensical values.
        Inputs are percent.
        """
        roe_pct, ke_pct, g_pct = roe / 100, cost_of_equity / 100, terminal_growth / 100
        denom = ke_pct - g_pct
        if denom <= 0:
            return 0.0
        pbr = (roe_pct - g_pct) / denom
        return pbr if pbr > 0 else 0.0

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
    info = st.session_state.kiwoom_handler.get_stock_info(stock_code)
    # Persist basic instrument metadata if available
    try:
        name = None
        try:
            df_list = get_stock_list()
            if not df_list.empty:
                row = df_list[df_list['code'] == str(stock_code)].iloc[0]
                name = row['name']
        except Exception:
            name = None
        upsert_instrument_metadata(stock_code, name=name, market=info.get('market'))
    except Exception:
        pass
    return info

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
    # Reset valuation widgets so new company defaults apply
    for key in (
        'est_bps_input',
        'est_roe_slider',
        'cost_of_equity_slider',
        'terminal_growth_slider',
    ):
        if key in st.session_state:
            del st.session_state[key]

# --- Batch Helpers ---

@st.cache_data(ttl=3600)
def load_stock_listing():
    try:
        df = pd.read_csv(os.path.join(APP_DIR, 'stock_list.csv'), dtype={'code': str, 'name': str})
        if 'name' in df.columns and 'code' in df.columns:
            return df[['name', 'code']]
        return pd.DataFrame(columns=['name', 'code'])
    except Exception:
        return pd.DataFrame(columns=['name', 'code'])

def collect_stock_metadata(delay_sec=0.5, limit=None):
    df_list = load_stock_listing()
    if df_list.empty:
        st.error('종목 리스트를 불러오지 못했습니다.')
        return pd.DataFrame()
    rows = []
    total = len(df_list) if limit is None else min(limit, len(df_list))
    prog = st.progress(0.0, text='시가총액/시장 정보 수집 중...')
    for idx, row in enumerate(df_list.itertuples(index=False)):
        if limit is not None and idx >= limit:
            break
        name, code = row.name, row.code
        try:
            info = get_kiwoom_stock_info(code)
            mcap = info.get('market_cap', 0)
            market = info.get('market', 'UNKNOWN')
            rows.append({'name': name, 'code': code, 'market': market, 'market_cap': mcap})
        except Exception:
            rows.append({'name': name, 'code': code, 'market': 'UNKNOWN', 'market_cap': 0})
        time.sleep(max(0.0, float(delay_sec)))
        prog.progress((idx + 1) / total, text=f'수집 중... {idx+1}/{total}')
    meta = pd.DataFrame(rows)
    try:
        os.makedirs(os.path.join(ROOT_DIR, 'cache'), exist_ok=True)
        meta.to_csv(os.path.join(ROOT_DIR, 'cache', 'stock_market_caps.csv'), index=False, encoding='utf-8-sig')
    except Exception:
        pass
    return meta

def build_candlestick_figure(df, title, height=300, rangebreaks=None):
    if df.empty or not all(c in df.columns for c in ['open', 'high', 'low', 'close']):
        return None
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        subplot_titles=(title, '거래량'), row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='캔들'), row=1, col=1)
    if 'volume' in df.columns:
        colors = ['red' if c < o else 'green' for o, c in zip(df['open'], df['close'])]
        fig.add_trace(go.Bar(x=df.index, y=df['volume'], name='거래량', marker_color=colors), row=2, col=1)
    if rangebreaks:
        fig.update_xaxes(rangebreaks=rangebreaks)
    fig.update_layout(xaxis_rangeslider_visible=False, showlegend=True, height=height, margin=dict(l=10, r=10, b=10, t=40))
    fig.update_layout(legend=dict(x=0.01, y=0.99, xanchor='left', yanchor='top', bgcolor='rgba(255,255,255,0.6)', bordercolor='rgba(0,0,0,0.1)', borderwidth=1, font=dict(size=10)))
    return fig

def generate_and_save_combined_chart_headless(stock_code, company_name, save_size=1400):
    if 'kiwoom_handler' not in st.session_state:
        return False, 'Kiwoom handler not initialized'
    rangebreaks = [dict(bounds=["sat", "mon"])]
    now = pd.Timestamp.now()
    start_3m = now - pd.DateOffset(months=3)
    # Load daily for last 3 months (daily panel)
    df_daily = load_daily_candles_from_store(stock_code, start_3m, now)
    if df_daily is None or df_daily.empty or not _has_coverage(df_daily, start_3m, now):
        # Fetch from API and persist
        api_daily, api_weekly, api_monthly = st.session_state.kiwoom_handler.fetch_all_chart_data(stock_code)
        df_daily = api_daily
        try:
            upsert_daily_candles_store(stock_code, api_daily)
        except Exception:
            pass
        df_weekly, df_monthly = api_weekly, api_monthly
    else:
        # Derive weekly/monthly from a wider daily window (up to 3 years)
        try:
            start_3y = now - pd.DateOffset(years=3)
            df_daily_full = load_daily_candles_from_store(stock_code, start_3y, now)
        except Exception:
            df_daily_full = pd.DataFrame()
        if df_daily_full is None or df_daily_full.empty:
            # Fallback to API to get weekly/monthly directly
            _, df_weekly, df_monthly = st.session_state.kiwoom_handler.fetch_all_chart_data(stock_code)
        else:
            df_weekly, df_monthly = derive_weekly_monthly_from_daily(df_daily_full)
        # Ensure monthly truly has monthly frequency
        try:
            df_monthly = ensure_monthly_frequency(df_monthly, df_daily_full, df_weekly)
        except Exception:
            pass

    df_daily_filtered = df_daily[df_daily.index >= start_3m]
    # Build daily with investor cumulative lines (3 rows)
    fig_daily = None
    if not df_daily_filtered.empty and all(c in df_daily_filtered.columns for c in ['open','high','low','close']):
        # Investor flows: prefer store for same window; else fetch and persist
        try:
            df_investor = load_investor_flows_from_store(stock_code, df_daily_filtered.index.min(), df_daily_filtered.index.max())
            if df_investor is None or df_investor.empty:
                df_investor = st.session_state.kiwoom_handler.fetch_investor_data(stock_code)
                try:
                    upsert_investor_flows_store(stock_code, df_investor)
                except Exception:
                    pass
        except Exception:
            df_investor = pd.DataFrame()
        """ Disabled due to encoding issues
        fig_daily = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                                  subplot_titles=(f'{company_name} 일봉 (3개월)', '거래량', '투자자별 누적 순매수(백만원)'),
                                  row_heights=[0.5, 0.2, 0.3])
        fig_daily.add_trace(go.Candlestick(x=df_daily_filtered.index, open=df_daily_filtered['open'], high=df_daily_filtered['high'], low=df_daily_filtered['low'], close=df_daily_filtered['close'], name='캔들'), row=1, col=1)
        if 'volume' in df_daily_filtered.columns:
            colors = ['red' if c < o else 'green' for o, c in zip(df_daily_filtered['open'], df_daily_filtered['close'])]
            fig_daily.add_trace(go.Bar(x=df_daily_filtered.index, y=df_daily_filtered['volume'], name='거래량', marker_color=colors), row=2, col=1)
        if not df_investor.empty and all(c in df_investor.columns for c in ['ind_netprps','for_netprps','orgn_netprps']):
            inv = df_investor[df_investor.index.isin(df_daily_filtered.index)].copy()
            inv['ind_cumulative'] = inv['ind_netprps'].cumsum()
            inv['for_cumulative'] = inv['for_netprps'].cumsum()
            inv['orgn_cumulative'] = inv['orgn_netprps'].cumsum()
            fig_daily.add_trace(go.Scatter(x=inv.index, y=inv['ind_cumulative'], mode='lines', name='개인(누적)', line=dict(color='#2ca02c', width=2)), row=3, col=1)
            fig_daily.add_trace(go.Scatter(x=inv.index, y=inv['for_cumulative'], mode='lines', name='외국인(누적)', line=dict(color='red', width=4)), row=3, col=1)
            fig_daily.add_trace(go.Scatter(x=inv.index, y=inv['orgn_cumulative'], mode='lines', name='기관(누적)', line=dict(color='#ff7f0e', width=2, dash='dash')), row=3, col=1)
            fig_daily.update_layout(yaxis3_title_text='누적 순매수 금액')
        fig_daily.update_xaxes(rangebreaks=rangebreaks)
        fig_daily.update_layout(xaxis_rangeslider_visible=False, showlegend=True, height=600, margin=dict(l=10, r=10, b=10, t=40))
        fig_daily.update_layout(legend=dict(x=0.01, y=0.99, xanchor='left', yanchor='top', bgcolor='rgba(255,255,255,0.6)', bordercolor='rgba(0,0,0,0.1)', borderwidth=1, font=dict(size=10)))
        """

        # Rebuilt (English) daily chart block
        fig_daily = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(
                f"{company_name} Daily (3M)",
                "Volume",
                "Investor net buy/sell (million KRW)"
            ),
            row_heights=[0.5, 0.2, 0.3]
        )
        fig_daily.add_trace(
            go.Candlestick(
                x=df_daily_filtered.index,
                open=df_daily_filtered['open'],
                high=df_daily_filtered['high'],
                low=df_daily_filtered['low'],
                close=df_daily_filtered['close'],
                name='Candles'
            ),
            row=1,
            col=1
        )
        if 'volume' in df_daily_filtered.columns:
            colors = ['red' if c < o else 'green' for o, c in zip(df_daily_filtered['open'], df_daily_filtered['close'])]
            fig_daily.add_trace(
                go.Bar(x=df_daily_filtered.index, y=df_daily_filtered['volume'], name='Volume', marker_color=colors),
                row=2,
                col=1
            )
        if not df_investor.empty and all(c in df_investor.columns for c in ['ind_netprps','for_netprps','orgn_netprps']):
            inv = df_investor[df_investor.index.isin(df_daily_filtered.index)].copy()
            inv['ind_cumulative'] = inv['ind_netprps'].cumsum()
            inv['for_cumulative'] = inv['for_netprps'].cumsum()
            inv['orgn_cumulative'] = inv['orgn_netprps'].cumsum()
            fig_daily.add_trace(go.Scatter(x=inv.index, y=inv['ind_cumulative'], mode='lines', name='Individuals (cum)', line=dict(color='#2ca02c', width=2)), row=3, col=1)
            fig_daily.add_trace(go.Scatter(x=inv.index, y=inv['for_cumulative'], mode='lines', name='Foreigners (cum)', line=dict(color='red', width=4)), row=3, col=1)
            fig_daily.add_trace(go.Scatter(x=inv.index, y=inv['orgn_cumulative'], mode='lines', name='Institutions (cum)', line=dict(color='#ff7f0e', width=2, dash='dash')), row=3, col=1)
            fig_daily.update_layout(yaxis3_title_text='Net buy/sell amount')
        fig_daily.update_xaxes(rangebreaks=rangebreaks)
        fig_daily.update_layout(xaxis_rangeslider_visible=False, showlegend=True, height=600, margin=dict(l=10, r=10, b=10, t=40))
        fig_daily.update_layout(legend=dict(x=0.01, y=0.99, xanchor='left', yanchor='top', bgcolor='rgba(255,255,255,0.6)', bordercolor='rgba(0,0,0,0.1)', borderwidth=1, font=dict(size=10)))

    fig_weekly = None
    if not df_weekly.empty:
        dfw = df_weekly[df_weekly.index >= (pd.Timestamp.now() - pd.DateOffset(years=1))]
        fig_weekly = build_candlestick_figure(dfw, f'{company_name} 주봉 (1년)', height=300, rangebreaks=rangebreaks)

    fig_monthly = None
    if not df_monthly.empty:
        dfm = df_monthly[df_monthly.index >= (pd.Timestamp.now() - pd.DateOffset(years=3))]
        fig_monthly = build_candlestick_figure(dfm, f'{company_name} 월봉 (3년)', height=300, rangebreaks=rangebreaks)

    # Normalize weekly/monthly titles to English
    try:
        if fig_weekly is not None:
            fig_weekly.update_layout(title=f'{company_name} Weekly (1Y)')
        if fig_monthly is not None:
            fig_monthly.update_layout(title=f'{company_name} Monthly (3Y)')
    except Exception:
        pass

    if not fig_daily or not fig_weekly or not fig_monthly:
        return False, 'Insufficient data to build figures'

    # Save combined with square layout
    ts = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    fname = f"{_safe_filename(company_name)}_{ts}.jpg"
    out_path = os.path.join(ROOT_DIR, 'reports', 'charts', fname)
    ok, saved = save_combined_charts_as_jpg(fig_daily, fig_weekly, fig_monthly, out_path, size=save_size)

    # Persist normalized, date-keyed daily data to the store
    try:
        # Update instrument metadata with provided name (market optional)
        try:
            upsert_instrument_metadata(stock_code, company_name)
        except Exception:
            pass
        upsert_daily_candles_store(stock_code, df_daily)
        try:
            if 'df_investor' in locals() and isinstance(df_investor, pd.DataFrame) and not df_investor.empty:
                upsert_investor_flows_store(stock_code, df_investor)
        except Exception:
            pass
    except Exception:
        # Ignore store write errors; chart export already succeeded
        pass

    return ok, saved

# --- Similarity Search (Time-Series) ---

def _similarity_dir() -> str:
    return os.path.join(_data_base_dir(), 'similarity')

def _similarity_index_path(window: int, feature: str) -> str:
    safe_feature = re.sub(r'[^a-zA-Z0-9_-]', '', str(feature or 'returns'))
    return os.path.join(_similarity_dir(), f'index-{int(window)}-{safe_feature}.json')

def _vectorize_series_close(df: pd.DataFrame, window: int) -> (List[str], np.ndarray):
    if df is None or df.empty or 'close' not in df.columns:
        return [], np.array([])
    s = df['close'].dropna().tail(int(window))
    if len(s) < max(3, int(window * 0.8)):
        return [], np.array([])
    std = s.std()
    z = (s - s.mean()) / (std if std and std > 0 else 1.0)
    dates = [pd.Timestamp(i).strftime('%Y-%m-%d') for i in s.index]
    return dates, z.values.astype(float)

def _vectorize_series_returns(df: pd.DataFrame, window: int) -> (List[str], np.ndarray):
    if df is None or df.empty or 'close' not in df.columns:
        return [], np.array([])
    c = df['close'].dropna()
    if c.empty:
        return [], np.array([])
    r = np.log(c).diff().dropna()
    r = r.tail(int(window))
    if len(r) < max(3, int(window * 0.8)):
        return [], np.array([])
    std = r.std()
    z = (r - r.mean()) / (std if std and std > 0 else 1.0)
    dates = [pd.Timestamp(i).strftime('%Y-%m-%d') for i in r.index]
    return dates, z.values.astype(float)

def _vectorize(df: pd.DataFrame, window: int, feature: str) -> (List[str], np.ndarray):
    f = (feature or 'returns').lower()
    if f == 'price':
        return _vectorize_series_close(df, window)
    return _vectorize_series_returns(df, window)

def _corr_distance(x: np.ndarray, y: np.ndarray) -> float:
    try:
        if x.size == 0 or y.size == 0:
            return float('inf')
        xm = x - np.mean(x)
        ym = y - np.mean(y)
        xs = np.std(xm)
        ys = np.std(ym)
        if xs == 0 or ys == 0:
            return float('inf')
        corr = float(np.dot(xm, ym) / (len(xm) * xs * ys))
        corr = max(-1.0, min(1.0, corr))
        return 1.0 - corr
    except Exception:
        return float('inf')

@st.cache_data(ttl=3600)
def build_similarity_index(window: int = 60, feature: str = 'returns', market: str = None) -> Dict[str, Any]:
    """Build a lightweight similarity index from the local store.
    - window: number of recent points per vector
    - feature: 'returns' (default) or 'price'
    - market: optional filter, expects values like '코스피', '코스닥', 'UNKNOWN', or None for all
    Returns index dict and also persists to @data/similarity.
    """
    try:
        _ensure_dirs(_similarity_dir())
    except Exception:
        pass

    now = pd.Timestamp.now()
    df_list = get_stock_list()
    if df_list is None or df_list.empty or 'code' not in df_list.columns:
        return {'window': int(window), 'feature': feature, 'vectors': {}, 'updated_at': now.isoformat()}

    meta_path = os.path.join(ROOT_DIR, 'cache', 'stock_market_caps.csv')
    meta = None
    try:
        if os.path.exists(meta_path):
            meta = pd.read_csv(meta_path, dtype={'code': str, 'name': str})
    except Exception:
        meta = None

    universe = df_list.copy()
    if isinstance(meta, pd.DataFrame) and not meta.empty and 'market' in meta.columns:
        universe = universe.merge(meta[['code', 'market']], on='code', how='left')
        if market and market != '전체':
            universe = universe[universe['market'] == market]

    vectors: Dict[str, Dict[str, Any]] = {}
    start = now - pd.Timedelta(days=int(window * 3))
    for row in universe.itertuples(index=False):
        try:
            code = getattr(row, 'code')
        except Exception:
            continue
        try:
            df = load_daily_candles_from_store(code, start, now)
            dates, vec = _vectorize(df, int(window), feature)
            if len(dates) >= max(3, int(window * 0.8)):
                vectors[str(code)] = {'dates': dates, 'vec': [float(v) for v in vec]}
        except Exception:
            continue

    index = {
        'schema_version': '1.0.0',
        'window': int(window),
        'feature': str(feature or 'returns'),
        'vectors': vectors,
        'updated_at': now.isoformat()
    }
    try:
        _save_json_atomic(_similarity_index_path(window, feature), index)
    except Exception:
        pass
    return index

def _load_similarity_index(window: int, feature: str) -> Dict[str, Any]:
    path = _similarity_index_path(window, feature)
    obj = _load_json(path)
    return obj if obj else {}

def find_similar(query_code: str, topn: int = 20, window: int = 60, feature: str = 'returns', market: str = None) -> List[Dict[str, Any]]:
    """Return a sorted list of similar tickers to query_code using correlation distance."""
    try:
        idx = _load_similarity_index(window, feature)
        if not idx or 'vectors' not in idx:
            idx = build_similarity_index(window=window, feature=feature, market=market)
    except Exception:
        idx = build_similarity_index(window=window, feature=feature, market=market)

    now = pd.Timestamp.now()
    start = now - pd.Timedelta(days=int(window * 3))
    df_q = load_daily_candles_from_store(query_code, start, now)
    q_dates, q_vec = _vectorize(df_q, int(window), feature)
    if len(q_dates) < max(3, int(window * 0.8)):
        return []

    q_map = {d: float(v) for d, v in zip(q_dates, q_vec)}
    min_overlap = max(3, int(window * 0.8))

    results = []
    for code, entry in idx.get('vectors', {}).items():
        if str(code) == str(query_code):
            continue
        try:
            cd = entry.get('dates') or []
            cv = entry.get('vec') or []
            if not cd or not cv:
                continue
            common = sorted(set(cd).intersection(q_dates))
            if len(common) < min_overlap:
                continue
            x = np.array([q_map[d] for d in common], dtype=float)
            c_map = {d: float(v) for d, v in zip(cd, cv)}
            y = np.array([c_map[d] for d in common], dtype=float)
            dist = _corr_distance(x, y)
            if not np.isfinite(dist):
                continue
            results.append({'code': code, 'distance': float(dist), 'overlap': int(len(common))})
        except Exception:
            continue
    results.sort(key=lambda r: (r['distance'], -r['overlap']))
    return results[:int(topn)]

def _overlay_figure_for_pair(query_code: str, other_code: str, window: int, feature: str, title_map: Dict[str, str]) -> Any:
    """Small overlay plot (query vs. candidate) normalized on overlapping dates."""
    now = pd.Timestamp.now()
    start = now - pd.Timedelta(days=int(window * 3))
    df_q = load_daily_candles_from_store(query_code, start, now)
    df_o = load_daily_candles_from_store(other_code, start, now)
    if df_q is None or df_q.empty or df_o is None or df_o.empty:
        return None
    qd, qv = _vectorize(df_q, int(window), 'price') if 'close' in df_q.columns else _vectorize(df_q, int(window), feature)
    od, ov = _vectorize(df_o, int(window), 'price') if 'close' in df_o.columns else _vectorize(df_o, int(window), feature)
    if not qd or not od:
        return None
    common = sorted(set(qd).intersection(od))
    if len(common) < max(3, int(window * 0.8)):
        return None
    q_map = {d: float(v) for d, v in zip(qd, qv)}
    o_map = {d: float(v) for d, v in zip(od, ov)}
    xs = common
    y1 = [q_map[d] for d in xs]
    y2 = [o_map[d] for d in xs]
    fig = go.Figure()
    q_title = title_map.get(str(query_code), str(query_code))
    o_title = title_map.get(str(other_code), str(other_code))
    fig.add_trace(go.Scatter(x=xs, y=y1, mode='lines', name=q_title, line=dict(color='#1f77b4', width=2)))
    fig.add_trace(go.Scatter(x=xs, y=y2, mode='lines', name=o_title, line=dict(color='#ff7f0e', width=2, dash='dash')))
    fig.update_layout(height=220, margin=dict(l=10, r=10, t=30, b=10), showlegend=False, title=f"{q_title} vs {o_title}")
    return fig

# --- Plotting Functions ---

def display_candlestick_chart(stock_code, company_name):
    # Remember selection and keep charts visible across reruns
    st.session_state['_charts_rendered_this_run'] = True
    st.session_state['show_charts'] = True
    st.session_state['last_stock_code'] = stock_code
    st.session_state['last_company_name'] = company_name
    if 'kiwoom_handler' not in st.session_state:
        st.error("차트를 생성하려면 Kiwoom 핸들러가 필요합니다.")
        return

    with st.spinner("주가 및 투자자 동향 데이터를 조회하고 차트를 생성 중입니다..."):
        rangebreaks = [dict(bounds=["sat", "mon"])]  # 주말 제외
        # Prefer locally stored data for daily; derive weekly/monthly when possible
        now = pd.Timestamp.now()
        start_3m = now - pd.DateOffset(months=3)
        df_daily = load_daily_candles_from_store(stock_code, start_3m, now)
        if df_daily is None or df_daily.empty or not _has_coverage(df_daily, start_3m, now):
            # Fallback to API and persist for reuse
            df_daily_api, df_weekly_api, df_monthly_api = st.session_state.kiwoom_handler.fetch_all_chart_data(stock_code)
            df_daily = df_daily_api
            try:
                upsert_daily_candles_store(stock_code, df_daily_api)
            except Exception:
                pass
            df_weekly, df_monthly = df_weekly_api, df_monthly_api
        else:
            # Derive weekly/monthly from a wider daily window (up to 3 years)
            try:
                now = pd.Timestamp.now()
                start_3y = now - pd.DateOffset(years=3)
                df_daily_full = load_daily_candles_from_store(stock_code, start_3y, now)
            except Exception:
                df_daily_full = pd.DataFrame()
            if df_daily_full is None or df_daily_full.empty:
                # Fallback to API for weekly/monthly if store lacks coverage
                _, df_weekly, df_monthly = st.session_state.kiwoom_handler.fetch_all_chart_data(stock_code)
            else:
                df_weekly, df_monthly = derive_weekly_monthly_from_daily(df_daily_full)
            # Ensure monthly truly has monthly frequency
            try:
                df_monthly = ensure_monthly_frequency(df_monthly, df_daily_full, df_weekly)
            except Exception:
                pass

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
                    # Try to reuse investor flows from store for the same daily range
                    inv_store = load_investor_flows_from_store(stock_code, df_daily_filtered.index.min(), df_daily_filtered.index.max())
                    if isinstance(inv_store, pd.DataFrame) and not inv_store.empty:
                        df_investor = inv_store
                    else:
                        df_investor = st.session_state.kiwoom_handler.fetch_investor_data(stock_code)
                        # Persist investor flows (date-keyed) for reuse
                        try:
                            upsert_investor_flows_store(stock_code, df_investor)
                        except Exception:
                            pass
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
                    fig_daily.update_layout(legend=dict(x=0.01, y=0.99, xanchor='left', yanchor='top', bgcolor='rgba(255,255,255,0.6)', bordercolor='rgba(0,0,0,0.1)', borderwidth=1, font=dict(size=10)))
                    # Colorize investor cumulative lines (last 3 traces)
                    try:
                        tr_ind, tr_for, tr_org = fig_daily.data[-3], fig_daily.data[-2], fig_daily.data[-1]
                        tr_ind.update(line=dict(color='#2ca02c', width=2), name='개인(누적)')
                        tr_for.update(line=dict(color='#1f77b4', width=2), name='외국인(누적)')
                        tr_org.update(line=dict(color='#ff7f0e', width=2, dash='dash'), name='기관(누적)')
                    except Exception:
                        pass
                    st.plotly_chart(fig_daily, use_container_width=True)
                    # Keep reference for combined export
                    st.session_state._last_fig_daily = fig_daily

        with col2:
            st.subheader("주봉 & 월봉")
            # --- Weekly & Monthly Charts ---
            chart_data = {
                '주봉 (1년)': (df_weekly[df_weekly.index >= (pd.Timestamp.now() - pd.DateOffset(years=1))], 300),
                '월봉 (3년)': (df_monthly[df_monthly.index >= (pd.Timestamp.now() - pd.DateOffset(years=3))], 300)
            }

            for _idx, (title, (df, height)) in enumerate(chart_data.items()):
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
                fig_period.update_layout(legend=dict(x=0.01, y=0.99, xanchor='left', yanchor='top', bgcolor='rgba(255,255,255,0.6)', bordercolor='rgba(0,0,0,0.1)', borderwidth=1, font=dict(size=10)))
                st.plotly_chart(fig_period, use_container_width=True)
                # Keep references for combined export
                if _idx == 0:
                    st.session_state._last_fig_weekly = fig_period
                elif _idx == 1:
                    st.session_state._last_fig_monthly = fig_period
        

        # Auto-save combined JPG once all three charts exist
        has_daily = '_last_fig_daily' in st.session_state
        has_weekly = '_last_fig_weekly' in st.session_state
        has_monthly = '_last_fig_monthly' in st.session_state
        if has_daily and has_weekly and has_monthly:
            ts_tag = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            base_name = f"{_safe_filename(company_name)}_{ts_tag}.jpg"
            if st.session_state.get('last_combined_saved') != base_name:
                out_path = os.path.join(ROOT_DIR, 'reports', 'charts', base_name)
                ok, saved = save_combined_charts_as_jpg(
                    st.session_state._last_fig_daily,
                    st.session_state._last_fig_weekly,
                    st.session_state._last_fig_monthly,
                    out_path
                )
                if ok:
                    st.session_state['last_combined_saved'] = base_name
                    st.success(f"통합 차트 자동 저장: {out_path}")
# --- Data Similarity (Stored Data, non-chart) ---

@st.cache_data(ttl=3600)
def _compute_features_from_store(code: str, window_days: int = 120) -> Dict[str, float]:
    """Compute data-driven features from locally stored JSON (candles + investor flows)."""
    try:
        now = pd.Timestamp.now()
        start = now - pd.Timedelta(days=int(window_days * 3))
        df = load_daily_candles_from_store(code, start, now)
        if df is None or df.empty or 'close' not in df.columns:
            return {}
        df = df.sort_index()
        c = df['close'].dropna()
        if c.empty:
            return {}
        r = np.log(c).diff().dropna()

        def _cum_return(lookback: int) -> float:
            s = c.tail(int(lookback))
            return float((s.iloc[-1] / s.iloc[0]) - 1.0) if len(s) >= 2 else float('nan')

        def _vol(lookback: int) -> float:
            s = r.tail(int(lookback))
            return float(s.std()) if len(s) >= 2 else float('nan')

        def _mdd(lookback: int) -> float:
            s = c.tail(int(lookback))
            if len(s) < 2:
                return float('nan')
            roll_max = s.cummax()
            dd = (s / roll_max - 1.0).min()
            return float(dd)

        def _avg_vol(lookback: int) -> float:
            if 'volume' not in df.columns:
                return float('nan')
            s = df['volume'].dropna().tail(int(lookback))
            return float(s.mean()) if not s.empty else float('nan')

        # Investor flows
        df_inv = load_investor_flows_from_store(code, start, now)
        def _inv_sum(col: str, lookback: int) -> float:
            if df_inv is None or df_inv.empty or col not in df_inv.columns:
                return float('nan')
            s = df_inv[col].dropna()
            cutoff = now - pd.Timedelta(days=int(lookback))
            s = s[s.index >= cutoff]
            return float(s.sum()) if not s.empty else float('nan')

        feats = {
            'ret20': _cum_return(20),
            'ret60': _cum_return(60),
            'vol20': _vol(20),
            'vol60': _vol(60),
            'mdd60': _mdd(60),
            'avg_vol20': _avg_vol(20),
            'inv_for20': _inv_sum('for_netprps', 20),
            'inv_for60': _inv_sum('for_netprps', 60),
            'inv_org60': _inv_sum('orgn_netprps', 60),
            'inv_ind60': _inv_sum('ind_netprps', 60),
        }
        valid = {k: v for k, v in feats.items() if v == v and np.isfinite(v)}
        return valid if len(valid) >= 3 else {}
    except Exception:
        return {}

@st.cache_data(ttl=3600)
def build_data_similarity_index(window_days: int = 120, market: str = None) -> Dict[str, Any]:
    now = pd.Timestamp.now()
    df_list = get_stock_list()
    if df_list is None or df_list.empty or 'code' not in df_list.columns:
        return {'window_days': int(window_days), 'features': {}, 'stats': {}, 'updated_at': now.isoformat()}

    universe = df_list.copy()
    try:
        meta_path = os.path.join(ROOT_DIR, 'cache', 'stock_market_caps.csv')
        if market and market != '전체' and os.path.exists(meta_path):
            meta_df = pd.read_csv(meta_path, dtype={'code': str, 'name': str})
            if 'market' in meta_df.columns:
                universe = universe.merge(meta_df[['code', 'market']], on='code', how='left')
                universe = universe[universe['market'] == market]
    except Exception:
        pass

    feats_map: Dict[str, Dict[str, float]] = {}
    for row in universe.itertuples(index=False):
        try:
            code = getattr(row, 'code')
        except Exception:
            continue
        f = _compute_features_from_store(str(code), int(window_days))
        if f:
            feats_map[str(code)] = f

    if not feats_map:
        return {'window_days': int(window_days), 'features': {}, 'stats': {}, 'updated_at': now.isoformat()}
    df_feats = pd.DataFrame.from_dict(feats_map, orient='index')
    stats = {}
    for col in df_feats.columns:
        s = df_feats[col]
        mu = float(s.mean(skipna=True)) if len(s) else 0.0
        sd = float(s.std(skipna=True)) if len(s) else 1.0
        stats[col] = {'mean': mu, 'std': sd if sd and sd > 0 else 1.0}

    return {'schema_version': '1.0.0', 'window_days': int(window_days), 'features': feats_map, 'stats': stats, 'updated_at': now.isoformat()}

def _standardize_vector(feats: Dict[str, float], stats: Dict[str, Dict[str, float]], order: List[str]) -> np.ndarray:
    vals = []
    for k in order:
        v = feats.get(k, float('nan'))
        mu = stats.get(k, {}).get('mean', 0.0)
        sd = stats.get(k, {}).get('std', 1.0)
        if v != v or not np.isfinite(v):
            z = 0.0
        else:
            z = (float(v) - mu) / (sd if sd else 1.0)
        vals.append(z)
    return np.array(vals, dtype=float)

def find_similar_by_data(query_code: str, topn: int = 20, window_days: int = 120, market: str = None) -> List[Dict[str, Any]]:
    idx = build_data_similarity_index(window_days=int(window_days), market=market)
    feats_map = idx.get('features', {})
    stats = idx.get('stats', {})
    if not feats_map or str(query_code) not in feats_map:
        return []
    feature_order = sorted(stats.keys())
    q = _standardize_vector(feats_map[str(query_code)], stats, feature_order)
    out = []
    for code, f in feats_map.items():
        if str(code) == str(query_code):
            continue
        v = _standardize_vector(f, stats, feature_order)
        dist = float(np.linalg.norm(q - v))
        if np.isfinite(dist):
            out.append({'code': code, 'distance': dist})
    out.sort(key=lambda r: r['distance'])
    return out[:int(topn)]

# --- Main Application ---

def main():
    st.set_page_config(layout="wide")
    today_str = pd.Timestamp.now().strftime('%Y-%m-%d')
    # Reset per-run render guard
    st.session_state['_charts_rendered_this_run'] = False

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

    # --- Data Similarity (Stored Data) ---
    st.header("8. 유사 종목 찾기 (Data Similarity)")
    with st.container(border=True):
        sim_cols = st.columns([1,1,1,1,1])
        window_days = sim_cols[0].selectbox('기간(일)', options=[60, 120, 180], index=1)
        topn = int(sim_cols[1].number_input('Top N', min_value=5, max_value=50, value=20, step=1))
        # Market filter if meta available
        market_opt = '전체'
        try:
            meta_path = os.path.join(ROOT_DIR, 'cache', 'stock_market_caps.csv')
            if os.path.exists(meta_path):
                meta_df = pd.read_csv(meta_path, dtype={'code': str, 'name': str})
                if 'market' in meta_df.columns:
                    markets = ['전체'] + sorted([m for m in meta_df['market'].dropna().unique().tolist()])
                    market_opt = sim_cols[2].selectbox('시장', options=markets, index=0)
        except Exception:
            pass

        if sim_cols[4].button('유사 종목 찾기', use_container_width=True):
            if not stock_code:
                st.error('종목을 먼저 선택하세요')
            else:
                with st.spinner('저장 데이터 기반 유사도 계산 중...'):
                    res = find_similar_by_data(stock_code, topn=topn, window_days=window_days, market=market_opt if market_opt != '전체' else None)

            # Map code->name
            name_map = {}
            try:
                df_l = get_stock_list()
                if isinstance(df_l, pd.DataFrame) and not df_l.empty:
                    name_map = {str(r.code): str(r.name) for r in df_l.itertuples(index=False) if getattr(r, 'code', None) is not None}
            except Exception:
                name_map = {}

            if not res:
                st.warning('유사 결과가 없습니다. 저장 데이터 커버리지를 확인하세요.')
            else:
                st.subheader('유사 종목 목록 (데이터 기반)')
                out_rows = []
                for i, r in enumerate(res, start=1):
                    c = str(r['code'])
                    out_rows.append({'rank': i, 'name': name_map.get(c, '-'), 'code': c, 'distance': round(float(r['distance']), 4)})
                st.dataframe(pd.DataFrame(out_rows), use_container_width=True, hide_index=True)

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
                    # Ensure valuation widgets reflect latest fetched values
                    try:
                        st.session_state['est_bps_input'] = int(float(kiwoom_data.get('bps', 0)))
                    except Exception:
                        st.session_state['est_bps_input'] = int(kiwoom_data.get('bps', 0) or 0)
                    try:
                        st.session_state['est_roe_slider'] = float(kiwoom_data.get('roe', 0))
                    except Exception:
                        st.session_state['est_roe_slider'] = kiwoom_data.get('roe', 0) or 0.0
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

        # 최소 조치: 현재 사용되는 지표/멀티플 진단 패널
        with st.expander("진단: 가격·BPS·PBR 일관성", expanded=False):
            k = st.session_state.kiwoom_data
            price = float(k.get('price', 0) or 0)
            bps = float(k.get('bps', 0) or 0)
            pbr = float(k.get('pbr', 0) or 0)

            market_pbr_from_price = (price / bps) if bps else 0.0
            price_from_bps_pbr = (bps * pbr) if bps else 0.0

            # 현재 입력(가정) 기반 목표 PBR/목표주가
            est_roe = float(st.session_state.get('est_roe_slider', k.get('roe', 0) or 0.0))
            cost_of_equity = float(st.session_state.get('cost_of_equity_slider', 9.0))
            terminal_growth = float(st.session_state.get('terminal_growth_slider', 3.0))
            est_bps = float(st.session_state.get('est_bps_input', k.get('bps', 0) or 0.0))
            target_pbr = ValuationCalculator.calculate_target_pbr(est_roe, cost_of_equity, terminal_growth)
            target_price = ValuationCalculator.calculate_target_price(target_pbr, est_bps)

            st.markdown(
                f"- 사용 중 주가(price): `{price:,.0f}` 원\n"
                f"- 사용 중 BPS: `{bps:,.0f}` 원\n"
                f"- 사용 중 PBR(응답): `{pbr:.2f}` 배\n"
                f"- 계산된 PBR(= price/BPS): `{market_pbr_from_price:.2f}` 배\n"
                f"- BPS×PBR로 재계산한 가격: `{price_from_bps_pbr:,.0f}` 원\n"
                f"- 목표 PBR(가정 기반): `{target_pbr:.2f}` 배\n"
                f"- 목표주가(= 목표PBR×BPS 가정): `{target_price:,.0f}` 원"
            )

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

    # --- Batch Generation UI ---
    st.header("7. 차트 일괄 생성 (Batch)")
    with st.container(border=True):
        meta_cols = st.columns([1,1,1,1])
        if meta_cols[0].button('시장/시총 메타데이터 수집', help='전체 종목에 대해 시장 구분/시가총액을 수집합니다.'):
            with st.spinner('시장/시총 메타데이터 수집 중...'):
                meta_df = collect_stock_metadata(delay_sec=0.5)
                if not meta_df.empty:
                    st.success(f"수집 완료: {len(meta_df)}개")
                    st.dataframe(meta_df.head(10), use_container_width=True)

        # Load metadata from cache if exists, else from listing
        meta_path = os.path.join(ROOT_DIR, 'cache', 'stock_market_caps.csv')
        if os.path.exists(meta_path):
            meta_df = pd.read_csv(meta_path, dtype={'code': str, 'name': str})
        else:
            meta_df = load_stock_listing()

        market_opt = st.selectbox('시장 선택', options=['전체', '코스피', '코스닥'], index=0)
        start_rank = st.number_input('시가총액 시작 순위', min_value=1, max_value=10000, value=1, step=1)
        end_rank = st.number_input('시가총액 끝 순위', min_value=1, max_value=10000, value=100, step=1)
        delay_call = st.slider('API 호출 간 딜레이(초)', min_value=0.1, max_value=2.0, value=0.5, step=0.1)
        save_size = st.select_slider('저장 이미지 크기(px, 정사각형 한 변)', options=[1000, 1200, 1400, 1600, 1800], value=1400)

        if st.button('차트 일괄 생성 시작', use_container_width=True):
            if meta_df.empty or 'market_cap' not in meta_df.columns:
                st.error('시장/시가총액 메타데이터가 없습니다. 먼저 "시장/시총 메타데이터 수집"을 실행하세요.')
            else:
                df = meta_df.copy()
                if market_opt != '전체' and 'market' in df.columns:
                    df = df[df['market'] == market_opt]
                if 'market_cap' in df.columns:
                    df = df.sort_values('market_cap', ascending=False)
                s = int(start_rank); e = int(end_rank)
                if e < s:
                    s, e = e, s
                s = max(1, min(s, len(df)))
                e = max(1, min(e, len(df)))
                top_n = e - s + 1
                df = df.iloc[s-1:e]
                st.info(f"대상 종목: {len(df)}개 (시장: {market_opt}, 상위 {int(top_n)})")

                prog = st.progress(0.0, text='일괄 생성 중...')
                logs = []
                total = len(df)
                for i, row in enumerate(df.itertuples(index=False)):
                    name, code = row.name, row.code
                    try:
                        ok, path_or_err = generate_and_save_combined_chart_headless(code, name, save_size=save_size)
                        if ok:
                            logs.append(f"[{i+1}/{total}] {name}({code}) -> 저장: {path_or_err}")
                        else:
                            logs.append(f"[{i+1}/{total}] {name}({code}) -> 실패: {path_or_err}")
                    except Exception as e:
                        logs.append(f"[{i+1}/{total}] {name}({code}) -> 예외 발생: {e}")
                    prog.progress((i+1)/total, text=f'일괄 생성 중... {i+1}/{total}')
                    time.sleep(float(delay_call))

                st.success('일괄 생성 완료')
                st.text_area('로그', value='\n'.join(logs), height=200)

    # --- Chart Image Clustering (reports/charts) ---
    st.header("9. 차트 이미지 군집 (Reports/Charts)")
    with st.container(border=True):
        cols = st.columns([1,1,1,1])
        k = int(cols[0].number_input('클러스터 수 (K)', min_value=2, max_value=12, value=4, step=1))
        window_days_cluster = int(cols[1].selectbox('특징 기간(일)', options=[60, 120, 180], index=1))
        max_images = int(cols[2].number_input('클러스터당 최대 이미지', min_value=2, max_value=30, value=8, step=1))
        img_width = int(cols[3].number_input('이미지 폭(px)', min_value=150, max_value=500, value=240, step=10))

        charts_dir = os.path.join(ROOT_DIR, 'reports', 'charts')

        def _scan_chart_images(dir_path: str):
            items = []
            try:
                if not os.path.exists(dir_path):
                    return []
                for fn in os.listdir(dir_path):
                    if not fn.lower().endswith(('.jpg', '.jpeg', '.png')):
                        continue
                    base, _ext = os.path.splitext(fn)
                    # Expect patterns:
                    #   <safe_name>_<YYYYmmdd>_<HHMMSS>
                    #   or <safe_name>_<YYYYmmdd>
                    m = re.match(r'^(.+)_([0-9]{8})(?:_([0-9]{6}))?$', base)
                    if not m:
                        # If unmatched, keep a best-effort split by first part as name
                        if '_' in base:
                            name_key = base.split('_')[0]
                            ts_part = base[len(name_key)+1:]
                            items.append({'name_key': name_key, 'ts': ts_part, 'path': os.path.join(dir_path, fn)})
                        continue
                    name_key = m.group(1)
                    ts_date = m.group(2)
                    ts_time = m.group(3) or ''
                    ts_part = f"{ts_date}_{ts_time}" if ts_time else ts_date
                    items.append({'name_key': name_key, 'ts': ts_part, 'path': os.path.join(dir_path, fn)})
            except Exception:
                return []
            return items

        def _namekey_to_code_map():
            df = get_stock_list()
            mapping = {}
            # Primary: stock_list.csv
            try:
                if isinstance(df, pd.DataFrame) and not df.empty:
                    for r in df.itertuples(index=False):
                        nm = getattr(r, 'name', None)
                        cd = getattr(r, 'code', None)
                        if nm is None or cd is None:
                            continue
                        mapping[_safe_filename(str(nm))] = str(cd)
            except Exception:
                pass
            # Augment: @data/instruments metadata (name->code)
            try:
                inst_dir = _instruments_dir()
                if os.path.exists(inst_dir):
                    for fn in os.listdir(inst_dir):
                        if not fn.lower().endswith('.json'):
                            continue
                        obj = _load_json(os.path.join(inst_dir, fn)) or {}
                        nm = obj.get('name')
                        cd = obj.get('code')
                        if nm and cd:
                            mapping[_safe_filename(str(nm))] = str(cd)
            except Exception:
                pass
            return mapping

        def _build_feature_matrix(entries, name_to_code, win_days: int):
            rows = []
            codes = []
            kept = []
            feats_all = {}
            feat_keys = None
            for e in entries:
                code = name_to_code.get(e['name_key'])
                if not code:
                    continue
                f = _compute_features_from_store(code, window_days=win_days)
                if not f:
                    continue
                feats_all[code] = f
                if feat_keys is None:
                    feat_keys = sorted(f.keys())
                vec = [float(f.get(k, 0.0)) for k in feat_keys]
                rows.append(vec)
                codes.append(code)
                kept.append(e)
            if not rows:
                return None, None, None, None
            X = np.array(rows, dtype=float)
            # Standardize features column-wise
            mu = np.nanmean(X, axis=0)
            sd = np.nanstd(X, axis=0)
            sd[sd == 0] = 1.0
            Z = (X - mu) / sd
            return Z, codes, kept, feat_keys

        def _kmeans(Z: np.ndarray, k: int, max_iter: int = 50, seed: int = 0):
            rs = np.random.RandomState(int(seed))
            n = Z.shape[0]
            if n < k:
                k = n
            idx = rs.choice(n, size=k, replace=False)
            C = Z[idx, :].copy()
            labels = np.zeros(n, dtype=int)
            for _ in range(max_iter):
                # Assign
                dists = ((Z[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
                new_labels = np.argmin(dists, axis=1)
                if np.array_equal(new_labels, labels):
                    break
                labels = new_labels
                # Update
                for j in range(k):
                    mask = labels == j
                    if not np.any(mask):
                        # reinitialize empty centroid
                        C[j] = Z[rs.choice(n)]
                    else:
                        C[j] = Z[mask].mean(axis=0)
            return labels, C

        # --- Trend + Foreign features ---
        def _linreg_slope(y: np.ndarray) -> (float, float):
            try:
                n = len(y)
                if n < 3:
                    return float('nan'), 0.0
                x = np.arange(n, dtype=float)
                x = (x - x.mean()) / (x.std() if x.std() else 1.0)
                y = (y - y.mean()) / (y.std() if y.std() else 1.0)
                A = np.vstack([x, np.ones(n)]).T
                # least squares
                beta, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                y_hat = A @ beta
                ss_res = float(np.sum((y - y_hat) ** 2))
                ss_tot = float(np.sum((y - np.mean(y)) ** 2))
                r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                return float(beta[0]), float(max(0.0, min(1.0, r2)))
            except Exception:
                return float('nan'), 0.0

        def _compute_trend_foreign_features(code: str, price_win: int, flow_win: int) -> Dict[str, float]:
            now = pd.Timestamp.now()
            start = now - pd.Timedelta(days=int(max(price_win, flow_win) * 3))
            df = load_daily_candles_from_store(code, start, now)
            if df is None or df.empty or 'close' not in df.columns:
                return {}
            df = df.sort_index()
            c = df['close'].dropna()
            if c.empty:
                return {}
            # Price features
            c_tail = c.tail(int(price_win))
            ret_price = float((c_tail.iloc[-1] / c_tail.iloc[0]) - 1.0) if len(c_tail) >= 2 else float('nan')
            slope_price, r2_price = _linreg_slope(np.log(c_tail.values + 1e-9)) if len(c_tail) >= 3 else (float('nan'), 0.0)
            ma20_gap = float(((c.tail(1).iloc[0]) - c.rolling(20).mean().tail(1).iloc[0]) / c.rolling(20).mean().tail(1).iloc[0]) if len(c) >= 25 else float('nan')

            # Foreign flow features
            df_inv = load_investor_flows_from_store(code, start, now)
            inv_for = df_inv['for_netprps'].dropna() if isinstance(df_inv, pd.DataFrame) and 'for_netprps' in df_inv.columns else pd.Series(dtype=float)
            inv_tail = inv_for.tail(int(flow_win)) if not inv_for.empty else pd.Series(dtype=float)
            for_sum = float(inv_tail.sum()) if not inv_tail.empty else float('nan')
            for_slope, _ = _linreg_slope(inv_tail.values) if len(inv_tail) >= 3 else (float('nan'), 0.0)
            # recently turned positive: last 5-day sum > 0 and previous window sum <= 0
            last5 = float(inv_for.tail(5).sum()) if not inv_for.empty else float('nan')
            prev_window = float(inv_for.tail(int(flow_win)+5).head(max(0, int(flow_win))).sum()) if len(inv_for) >= (flow_win + 5) else float('nan')
            for_recent_turn = 1.0 if (last5 == last5 and last5 > 0 and prev_window == prev_window and prev_window <= 0) else 0.0
            # positive days count and recent positive streak length
            pos_days = int((inv_tail > 0).sum()) if len(inv_tail) else 0
            streak = 0
            for v in reversed(inv_tail.values.tolist() if len(inv_tail) else []):
                if v > 0:
                    streak += 1
                else:
                    break
            return {
                'ret_price': ret_price,
                'slope_price': slope_price,
                'r2_price': r2_price,
                'ma20_gap': ma20_gap,
                'for_sum': for_sum,
                'for_slope': for_slope,
                'for_recent_turn': float(for_recent_turn),
                'for_pos_days': float(pos_days),
                'for_streak_up': float(streak),
            }

        # Mode controls
        mode = st.selectbox('클러스터 기준', options=['최근추세+외국인', '기본(종합)'], index=0)
        if mode == '최근추세+외국인':
            c1, c2, c3 = st.columns([1,1,2])
            price_win = int(c1.selectbox('가격창(일)', options=[20, 30, 60], index=1))
            flow_win = int(c2.selectbox('외국인창(일)', options=[10, 20, 30], index=1))
            w_price = float(c3.slider('가중치: 가격추세', min_value=0.0, max_value=1.0, value=0.7, step=0.05))
            w_flow = 1.0 - w_price
        else:
            price_win = window_days_cluster
            flow_win = 20
            w_price = 0.5
            w_flow = 0.5

        if st.button('이미지 군집 실행', use_container_width=True):
            with st.spinner('차트 이미지 스캔 및 군집화...'):
                entries = _scan_chart_images(charts_dir)
                # Keep most recent per name_key
                latest = {}
                for e in entries:
                    prev = latest.get(e['name_key'])
                    if prev is None or str(e['ts']) > str(prev['ts']):
                        latest[e['name_key']] = e
                entries = list(latest.values())
                name_to_code = _namekey_to_code_map()
                # Diagnostics: mapping coverage
                total_imgs = len(entries)
                mapped = [e for e in entries if e['name_key'] in name_to_code]
                unmapped = [e for e in entries if e['name_key'] not in name_to_code]
                if total_imgs == 0:
                    st.warning('reports/charts 폴더에 이미지가 없습니다. 먼저 차트를 저장하세요.')
                else:
                    st.info(f"스캔된 최신 이미지: {total_imgs}개 | 코드 매핑됨: {len(mapped)}개 | 미매핑: {len(unmapped)}개")
                    if unmapped:
                        sample = ', '.join(sorted({e['name_key'] for e in unmapped})[:10])
                        st.caption(f"미매핑 예시(최대 10개): {sample}")
                # Feature matrix
                if mode == '최근추세+외국인':
                    # Build from trend+foreign features
                    rows = []
                    kept_entries = []
                    feat_keys = ['ret_price','slope_price','r2_price','ma20_gap','for_sum','for_slope','for_recent_turn','for_pos_days','for_streak_up']
                    for e in entries:
                        code = name_to_code.get(e['name_key'])
                        if not code:
                            continue
                        f = _compute_trend_foreign_features(code, price_win=price_win, flow_win=flow_win)
                        if not f:
                            continue
                        vec = [float(f.get(k, 0.0)) for k in feat_keys]
                        rows.append(vec)
                        kept_entries.append(e)
                    if not rows:
                        Z = None
                        codes = []
                    else:
                        X = np.array(rows, dtype=float)
                        mu = np.nanmean(X, axis=0)
                        sd = np.nanstd(X, axis=0)
                        sd[sd == 0] = 1.0
                        Z = (X - mu) / sd
                        # Apply group weights
                        price_idx = [0,1,2,3]
                        flow_idx = [4,5,6,7,8]
                        Z[:, price_idx] *= w_price
                        Z[:, flow_idx] *= w_flow
                        codes = [name_to_code.get(e['name_key']) for e in kept_entries]
                else:
                    Z, codes, kept_entries, feat_keys = _build_feature_matrix(entries, name_to_code, window_days_cluster)
                if Z is None or Z.shape[0] == 0:
                    st.warning('특징 벡터를 만들 수 있는 이미지가 없습니다. 저장된 데이터/코드를 확인하세요.')
                    st.caption(f"매핑된 코드 수: {len(mapped)} | 특징 벡터 생성 성공: 0")
                else:
                    labels, C = _kmeans(Z, k=k, max_iter=50, seed=42)
                    # Group by cluster
                    clusters = {}
                    for lab, e in zip(labels, kept_entries):
                        clusters.setdefault(int(lab), []).append(e)
                    st.write(f"군집 수: {len(clusters)} | 사용된 이미지: {sum(len(v) for v in clusters.values())}")
                    # Render clusters
                    for lab in sorted(clusters.keys()):
                        st.subheader(f"Cluster {lab+1}")
                        row = clusters[lab][:max_images]
                        cols_row = st.columns(min(len(row), 5))
                        for i, e in enumerate(row):
                            with cols_row[i % len(cols_row)]:
                                st.image(e['path'], caption=e['name_key'], width=img_width)

    # --- Multiview Similarity & Clustering (Batch) ---
    st.header("10. 유사도 기반 차트 군집 생성 (Multiview)")
    with st.container(border=True):
        # Controls
        c1, c2, c3, c4 = st.columns([1,1,1,1])
        nd = int(c1.selectbox('Daily window', options=[40, 60, 90], index=1))
        nw = int(c2.selectbox('Weekly window', options=[26, 52, 78], index=1))
        nm = int(c3.selectbox('Monthly window', options=[24, 36, 48], index=1))
        inv20 = 20
        inv60 = 60

        w_cols = st.columns([1,1,1,1])
        wD = float(w_cols[0].slider('Weight: Daily', min_value=0.0, max_value=1.0, value=0.35, step=0.05))
        wW = float(w_cols[1].slider('Weight: Weekly', min_value=0.0, max_value=1.0, value=0.25, step=0.05))
        wM = float(w_cols[2].slider('Weight: Monthly', min_value=0.0, max_value=1.0, value=0.25, step=0.05))
        wI = float(w_cols[3].slider('Weight: Investor', min_value=0.0, max_value=1.0, value=0.15, step=0.05))

        # Universe filter
        u_cols = st.columns([1,1,1,1])
        market_opt = '전체'
        topN = int(u_cols[1].number_input('Top-N by MarketCap', min_value=20, max_value=500, value=150, step=10))
        try:
            meta_path = os.path.join(ROOT_DIR, 'cache', 'stock_market_caps.csv')
            markets = ['전체']
            if os.path.exists(meta_path):
                meta_df = pd.read_csv(meta_path, dtype={'code': str, 'name': str})
                if 'market' in meta_df.columns:
                    markets = ['전체'] + sorted([m for m in meta_df['market'].dropna().unique().tolist()])
            market_opt = u_cols[0].selectbox('시장(Market)', options=markets, index=0)
        except Exception:
            market_opt = '전체'
            meta_df = None

        method = u_cols[2].selectbox('Clustering', options=['agglomerative', 'dbscan'], index=0)
        eps = float(u_cols[3].number_input('DBSCAN eps', min_value=0.1, max_value=1.0, value=0.35, step=0.05)) if method == 'dbscan' else 0.35

        def _build_universe() -> list:
            codes = []
            try:
                df_l = get_stock_list()
                if isinstance(df_l, pd.DataFrame) and not df_l.empty:
                    df_l = df_l.copy()
                    df_l['code'] = df_l['code'].astype(str)
                    if market_opt != '전체' and 'market' in df_l.columns:
                        df_l = df_l[df_l['market'] == market_opt]
                    # Join market cap if available
                    try:
                        if 'meta_df' in locals() and isinstance(meta_df, pd.DataFrame) and not meta_df.empty:
                            meta_df_ = meta_df[['code', 'market_cap']].copy() if 'market_cap' in meta_df.columns else None
                            if meta_df_ is not None:
                                df_l = df_l.merge(meta_df_, on='code', how='left')
                                df_l['market_cap'] = pd.to_numeric(df_l['market_cap'], errors='coerce')
                                df_l = df_l.sort_values(by='market_cap', ascending=False)
                        else:
                            df_l = df_l.sort_values(by='code')
                    except Exception:
                        pass
                    codes = df_l['code'].dropna().astype(str).head(topN).tolist()
            except Exception:
                codes = []
            return codes

        def _scan_chart_images(dir_path: str):
            items = []
            try:
                if not os.path.exists(dir_path):
                    return []
                for fn in os.listdir(dir_path):
                    if not fn.lower().endswith(('.jpg', '.jpeg', '.png')):
                        continue
                    base, _ext = os.path.splitext(fn)
                    m = re.match(r'^(.+)_([0-9]{8})(?:_([0-9]{6}))?$', base)
                    if not m:
                        if '_' in base:
                            name_key = base.split('_')[0]
                            ts_part = base[len(name_key)+1:]
                            items.append({'name_key': name_key, 'ts': ts_part, 'path': os.path.join(dir_path, fn)})
                        continue
                    name_key = m.group(1)
                    ts_date = m.group(2)
                    ts_time = m.group(3) or ''
                    ts_part = f"{ts_date}_{ts_time}" if ts_time else ts_date
                    items.append({'name_key': name_key, 'ts': ts_part, 'path': os.path.join(dir_path, fn)})
            except Exception:
                return []
            return items

        def _namekey_to_code_map():
            df = get_stock_list()
            mapping = {}
            try:
                if isinstance(df, pd.DataFrame) and not df.empty:
                    for r in df.itertuples(index=False):
                        nm = getattr(r, 'name', None)
                        cd = getattr(r, 'code', None)
                        if nm is None or cd is None:
                            continue
                        mapping[_safe_filename(str(nm))] = str(cd)
            except Exception:
                pass
            try:
                inst_dir = _instruments_dir()
                if os.path.exists(inst_dir):
                    for fn in os.listdir(inst_dir):
                        if not fn.lower().endswith('.json'):
                            continue
                        obj = _load_json(os.path.join(inst_dir, fn)) or {}
                        nm = obj.get('name')
                        cd = obj.get('code')
                        if nm and cd:
                            mapping[_safe_filename(str(nm))] = str(cd)
            except Exception:
                pass
            return mapping

        c5, c6, c7 = st.columns([1,1,1])
        if c5.button('Build Index', use_container_width=True):
            try:
                from apps import similarity_multiview as smv
            except Exception:
                st.error('similarity_multiview 모듈을 찾을 수 없습니다.')
                return
            universe = _build_universe()
            if len(universe) < 10:
                st.warning('Universe is too small. Adjust filters to include >= 50 codes for better results.')
            params = {
                'windows': {'daily': nd, 'weekly': nw, 'monthly': nm, 'investor': (inv20, inv60)},
                'load_daily': load_daily_candles_from_store,
                'load_investor': load_investor_flows_from_store,
                'derive_weekly_monthly': derive_weekly_monthly_from_daily,
                'ensure_monthly_frequency': ensure_monthly_frequency,
            }
            st.info(f"Building multiview index for {len(universe)} codes...")
            # Chunk progress
            prog = st.progress(0.0, text='Indexing...')
            # Build in batches to keep UI responsive
            vectors = {}
            stats_accum = None
            batch = 50
            for i in range(0, len(universe), batch):
                sub = universe[i:i+batch]
                idx_sub = smv.build_multiview_index(sub, params)
                vectors.update(idx_sub.get('vectors', {}))
                stats_accum = idx_sub.get('stats', stats_accum)
                prog.progress(min(1.0, (i + len(sub)) / max(1, len(universe))), text=f'Indexing... {i+len(sub)}/{len(universe)}')
                time.sleep(0.01)
            index = {
                'vectors': vectors,
                'stats': stats_accum or {},
                'meta': {'schema_version': '2.0.0', 'windows': {'daily': nd, 'weekly': nw, 'monthly': nm, 'investor': [inv20, inv60]}, 'updated_at': pd.Timestamp.now().isoformat()},
            }
            st.session_state['mv_index'] = index
            st.session_state['mv_codes'] = sorted(list(vectors.keys()))
            # Persist to @data/similarity
            try:
                out_path = smv.save_multiview_index(ROOT_DIR, index)
                st.success(f"Index saved: {out_path}")
            except Exception as e:
                st.warning(f"Index saved in memory only. Save failed: {e}")

        if c6.button('Build Clusters', use_container_width=True):
            try:
                from apps import similarity_multiview as smv
            except Exception:
                st.error('similarity_multiview 모듈 import 실패')
                return
            index = st.session_state.get('mv_index')
            if not index:
                st.error('Index not found. Build Index first.')
            else:
                weights = {'daily': wD, 'weekly': wW, 'monthly': wM, 'investor': wI}
                params = {'alpha': 0.7, 'beta_bonus': 0.05, 'penalty_disagree': 0.03}
                with st.spinner('Computing distance matrix...'):
                    D, codes = smv.build_distance_matrix(index, weights, params)
                with st.spinner('Clustering...'):
                    if method == 'agglomerative':
                        res = smv.cluster_from_distance_matrix(D, codes, method='agglomerative', params={'K_grid': [6,8,10,12,14]})
                    else:
                        res = smv.cluster_from_distance_matrix(D, codes, method='dbscan', params={'eps': eps, 'min_samples': 4})
                st.session_state['mv_D'] = D
                st.session_state['mv_clusters'] = res.get('clusters', {})
                st.session_state['mv_labels'] = res.get('labels', {})
                st.session_state['mv_medoids'] = res.get('medoids', {})
                K = res.get('K')
                sil = res.get('silhouette')
                st.info(f"Clusters built. method={method}, K={K}, silhouette={sil}")

        if c7.button('Materialize Folders', use_container_width=True):
            clusters = st.session_state.get('mv_clusters') or {}
            labels = st.session_state.get('mv_labels') or {}
            if not clusters or not labels:
                st.error('No clusters found. Build Clusters first.')
            else:
                # Build folder structure and copy the latest chart image per code into cluster folders
                charts_dir = os.path.join(ROOT_DIR, 'reports', 'charts')
                out_date = pd.Timestamp.now().strftime('%Y%m%d')
                out_root = os.path.join(ROOT_DIR, 'reports', 'charts_clusters', out_date)
                try:
                    os.makedirs(out_root, exist_ok=True)
                except Exception:
                    pass
                items = _scan_chart_images(charts_dir)
                name_to_code = _namekey_to_code_map()
                # Map code -> latest image path
                code2img = {}
                # build code occurrences
                for e in items:
                    code = name_to_code.get(e['name_key'])
                    if not code:
                        continue
                    prev = code2img.get(code)
                    if prev is None or (str(e['ts']) > str(prev.get('ts'))):
                        code2img[code] = e
                # Copy
                copied = 0
                for lab, codes_in in clusters.items():
                    cdir = os.path.join(out_root, str(lab))
                    try:
                        os.makedirs(cdir, exist_ok=True)
                    except Exception:
                        pass
                    for code in codes_in:
                        e = code2img.get(code)
                        if not e:
                            continue
                        fn = os.path.basename(e['path'])
                        dst = os.path.join(cdir, fn)
                        try:
                            # copy file
                            with open(e['path'], 'rb') as sf, open(dst, 'wb') as df:
                                df.write(sf.read())
                            copied += 1
                        except Exception:
                            continue
                # Save manifest
                try:
                    from apps import similarity_multiview as smv
                    manifest_dir = os.path.join(ROOT_DIR, '@data', 'clusters')
                    os.makedirs(manifest_dir, exist_ok=True)
                    manifest_path = os.path.join(manifest_dir, f'charts-{out_date}.json')
                    params = {
                        'windows': {'daily': nd, 'weekly': nw, 'monthly': nm, 'investor': [inv20, inv60]},
                        'weights': {'daily': wD, 'weekly': wW, 'monthly': wM, 'investor': wI},
                        'method': method,
                        'universe_filter': {'market': market_opt, 'topN': topN}
                    }
                    meta = {'params': params, 'reports_root': f'reports/charts_clusters/{out_date}/', 'images': {k: (code2img[k]['path'] if k in code2img else None) for k in st.session_state.get('mv_codes', [])}}
                    smv.save_clusters_manifest(manifest_path, clusters, labels, st.session_state.get('mv_medoids', {}), meta)
                    st.success(f"Materialized {copied} images to {out_root}. Manifest: {manifest_path}")
                except Exception as e:
                    st.warning(f"Materialization done, but manifest save failed: {e}")

    # --- Cluster Viewer ---
    st.header("11. Cluster Viewer (Multiview)")
    with st.container(border=True):
        def _clusters_dir():
            return os.path.join(ROOT_DIR, '@data', 'clusters')

        def _list_cluster_manifests():
            try:
                d = _clusters_dir()
                if not os.path.exists(d):
                    return []
                files = [fn for fn in os.listdir(d) if fn.startswith('charts-') and fn.endswith('.json')]
                # sort by date descending if possible
                def _key(fn):
                    m = re.match(r'^charts-([0-9]{8})\.json$', fn)
                    return m.group(1) if m else '00000000'
                files = sorted(files, key=_key, reverse=True)
                return files
            except Exception:
                return []

        def _load_manifest(path: str):
            cache = st.session_state.setdefault('cluster_viewer_cache', {})
            key = ('manifest', path)
            if key in cache:
                return cache[key]
            obj = _load_json(path) or {}
            cache[key] = obj
            return obj

        def _load_mv_index_for_windows(windows: dict):
            # Try session first
            idx = st.session_state.get('mv_index')
            meta_windows = idx.get('meta', {}).get('windows') if idx else None
            if meta_windows and all(str(meta_windows.get(k)) == str(windows.get(k)) for k in ['daily', 'weekly', 'monthly']):
                return idx
            # Else load from disk per schema path
            try:
                nd = int(windows.get('daily', 60))
                nw = int(windows.get('weekly', 52))
                nm = int(windows.get('monthly', 36))
            except Exception:
                nd, nw, nm = 60, 52, 36
            idx_path = os.path.join(ROOT_DIR, '@data', 'similarity', f'multiview-index-d{nd}-w{nw}-m{nm}.json')
            cache = st.session_state.setdefault('cluster_viewer_cache', {})
            key = ('index', os.path.basename(idx_path))
            if key in cache:
                return cache[key]
            try:
                with open(idx_path, 'r', encoding='utf-8') as f:
                    obj = json.load(f)
                cache[key] = obj
                return obj
            except Exception:
                return None

        def _code_to_name_map():
            try:
                df = get_stock_list()
                if isinstance(df, pd.DataFrame) and not df.empty:
                    return {str(r.code): str(r.name) for r in df.itertuples(index=False)}
            except Exception:
                pass
            return {}

        def _resolve_image_path(date_str: str, cluster_id: str, code: str, images_map: dict, name_map: dict):
            # 1) Prefer clustered folder image
            cluster_dir = os.path.join(ROOT_DIR, 'reports', 'charts_clusters', date_str, str(cluster_id))
            if os.path.exists(cluster_dir):
                try:
                    nm = name_map.get(str(code), '')
                    key = _safe_filename(str(nm)) if nm else None
                    candidates = []
                    for fn in os.listdir(cluster_dir):
                        if not fn.lower().endswith(('.jpg', '.jpeg', '.png')):
                            continue
                        if key and fn.startswith(key + '_'):
                            candidates.append(os.path.join(cluster_dir, fn))
                    # latest
                    if candidates:
                        candidates = sorted(candidates)
                        return candidates[-1]
                except Exception:
                    pass
            # 2) images map
            p = images_map.get(str(code)) if isinstance(images_map, dict) else None
            if p:
                # normalize to absolute
                if not os.path.isabs(p):
                    p = os.path.join(ROOT_DIR, p)
                if os.path.exists(p):
                    return p
            # 3) fallback scan reports/charts by company name
            try:
                nm = name_map.get(str(code), '')
                key = _safe_filename(str(nm)) if nm else None
                charts_dir = os.path.join(ROOT_DIR, 'reports', 'charts')
                if key and os.path.exists(charts_dir):
                    cand = [os.path.join(charts_dir, fn) for fn in os.listdir(charts_dir) if fn.startswith(key + '_') and fn.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    if cand:
                        return sorted(cand)[-1]
            except Exception:
                pass
            return None

        manifests = _list_cluster_manifests()
        cols_top = st.columns([2, 1, 1])
        if not manifests:
            cols_top[0].info('No cluster manifests found. Build and materialize clusters first.')
        else:
            chosen = cols_top[0].selectbox('Select clusters manifest (@data/clusters)', options=manifests, index=0)
            manifest_path = os.path.join(_clusters_dir(), chosen)
            obj = _load_manifest(manifest_path)
            params = obj.get('params', {})
            windows = params.get('windows', {'daily': 60, 'weekly': 52, 'monthly': 36, 'investor': [20, 60]})
            weights = params.get('weights', {'daily': 0.35, 'weekly': 0.25, 'monthly': 0.25, 'investor': 0.15})
            reports_root = obj.get('reports_root', '')
            date_str = re.search(r'charts-([0-9]{8})\.json$', chosen).group(1) if re.search(r'charts-([0-9]{8})\.json$', chosen) else ''
            clusters = {int(k): [str(x) for x in v] for k, v in (obj.get('clusters') or {}).items()}
            labels = {str(k): int(v) for k, v in (obj.get('labels') or {}).items()}
            medoids = {int(k): str(v) for k, v in (obj.get('medoids') or {}).items()}
            images_map = obj.get('images', {})

            # Summary
            sizes = {k: len(v) for k, v in clusters.items()}
            cols_top[1].metric('Clusters', len(clusters))
            cols_top[2].metric('Total items', sum(sizes.values()) if sizes else 0)
            st.caption(f"Medoids: {', '.join([f'{k}->{v}' for k,v in medoids.items()])}")

            # Controls
            c1, c2, c3, c4, c5 = st.columns([1,1,1,1,1])
            cluster_ids = sorted(list(clusters.keys()))
            if not cluster_ids:
                st.warning('No clusters in manifest.')
                st.stop()
            cid = int(c1.selectbox('Cluster ID', options=cluster_ids, index=0))
            sort_opt = c2.selectbox('Sort by', options=['code', 'distance-to-medoid'], index=1 if cid in medoids else 0)
            topK = int(c3.slider('Top-K around medoid', min_value=10, max_value=60, value=30, step=5))
            ncols = int(c4.slider('Grid columns', min_value=2, max_value=5, value=4))
            img_w = int(c5.slider('Image width(px)', min_value=160, max_value=400, value=220, step=10))

            name_map = _code_to_name_map()
            codes = clusters.get(cid, [])
            med_code = medoids.get(cid) if cid in medoids else (codes[0] if codes else None)
            order = codes[:]
            distances = None
            if sort_opt == 'distance-to-medoid' and med_code and codes:
                try:
                    from apps import similarity_multiview as smv
                except Exception:
                    smv = None
                if smv is None:
                    st.warning('similarity_multiview not available; fallback to code sort.')
                else:
                    idx = _load_mv_index_for_windows(windows)
                    if not idx:
                        st.warning('Multiview index not found for selected windows; fallback to code sort.')
                    else:
                        with st.spinner('Computing distances to medoid...'):
                            params_dist = {'alpha': 0.7, 'beta_bonus': 0.05, 'penalty_disagree': 0.03}
                            dists = []
                            for c in codes:
                                try:
                                    res = smv.compute_distance(med_code, c, idx, weights, params_dist)
                                    d = float(res.get('total', float('nan')))
                                    if not (d == d):
                                        d = 1.0
                                    dists.append((c, d))
                                except Exception:
                                    dists.append((c, 1.0))
                            # ensure medoid first
                            dists = sorted([x for x in dists if x[0] != med_code], key=lambda t: t[1])
                            if topK and len(dists) > (topK - 1):
                                dists = dists[:(topK - 1)]
                            order = [med_code] + [c for c, _ in dists]
                            distances = {c: d for c, d in dists}

            # Missing images regeneration
            regen = st.checkbox('Regenerate missing images if needed')
            if regen and st.button('Generate Missing', use_container_width=True):
                if 'kiwoom_handler' not in st.session_state:
                    st.warning('Kiwoom handler is not initialized; cannot regenerate.')
                else:
                    with st.spinner('Generating images...'):
                        prog = st.progress(0.0, text='Generating...')
                        total = len(order)
                        errors = 0
                        for i, code in enumerate(order):
                            p = _resolve_image_path(date_str, cid, code, images_map, name_map)
                            if not p or not os.path.exists(p):
                                nm = name_map.get(str(code), str(code))
                                ok, msg = generate_and_save_combined_chart_headless(code, nm)
                                if not ok:
                                    errors += 1
                            prog.progress((i + 1) / max(1, total), text=f'Generating... {i+1}/{total}')
                        if errors:
                            st.warning(f'Generation finished with {errors} errors.')
                        else:
                            st.success('Generation finished.')

            # Render grid
            st.subheader(f"Cluster {cid} ({len(codes)} items)")
            cols = st.columns(ncols)
            idx_col = 0
            for i, code in enumerate(order):
                p = _resolve_image_path(date_str, cid, code, images_map, name_map)
                cap = f"{name_map.get(str(code), str(code))} ({code})"
                if code == med_code:
                    cap = f"★ Medoid | {cap}"
                if distances and code in distances:
                    cap = f"{cap} | d={distances[code]:.3f}"
                with cols[idx_col % ncols]:
                    if p and os.path.exists(p):
                        st.image(p, caption=cap, width=img_w)
                        st.caption(p)
                    else:
                        st.info(f"Image not found for {cap}")
                idx_col += 1

    # Stop here; legacy similarity removed below
    return

    # --- Similarity Finder ---
    st.header("8. 유사 차트 찾기 (Similarity Search)")
    with st.container(border=True):
        sim_cols = st.columns([1,1,1,1,1])
        window = sim_cols[0].selectbox('창 길이', options=[20, 60, 120], index=1)
        feat_label = sim_cols[1].selectbox('특징', options=['모양(수익률)', '추세(가격)'], index=0)
        feature = 'returns' if '수익률' in feat_label else 'price'
        topn = int(sim_cols[2].number_input('Top N', min_value=5, max_value=50, value=20, step=1))

        # Market filter if meta available
        market_opt = '전체'
        try:
            meta_path = os.path.join(ROOT_DIR, 'cache', 'stock_market_caps.csv')
            if os.path.exists(meta_path):
                meta_df = pd.read_csv(meta_path, dtype={'code': str, 'name': str})
                if 'market' in meta_df.columns:
                    markets = ['전체'] + sorted([m for m in meta_df['market'].dropna().unique().tolist()])
                    market_opt = sim_cols[3].selectbox('시장', options=markets, index=0)
        except Exception:
            pass

        if sim_cols[4].button('유사 차트 찾기', use_container_width=True):
            if not stock_code:
                st.error('종목을 먼저 선택하세요.')
            else:
                with st.spinner('인덱스 구축/조회 및 유사도 계산 중...'):
                    try:
                        idx = build_similarity_index(window=window, feature=feature, market=market_opt if market_opt != '전체' else None)
                    except Exception:
                        idx = {}
                    res = find_similar(stock_code, topn=topn, window=window, feature=feature, market=market_opt if market_opt != '전체' else None)

                # Map code->name for titles
                name_map = {}
                try:
                    df_l = get_stock_list()
                    if isinstance(df_l, pd.DataFrame) and not df_l.empty:
                        name_map = {str(r.code): str(r.name) for r in df_l.itertuples(index=False) if getattr(r, 'code', None) is not None}
                except Exception:
                    name_map = {}
                name_map[str(stock_code)] = company_name

                if not res:
                    st.warning('유사 결과가 없습니다. 데이터 커버리지를 확인하세요.')
                else:
                    st.subheader('유사 종목 목록')
                    out_rows = []
                    for i, r in enumerate(res, start=1):
                        c = str(r['code'])
                        out_rows.append({
                            'rank': i,
                            'name': name_map.get(c, '-'),
                            'code': c,
                            'distance(낮을수록 유사)': round(float(r['distance']), 4),
                            'overlap': int(r.get('overlap', 0))
                        })
                    st.dataframe(pd.DataFrame(out_rows), use_container_width=True, hide_index=True)

                    st.subheader('오버레이 비교')
                    # small multiples grid
                    cols_per_row = 3
                    for i in range(0, len(res), cols_per_row):
                        row = res[i:i+cols_per_row]
                        cols = st.columns(cols_per_row)
                        for j, item in enumerate(row):
                            with cols[j]:
                                fig = _overlay_figure_for_pair(stock_code, item['code'], window, feature, name_map)
                                if fig is not None:
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.caption(f"{name_map.get(str(item['code']), str(item['code']))}: 데이터 부족")

if __name__ == "__main__":
    main()
