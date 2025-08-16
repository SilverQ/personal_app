"""
stock_analysis_app_v2 (리팩터링 + 진단 로그 강화)
- 메인 앱의 사이드바를 사용하지 않는 서브앱 버전
- 상단 컨트롤 패널(본문 영역)에서 1회 입력 → 단일 종합 보고서 렌더
- ReportBuilder 클래스로 수집/검증/시각화/내보내기 일원화
- DART 기반 재무/밸류에이션(옵션), PyKrx 기반 가격/수급
- PDF(ReportLab+Kaleido) 또는 HTML 자동 내보내기
- 🔧 어디서 실패했는지 알 수 있도록 단계별 진단 로그 출력

주의: set_page_config는 메인에서만 호출합니다.
"""

from __future__ import annotations

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path
import time
import pickle
import configparser
import warnings
import traceback
import math
from datetime import timezone, datetime
import pandas as pd
import re
import os
import inspect

warnings.filterwarnings("ignore")

# =============================
# 환경 설정/외부 라이브러리 체크
# =============================
IS_DEBUG = False

CONFIG = configparser.ConfigParser()
CONFIG.read("config.ini")
DART_KEY = CONFIG.get("DART", "key", fallback=None)

try:
    from pykrx import stock
    from pykrx.stock import get_market_ticker_name
    PYKRX_AVAILABLE = True
except Exception as _e:
    PYKRX_AVAILABLE = False

# OpenDartReader는 배포 버전에 따라 생성자 접근 방식이 다릅니다.
# - 어떤 환경: from OpenDartReader import OpenDartReader; OpenDartReader(api_key)
# - 다른 환경: import OpenDartReader; OpenDartReader.OpenDartReader(api_key)
try:
    import OpenDartReader as _odr_module  # 모듈로 임포트 시도
    if hasattr(_odr_module, "OpenDartReader"):
        ODR_CTOR = _odr_module.OpenDartReader
    else:
        ODR_CTOR = _odr_module  # type: ignore
    DART_AVAILABLE = True
except Exception:
    try:
        from OpenDartReader import OpenDartReader as _odr_class  # type: ignore
        ODR_CTOR = _odr_class
        DART_AVAILABLE = True
    except Exception:
        DART_AVAILABLE = False
        ODR_CTOR = None

try:
    # PDF 생성(선택): ReportLab
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas as pdf_canvas
    from reportlab.lib.units import cm
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

try:
    # Plotly 정적 이미지 저장(선택)
    import kaleido  # ← 실제 엔진
    KALEIDO_AVAILABLE = True
except Exception:
    KALEIDO_AVAILABLE = False

import plotly.io as pio

# =============================
# 캐시/헬퍼
# =============================
CACHE_DIR = Path("./cache"); CACHE_DIR.mkdir(exist_ok=True)
SIMPLE_CACHE = Path("simple_trading_cache.pkl")


def _now_kst():
    # pytz 없이도 동작하도록 간단 변환
    return datetime.utcnow() + timedelta(hours=9)


def _load_simple_cache() -> dict:
    if SIMPLE_CACHE.exists():
        try:
            return pickle.load(open(SIMPLE_CACHE, "rb"))
        except Exception:
            return {}
    return {}


def _save_simple_cache(data: dict) -> None:
    try:
        pickle.dump(data, open(SIMPLE_CACHE, "wb"))
    except Exception:
        pass


def _ticker_name(ticker: str) -> str:
    if not PYKRX_AVAILABLE:
        return ticker
    try:
        name = get_market_ticker_name(ticker)
        return name or ticker
    except Exception:
        return ticker


def _ohlcv(ticker: str, start: str, end: str, adjusted: bool=False) -> pd.DataFrame:
    if not PYKRX_AVAILABLE:
        return pd.DataFrame()
    return stock.get_market_ohlcv_by_date(start, end, ticker, adjusted=adjusted)



def _investor_daily(ticker: str, start: str, end: str, debug=None) -> pd.DataFrame:
    """영업일 기준 일자 index, 컬럼은 투자자 구분(개인/기관합계/외국인/기타법인) 순매수만 반환.
    debug: callable(event: str, message: str, **context)
    """
    if not PYKRX_AVAILABLE:
        if debug: debug("pykrx_missing", "PyKrx 미설치로 투자자 데이터 수집 불가")
        return pd.DataFrame()
    key = f"INV_{ticker}_{start}_{end}"
    cache = _load_simple_cache()
    if key in cache:
        if debug: debug("cache_hit", "투자자 데이터 캐시 적중", key=key)
        return cache[key]
    dates = pd.bdate_range(pd.to_datetime(start), pd.to_datetime(end))
    out = []
    for d in dates:
        ds = d.strftime("%Y%m%d")
        try:
            df = stock.get_market_trading_value_by_investor(ds, ds, ticker)
            if df is not None and not df.empty:
                if "순매수" in df.columns:
                    day = df["순매수"].to_frame().T
                    day.index = [pd.to_datetime(ds)]
                    out.append(day)
                else:
                    if debug: debug("no_column", "'순매수' 컬럼 없음", date=ds, cols=list(df.columns))
            else:
                if debug: debug("empty", "해당 일자 데이터 없음", date=ds)
        except Exception as ex:
            if debug: debug("error", "일자 데이터 수집 실패", date=ds, error=str(ex), tb=traceback.format_exc())
        time.sleep(0.2)
    if not out:
        if debug: debug("result_empty", "누적 결과가 비어 있습니다.")
        return pd.DataFrame()
    res = pd.concat(out).fillna(0)
    cache[key] = res
    _save_simple_cache(cache)
    if debug: debug("cache_save", "투자자 데이터 캐시에 저장", rows=len(res))
    return res


def _current_close(ticker: str) -> float | None:
    if not PYKRX_AVAILABLE:
        return None
    try:
        today = datetime.now().strftime("%Y%m%d")
        yday = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
        df = stock.get_market_ohlcv_by_date(yday, today, ticker)
        if df is not None and not df.empty:
            return float(df["종가"].iloc[-1])
    except Exception:
        return None
    return None


def _latest_prices(ticker: str, log=None):
    if not PYKRX_AVAILABLE:
        if log: log("warning", "PRICE_NOW", "PyKrx 미설치")
        return None

    now = _now_kst()
    start = (now - timedelta(days=20)).strftime("%Y%m%d")
    end   = now.strftime("%Y%m%d")

    try:
        df = stock.get_market_ohlcv_by_date(start, end, ticker)
        if df is None or df.empty:
            if log: log("warning", "PRICE_NOW", "OHLCV empty", start=start, end=end)
            return None
        df = df.sort_index()

        last_idx = df.index[-1]
        last     = df.iloc[-1]

        # KRX 일일데이터 확정 시간을 넉넉히 18:00(KST)로 설정
        cutoff = now.replace(hour=18, minute=0, second=0, microsecond=0)

        if now < cutoff and len(df) >= 2:
            settled_idx = df.index[-2]
            settled     = df.iloc[-2]
        else:
            settled_idx = last_idx
            settled     = last

        out = {
            "latest_date":  last_idx.strftime("%Y-%m-%d"),
            "latest_open":  float(last.get("시가", np.nan))  if "시가" in last else np.nan,
            "latest_high":  float(last.get("고가", np.nan))  if "고가" in last else np.nan,
            "latest_low":   float(last.get("저가", np.nan))  if "저가" in last else np.nan,
            "latest_close": float(last.get("종가", np.nan))  if "종가" in last else np.nan,
            "latest_volume":float(last.get("거래량", np.nan)) if "거래량" in last else np.nan,

            "settled_date":  settled_idx.strftime("%Y-%m-%d"),
            "settled_close": float(settled.get("종가", np.nan)) if "종가" in settled else np.nan,
        }
        if log:
            log("info", "PRICE_NOW",
                "가장 최근(잠정) & 확정 종가 산출",
                latest_date=out["latest_date"], latest_close=out["latest_close"],
                settled_date=out["settled_date"], settled_close=out["settled_close"])
        return out
    except Exception as ex:
        if log: log("error", "PRICE_NOW", "조회 예외", error=str(ex), tb=traceback.format_exc())
        return None

# =============================
# DART 수집 & 모델 (견고화)
# =============================
class DartDataCollector:
    def __init__(self, api_key: str | None, log=None):
        self.api_key = api_key
        self._log = log
        self.dart = None
        self._finstate_supports_fs_div: bool | None = None  # ← 추가: 지원여부 캐시

        if api_key and DART_AVAILABLE and ODR_CTOR is not None:
            try:
                self.dart = ODR_CTOR(api_key)
                if self._log: self._log("info", "DART_INIT", "OpenDartReader 초기화 성공")
            except Exception as ex:
                if self._log: self._log("error", "DART_INIT", "OpenDartReader 초기화 실패",
                                         error=str(ex), tb=traceback.format_exc())
        else:
            if self._log: self._log("warning", "DART_INIT", "DART 사용 불가 (키 없음/미설치)")

    # ---------------------------
    # 내부 유틸
    # ---------------------------
    def _logx(self, level, stage, msg, **ctx):
        if callable(self._log):
            self._log(level, stage, msg, **ctx)

    def _corp_codes_df(self):
        """버전별 corp_codes 접근(속성/함수 모두 처리)"""
        try:
            cc = getattr(self.dart, "corp_codes", None)
            if cc is None:
                return None
            # 속성인가?
            if isinstance(cc, (pd.DataFrame, list)):
                return pd.DataFrame(cc)
            # 함수인가?
            if callable(cc):
                df = cc()
                return pd.DataFrame(df)
        except Exception as ex:
            self._logx("warning", "DART_COMPANY", "corp_codes 접근 실패", error=str(ex))
        return None

    def _normalize_fs(self, df: pd.DataFrame, year: int, rc: str):
        if df is None or df.empty:
            return df
        out = df.copy()
        for c in ["fs_div", "sj_div", "account_id", "account_nm",
                  "thstrm_amount", "thstrm_add_amount", "frmtrm_amount", "bfefrmtrm_amount"]:
            if c in out.columns:
                out[c] = out[c].astype(str)
        out["__bsns_year"] = int(year)
        out["__reprt_code"] = str(rc)
        return out

    def _call_finstate(self, corp_code: str, y: int, rc: str, retry=2, wait=0.2):
        last_err = None
        for k in range(retry + 1):
            try:
                fs = self.dart.finstate(str(corp_code), int(y), reprt_code=str(rc), fs_div="CFS")
                if fs is not None and not getattr(fs, "empty", False):
                    return self._normalize_fs(fs, y, rc)
                else:
                    self._logx("debug", "DART_FS", "빈 재무제표", year=y, reprt_code=rc)
            except Exception as ex:
                last_err = ex
                self._logx("warning", "DART_FS", "재무제표 조회 실패(재시도 예정)",
                           year=y, reprt_code=rc, attempt=k, error=str(ex))
            time.sleep(wait * (k + 1))  # 점증 백오프
        if last_err:
            self._logx("error", "DART_FS", "재무제표 조회 최종 실패",
                       year=y, reprt_code=rc, error=str(last_err), tb=traceback.format_exc())
        return None

    # ---------------------------
    # 회사 식별
    # ---------------------------
    def company(self, ticker: str) -> dict | None:
        """DART 회사 기본 정보 조회 (강화된 폴백 순서).
        1) dart.company(티커)  → dict
        2) corp_codes DF에서 stock_code 일치 행
        3) find_corp_code(티커)
        4) 종목명 → company_by_name(종목명)
        """
        if not self.dart:
            self._logx("warning", "DART_COMPANY", "DART 핸들 없음")
            return None

        t = str(ticker)

        # 1) company(ticker)
        try:
            info = self.dart.company(t)
            if isinstance(info, dict):
                cc = info.get("corp_code") or info.get("corpcode")
                nm = info.get("corp_name") or info.get("corpname")
                if cc:
                    self._logx("info", "DART_COMPANY", "company() 성공", corp_code=cc, corp_name=nm)
                    return {"corp_code": str(cc), "corp_name": str(nm or t), "stock_code": t}
        except Exception as ex:
            self._logx("warning", "DART_COMPANY", "company() 실패", error=str(ex))

        # 2) corp_codes DF 탐색
        try:
            cdf = self._corp_codes_df()
            if isinstance(cdf, pd.DataFrame) and not cdf.empty:
                # 컬럼 이름 버전차 가드
                cols = {c.lower(): c for c in cdf.columns}
                sc = cols.get("stock_code") or cols.get("stockcode") or "stock_code"
                cc = cols.get("corp_code") or cols.get("corpcode") or "corp_code"
                cn = cols.get("corp_name") or cols.get("corpname") or "corp_name"
                if sc in cdf.columns and cc in cdf.columns:
                    row = cdf[cdf[sc].astype(str) == t]
                    if not row.empty:
                        r0 = row.iloc[0]
                        corp_code = str(r0.get(cc))
                        corp_name = str(r0.get(cn, t))
                        self._logx("info", "DART_COMPANY", "corp_codes 매칭", corp_code=corp_code, corp_name=corp_name)
                        return {"corp_code": corp_code, "corp_name": corp_name, "stock_code": t}
                else:
                    self._logx("debug", "DART_COMPANY", "corp_codes DF에 필요한 컬럼 없음",
                               columns=list(cdf.columns))
            else:
                self._logx("warning", "DART_COMPANY", "corp_codes 비어있음 또는 미지원")
        except Exception as ex:
            self._logx("warning", "DART_COMPANY", "corp_codes 탐색 실패", error=str(ex))

        # 3) find_corp_code
        try:
            cc = self.dart.find_corp_code(t)
            if cc:
                try:
                    nm = get_market_ticker_name(t)
                except Exception:
                    nm = t
                self._logx("info", "DART_COMPANY", "find_corp_code() 성공", corp_code=cc, corp_name=nm)
                return {"corp_code": str(cc), "corp_name": str(nm), "stock_code": t}
        except Exception as ex:
            self._logx("warning", "DART_COMPANY", "find_corp_code() 실패", error=str(ex))

        # 4) 종목명으로 조회
        try:
            nm = get_market_ticker_name(t)
            byname = self.dart.company_by_name(str(nm))
            if hasattr(byname, "empty") and not byname.empty:
                row = byname.iloc[0]
                corp_code = row.get("corp_code") or row.get("corpcode")
                corp_name = row.get("corp_name") or row.get("corpname") or nm
                if corp_code:
                    self._logx("info", "DART_COMPANY", "company_by_name() 성공",
                               corp_code=corp_code, corp_name=corp_name)
                    return {"corp_code": str(corp_code), "corp_name": str(corp_name), "stock_code": t}
        except Exception as ex:
            self._logx("warning", "DART_COMPANY", "company_by_name() 실패", error=str(ex))

        self._logx("warning", "DART_COMPANY", "모든 매칭 실패", ticker=t)
        return None

    # ---------------------------
    # 재무제표 수집
    # ---------------------------
    def fin_map(self, corp_code: str, years: int = 5) -> dict[int, pd.DataFrame]:
        if not self.dart:
            self._logx("warning", "DART_FS", "DART 핸들 없음")
            return {}
        out: dict[int, pd.DataFrame] = {}
        this_year = datetime.now().year
        reprt_codes = ["11011", "11012", "11013", "11014"]  # 사업→반기→1Q→3Q

        for y in range(this_year - years, this_year):
            got = False
            for rc in reprt_codes:
                fs = self._call_finstate(corp_code, y, rc, retry=2, wait=0.25)
                if fs is not None and not fs.empty:
                    out[y] = fs
                    self._logx("info", "DART_FS", "재무제표 수집", year=y, reprt_code=rc, rows=len(fs))
                    got = True
                    break
            if not got:
                self._logx("warning", "DART_FS", "해당 연도 보고서 미확보", year=y)

        if not out:
            self._logx("warning", "DART_FS", "수집된 재무제표가 없습니다.")
        else:
            y0 = sorted(out.keys())[0]
            try:
                self._logx("debug", "DART_FS_SAMPLE", "예시 로우",
                           sample=out[y0].head(3).to_dict(orient="records"))
            except Exception:
                pass
        return out

    def _detect_fs_div_support(self):
        """첫 호출 전에 한 번만 시그니처를 보고 추정. (완벽하지 않으면 런타임 예외로 재확정)"""
        try:
            sig = inspect.signature(self.dart.finstate)
            self._finstate_supports_fs_div = "fs_div" in sig.parameters
            self._logx("debug", "DART_FS", f"fs_div 지원 탐지: {self._finstate_supports_fs_div}")
        except Exception:
            # 알 수 없으면 None → 실제 호출에서 예외를 보고 결정
            self._finstate_supports_fs_div = None

    def _call_finstate(self, corp_code: str, y: int, rc: str, retry=2, wait=0.25):
        if self._finstate_supports_fs_div is None:
            self._detect_fs_div_support()

        last_err = None
        for k in range(retry + 1):
            try:
                if self._finstate_supports_fs_div:
                    fs = self.dart.finstate(str(corp_code), int(y), reprt_code=str(rc), fs_div="CFS")
                else:
                    fs = self.dart.finstate(str(corp_code), int(y), reprt_code=str(rc))
                if fs is not None and not getattr(fs, "empty", False):
                    return self._normalize_fs(fs, y, rc)
                else:
                    self._logx("debug", "DART_FS", "빈 재무제표", year=y, reprt_code=rc)
            except TypeError as ex:
                # "unexpected keyword argument 'fs_div'" → 지원 안 함으로 전환하고 즉시 재시도
                msg = str(ex)
                if "fs_div" in msg:
                    if self._finstate_supports_fs_div is not False:
                        self._logx("warning", "DART_FS",
                                   "설치된 OpenDartReader는 fs_div 미지원 → 폴백(인자 제거)로 전환")
                    self._finstate_supports_fs_div = False
                    last_err = ex
                else:
                    last_err = ex
                    self._logx("warning", "DART_FS", "TypeError", year=y, reprt_code=rc,
                               attempt=k, error=msg)
            except Exception as ex:
                last_err = ex
                self._logx("warning", "DART_FS", "재무제표 조회 실패(재시도 예정)",
                           year=y, reprt_code=rc, attempt=k, error=str(ex))
            time.sleep(wait * (k + 1))

        if last_err:
            self._logx("error", "DART_FS", "재무제표 조회 최종 실패",
                       year=y, reprt_code=rc, error=str(last_err),
                       tb=traceback.format_exc())
        return None

    def company(self, ticker: str) -> dict | None:
        # (이 부분은 기존 강화 버전 그대로 사용하셔도 됩니다)
        t = str(ticker)
        if not self.dart:
            self._logx("warning", "DART_COMPANY", "DART 핸들 없음")
            return None
        try:
            info = self.dart.company(t)
            if isinstance(info, dict):
                cc = info.get("corp_code") or info.get("corpcode")
                nm = info.get("corp_name") or info.get("corpname")
                if cc:
                    self._logx("info", "DART_COMPANY", "company() 성공", corp_code=cc, corp_name=nm)
                    return {"corp_code": str(cc), "corp_name": str(nm or t), "stock_code": t}
        except Exception as ex:
            self._logx("warning", "DART_COMPANY", "company() 실패", error=str(ex))

        # corp_codes → find_corp_code → company_by_name 순 폴백 (생략)
        # ... (사용 중인 버전 그대로 두세요)

        self._logx("warning", "DART_COMPANY", "모든 매칭 실패", ticker=t)
        return None


# --- DCFModel 보강: FCFF 구성요소 추출 ---
# ==== 교체: DCFModel 전체 ====
class DCFModel:
    def __init__(self):
        # 기본 가정(필요시 UI로 노출해 조정 가능)
        self.rfr = 0.035   # 무위험수익률
        self.mrp = 0.06    # 시장위험프리미엄
        self.crp = 0.005   # 국가위험프리미엄
        self.tax = 0.25    # 법인세율
        self.g   = 0.025   # 말기성장률

    def wacc(self, beta: float, debt_ratio: float = 0.3) -> float:
        coe = self.rfr + beta * (self.mrp + self.crp)   # 주주요구수익률
        cod = self.rfr + 0.02                           # 부채비용(단순 가정)
        return (1 - debt_ratio) * coe + debt_ratio * cod * (1 - self.tax)

    @staticmethod
    def _parse_amount(v):
        """DART 금액(문자/숫자)을 억원(float)으로 파싱"""
        import numpy as np, re
        if v is None:
            return None
        if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v)):
            return float(v) / 1e8
        s = str(v).strip()
        if s in ("", "-", "nan", "None"):
            return None
        if s.startswith("(") and s.endswith(")"):
            s = "-" + s[1:-1]
        s = s.replace(",", "")
        m = re.match(r"^-?\d+(\.\d+)?$", s)
        if not m:
            digits = re.sub(r"[^0-9\.\-]", "", s)
            if digits in ("", "-", "."):
                return None
            s = digits
        try:
            return float(s) / 1e8
        except Exception:
            return None

    @staticmethod
    def _prefer_cfs(df: pd.DataFrame) -> pd.DataFrame:
        """가능하면 연결재무제표(CFS) 우선"""
        if df is None or df.empty:
            return df
        if "fs_div" in df.columns:
            cfs = df[df["fs_div"].astype(str).str.upper() == "CFS"]
            if not cfs.empty:
                return cfs
        return df

    def _pick_row(self, df: pd.DataFrame, names=(), ids=()):
        """IFRS account_id 정확 일치 우선, 없으면 account_nm 부분일치"""
        if df is None or df.empty:
            return None
        d = self._prefer_cfs(df)

        # 1) account_id 정확 일치
        if ids and "account_id" in d.columns:
            target = set(i.lower() for i in ids)
            mask = d["account_id"].astype(str).str.lower().isin(target)
            cand = d[mask]
            if not cand.empty:
                return cand.iloc[0]

        # 2) account_nm 부분일치
        if names and "account_nm" in d.columns:
            m = pd.Series(False, index=d.index)
            for n in names:
                m = m | d["account_nm"].astype(str).str.contains(n, na=False)
            cand = d[m]
            if not cand.empty:
                return cand.iloc[0]
        return None

    def _amount_from_row(self, row: pd.Series) -> float:
        """여러 컬럼(thstrm_amount 등)에서 금액 추출 → 억원"""
        for col in ["thstrm_amount", "thstrm_add_amount", "frmtrm_amount", "bfefrmtrm_amount"]:
            if isinstance(row, pd.Series) and (col in row.index):
                v = self._parse_amount(row[col])
                if v is not None:
                    return float(v)
        return 0.0

    def extract_metrics(self, fs_map: dict[int, pd.DataFrame], log=None) -> dict[int, dict]:
        m = {}
        for y, fs in fs_map.items():
            try:
                def pick(df, sj):
                    if df is None or df.empty:
                        return df
                    if "sj_div" in df.columns:
                        return df[df["sj_div"] == sj]
                    return df

                is_df = pick(fs, "IS")
                bs_df = pick(fs, "BS")
                cf_df = pick(fs, "CF")

                # IFRS id 우선 + 한글명 폴백 (몇 가지 변형 id도 포함)
                rev_row   = self._pick_row(is_df, names=["매출", "수익"],
                                           ids=["ifrs-full_Revenue", "ifrs_Revenue", "Revenue"])
                ebit_row  = self._pick_row(is_df, names=["영업이익"],
                                           ids=["ifrs-full_OperatingIncomeLoss", "OperatingIncomeLoss"])
                net_row   = self._pick_row(is_df, names=["당기순이익", "분기순이익", "지배기업 소유주지분 순이익"],
                                           ids=["ifrs-full_ProfitLoss",
                                                "ifrs-full_ProfitLossAttributableToOwnersOfParent",
                                                "ProfitLoss"])
                assets_row = self._pick_row(bs_df, names=["자산총계"],
                                            ids=["ifrs-full_Assets", "Assets"])
                equity_row = self._pick_row(bs_df, names=["자본총계", "지배기업 소유주지분"],
                                            ids=["ifrs-full_Equity",
                                                 "ifrs-full_EquityAttributableToOwnersOfParent",
                                                 "Equity"])
                ocf_row = self._pick_row(
                    cf_df,
                    names=["영업활동현금흐름", "영업활동 현금흐름", "영업활동으로 인한 현금흐름", "영업활동으로부터의 현금흐름", "영업활동현금흐름(간접법)"],
                    ids=[
                        "ifrs-full_CashFlowsFromUsedInOperatingActivities",
                        "ifrs_CashFlowsFromUsedInOperatingActivities",
                        "ifrs-full_CashFlowsFromUsedInOperatingActivitiesIndirectMethod",
                    ],
                )

                vals = {
                    "revenue":       self._amount_from_row(rev_row)    if rev_row   is not None else 0.0,
                    "ebit":          self._amount_from_row(ebit_row)   if ebit_row  is not None else 0.0,
                    "net_income":    self._amount_from_row(net_row)    if net_row   is not None else 0.0,
                    "total_assets":  self._amount_from_row(assets_row) if assets_row is not None else 0.0,
                    "total_equity":  self._amount_from_row(equity_row) if equity_row is not None else 0.0,
                    "operating_cf":  self._amount_from_row(ocf_row)    if ocf_row   is not None else 0.0,
                }
                m[y] = vals

                if log:
                    def _lr(tag, row):
                        if row is None:
                            log("warning", "METRIC_MATCH", f"{tag} 미발견", year=y)
                        else:
                            log("debug", "METRIC_MATCH", f"{tag} 매칭",
                                year=y,
                                account_id=str(row.get("account_id", "")),
                                account_nm=str(row.get("account_nm", "")),
                                thstrm=str(row.get("thstrm_amount", "")))
                    _lr("revenue", rev_row); _lr("ebit", ebit_row); _lr("net_income", net_row)
                    _lr("assets", assets_row); _lr("equity", equity_row); _lr("operating_cf", ocf_row)

            except Exception as ex:
                if log:
                    log("error", "METRICS", "지표 추출 실패", year=y, error=str(ex))
        return m

    def value(
        self,
        metrics: dict[int, dict],
        beta: float = 1.0,
        years: int = 5,
        window=None,
        *,
        # ▶ 새로 추가된 선택 인자들 (넘겨와도 무시 가능)
        shares_out: float | None = None,      # 발행주식수(주)
        price_now: float | None = None,       # 현재가(원)
        net_debt: float | None = None,        # 순차입금(억원) 있으면 주면 좋음
        log=None,
        **_                                         # 앞으로 추가될 인자도 안전 흡수
    ) -> dict | None:
        if len(metrics) < 3:
            return None
        if window is None:
            window = min(years, len(metrics))
        ys = sorted(metrics.keys())
        latest = ys[-1]

        # 성장률 g
        revs = [metrics[y]["revenue"] for y in ys[-window:]]
        g_list = []
        for i in range(1, len(revs)):
            if revs[i-1] > 0 and revs[i] > 0:
                g_list.append(revs[i]/revs[i-1] - 1)
        g = float(np.mean(g_list)) if g_list else 0.05
        g = max(-0.1, min(0.3, g))

        # 영업이익률 margin
        m_list = []
        for y in ys[-window:]:
            rev = metrics[y]["revenue"]
            if rev > 0:
                m_list.append(metrics[y]["ebit"]/rev)
        margin = float(np.mean(m_list)) if m_list else 0.1
        margin = max(0.0, min(0.5, margin))

        w = self.wacc(beta)
        base = metrics[latest]["revenue"]
        if base <= 0:
            return None

        rows = []
        for t in range(1, years + 1):
            rev = base * ((1 + g) ** t)
            ebit = rev * margin
            nopat = ebit * (1 - self.tax)
            fcf = nopat * 0.8
            pv = fcf / ((1 + w) ** t)
            rows.append({"year": t, "revenue": rev, "ebit": ebit, "fcf": fcf, "pv_fcf": pv})

        terminal_fcf = rows[-1]["fcf"] * (1 + self.g)
        tv = terminal_fcf / (w - self.g)
        pv_tv = tv / ((1 + w) ** years)
        ev = sum(r["pv_fcf"] for r in rows) + pv_tv  # 기업가치(억원)

        # 선택: 순차입금/주식수 있으면 자본가치/주당가도 계산
        equity_value = None
        intrinsic_per_share = None
        if net_debt is not None:
            equity_value = ev - net_debt
        if equity_value is not None and shares_out and shares_out > 0:
            intrinsic_per_share = (equity_value * 1e8) / shares_out  # 억원→원

        if callable(log):
            log("debug", "DCF_VALUE", "calc done",
                wacc=f"{w:.4f}", g=f"{g:.4f}", margin=f"{margin:.4f}",
                ev_억원=f"{ev:,.0f}",
                equity_억원=None if equity_value is None else f"{equity_value:,.0f}",
                per_share=None if intrinsic_per_share is None else f"{intrinsic_per_share:,.0f}",
                shares_out=shares_out, net_debt=net_debt, price_now=price_now)

        return {
            "enterprise_value": ev,
            "equity_value": equity_value,
            "intrinsic_per_share": intrinsic_per_share,
            "fcf_rows": rows,
            "terminal_value": tv,
            "wacc": w,
            "growth": g,
            "margin": margin,
            "assumptions": {
                "years": years,
                "terminal_growth": self.g,
                "tax": self.tax,
                "beta": beta,
                "shares_out": shares_out,
                "price_now": price_now,
                "net_debt": net_debt,
            },
        }


# --- SRIMModel 보강: BPS·지속구간 반영 ---
class SRIMModel:
    def __init__(self):
        self.rfr = 0.035
        self.mrp = 0.06
        self.crp = 0.005

    def roe_parts(self, metrics: dict[int, dict]) -> dict[int, dict]:
        out = {}
        for y, m in metrics.items():
            if m["total_equity"] > 0 and m["revenue"] > 0 and m["total_assets"] > 0:
                nm = m["net_income"] / m["revenue"]
                at = m["revenue"] / m["total_assets"]
                em = m["total_assets"] / m["total_equity"]
                out[y] = {"roe": nm*at*em, "net_margin": nm, "asset_turnover": at, "equity_multiplier": em}
        return out

    def value(
        self,
        metrics: dict[int, dict],
        parts: dict[int, dict],
        beta: float = 1.0,
        window=3,
        *,
        bps_now: float | None = None,   # ▶ 새 인자
        payout: float = 0.3,
        log=None,
        **_
    ) -> dict | None:
        if len(parts) < 2:
            return None

        rr = self.rfr + beta * (self.mrp + self.crp)
        window = min(window, len(parts))
        recent = [parts[y]["roe"] for y in sorted(parts.keys())[-window:]]
        s_roe = float(np.mean(recent)) if recent else 0.1
        s_roe = max(0.0, min(0.5, s_roe))

        retain = 1 - payout
        g = s_roe * retain

        # 현재 BPS가 들어오면 사용, 없으면 단순치(개선 여지)
        current_bps = bps_now if (bps_now and bps_now > 0) else 50000.0

        if s_roe <= rr:
            iv = current_bps
            excess = 0.0
        else:
            excess = s_roe - rr
            if rr <= g:
                iv = current_bps * 2
            else:
                iv = current_bps + (excess * current_bps) / (rr - g)

        if callable(log):
            log("debug", "SRIM_VALUE", "calc done",
                s_roe=f"{s_roe:.4f}", rr=f"{rr:.4f}", g=f"{g:.4f}",
                payout=f"{payout:.2f}", bps_now=current_bps, iv=f"{iv:,.0f}")

        return {
            "intrinsic_value": iv,
            "sustainable_roe": s_roe,
            "required_return": rr,
            "growth_rate": g,
            "excess_roe": excess,
            "current_bps": current_bps,
            "roe_components": parts,
        }


# =============================
# ReportBuilder (+ 진단 로그)
# =============================
class ReportBuilder:
    def __init__(self, ticker: str, days: int, beta: float, years: int, use_dart: bool):
        self.ticker = ticker.strip()
        self.days = int(days)
        self.beta = float(beta)
        self.years = int(years)
        self.use_dart = use_dart and bool(DART_KEY)
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.logs: list[dict] = []  # 🔧 진단 로그
        self.company: dict | None = None
        self.metrics: dict | None = None
        self.dcf: dict | None = None
        self.srim: dict | None = None
        self.price_df: pd.DataFrame | None = None
        self.inv_df: pd.DataFrame | None = None
        self.ticker_name: str = _ticker_name(self.ticker)
        self.current_price: float | None = None
        self._price_info: dict | None = None  # 최근일 잠정/확정가 모두 보관
        self.fig_price = None
        self.fig_flow = None
        self.fig_flow_cum = None

    # ---------- 로깅 유틸 ----------
    def _log(self, level: str, stage: str, msg: str, **ctx):
        safe_ctx = {str(k): str(v) for k, v in (ctx or {}).items()}
        self.logs.append({
            "ts": datetime.now().strftime("%H:%M:%S"),
            "level": level.upper(),
            "stage": stage,
            "message": msg,
            "context": safe_ctx,
        })

    def _dbg_fn(self, stage: str):
        def _fn(event: str, message: str, **ctx):
            self._log("debug", stage, f"{event}: {message}", **ctx)
        return _fn

    # ---------- 내부: 최신/확정 종가 산출(폴백) ----------
    @staticmethod
    def _calc_latest_prices(ticker: str):
        """pykrx 일자 3일치로 '잠정 최신 종가'와 '확정 종가'를 구분해 반환
        - latest_close/latest_date: 조회 구간의 마지막 영업일 종가
        - settled_close/settled_date: 마지막 전일(=데이터가 하루 더 쌓여 확정) 종가
        """
        if not PYKRX_AVAILABLE:
            return None
        try:
            end = datetime.now()
            start = end - timedelta(days=7)
            s, e = start.strftime("%Y%m%d"), end.strftime("%Y%m%d")
            df = stock.get_market_ohlcv_by_date(s, e, ticker)
            if df is None or df.empty:
                return None
            df = df.sort_index()
            dates = list(df.index)
            if not dates:
                return None
            latest_date = dates[-1]
            latest_close = float(df.loc[latest_date, "종가"])
            # 확정 종가: 직전 영업일(데이터가 하루 더 들어와 '잠정 오차'가 해소된 값으로 간주)
            settled_date = dates[-2] if len(dates) >= 2 else dates[-1]
            settled_close = float(df.loc[settled_date, "종가"])
            return {
                "latest_date": latest_date.strftime("%Y-%m-%d"),
                "latest_close": latest_close,
                "settled_date": settled_date.strftime("%Y-%m-%d"),
                "settled_close": settled_close,
            }
        except Exception:
            return None

    # ---------- 검증 ----------
    def validate(self) -> bool:
        self._log("info", "VALIDATE", "입력 검증 시작", ticker=self.ticker, days=self.days, beta=self.beta, years=self.years, use_dart=self.use_dart)
        if len(self.ticker) != 6 or not self.ticker.isdigit():
            self.errors.append("종목 코드는 6자리 숫자여야 합니다 (예: 005930).")
            self._log("error", "VALIDATE", "종목 코드 형식 오류")
        if self.days <= 0 or self.days > 365 * 2:
            self.errors.append("분석 기간은 1~730일 사이로 입력해주세요.")
            self._log("error", "VALIDATE", "기간 범위 오류", days=self.days)
        if not PYKRX_AVAILABLE:
            self.errors.append("PyKrx가 필요합니다. 'pip install pykrx'")
            self._log("error", "ENV", "PyKrx 미설치")
        if self.use_dart and not DART_AVAILABLE:
            self.warnings.append("OpenDartReader 미설치: 밸류에이션이 제한될 수 있습니다.")
            self._log("warning", "ENV", "OpenDartReader 미설치")
        if not self.use_dart:
            self.warnings.append("DART 비활성화: 재무/밸류에이션 섹션이 축약됩니다.")
            self._log("info", "ENV", "DART 비활성화")
        ok = len(self.errors) == 0
        self._log("info", "VALIDATE", "입력 검증 종료", ok=ok)
        return ok

    # ---------- 수집 ----------
    def collect(self):
        self._log("info", "COLLECT", "수집 시작")
        end = datetime.now() - timedelta(days=1)
        start = end - timedelta(days=self.days)
        s, e = start.strftime("%Y%m%d"), end.strftime("%Y%m%d")

        # OHLCV
        try:
            self.price_df = _ohlcv(self.ticker, s, e)
            n = 0 if self.price_df is None else len(self.price_df)
            if self.price_df is None or self.price_df.empty:
                self._log("warning", "OHLCV", "가격 데이터 없음", start=s, end=e)
            else:
                self._log("info", "OHLCV", "가격 데이터 수집", rows=n, cols=list(self.price_df.columns))
        except Exception as ex:
            self._log("error", "OHLCV", "가격 데이터 수집 실패", error=str(ex), tb=traceback.format_exc())

        # 투자자별
        try:
            self.inv_df = _investor_daily(self.ticker, s, e, debug=self._dbg_fn("INVESTOR"))
            n = 0 if self.inv_df is None else len(self.inv_df)
            if self.inv_df is None or self.inv_df.empty:
                self._log("warning", "INVESTOR", "투자자별 데이터 없음", start=s, end=e)
            else:
                self._log("info", "INVESTOR", "투자자별 데이터 수집", rows=n, cols=list(self.inv_df.columns))
        except Exception as ex:
            self._log("error", "INVESTOR", "투자자별 데이터 수집 실패", error=str(ex), tb=traceback.format_exc())

        # 현재가(잠정/확정 구분)
        try:
            price_info = None
            try:
                # 외부 헬퍼가 있으면 우선 사용
                price_info = _latest_prices(self.ticker, log=self._log)  # type: ignore[name-defined]
            except Exception:
                # 없으면 내부 폴백
                price_info = self._calc_latest_prices(self.ticker)

            if price_info:
                self._price_info = price_info
                # 보고서 KPI는 '확정 종가' 사용
                self.current_price = price_info.get("settled_close")
                self._log("info", "PRICE_NOW", "현재가 조회",
                          settled=price_info.get("settled_close"),
                          settled_date=price_info.get("settled_date"),
                          latest=price_info.get("latest_close"),
                          latest_date=price_info.get("latest_date"))
            else:
                self._log("warning", "PRICE_NOW", "가격 정보 산출 실패")
        except Exception as ex:
            self._log("error", "PRICE_NOW", "현재가 조회 예외", error=str(ex), tb=traceback.format_exc())

        # DART → 재무/밸류에이션
        if self.use_dart and DART_KEY:
            dart = DartDataCollector(DART_KEY, log=self._log)
            self.company = dart.company(self.ticker)  # 표시용
            corp = (self.company or {}).get("corp_code")
            if corp:
                fs_map = dart.fin_map(corp, years=self.years)
                try:
                    dcf_model = DCFModel()

                    # 재무 지표 추출(감가/Capex/NWC/이자비용/이자성부채까지)
                    self.metrics = dcf_model.extract_metrics(fs_map, log=self._log)
                    if not self.metrics:
                        self._log("warning", "METRICS", "재무 지표 추출 결과 없음")
                    else:
                        # 합계 로그(파싱 성공 여부 가늠)
                        sum_keys = ["revenue", "ebit", "net_income", "total_assets", "total_equity",
                                    "operating_cf", "da", "capex", "nwc", "interest_expense", "ib_debt"]
                        sums = {k: float(sum((v.get(k, 0.0) or 0.0) for v in self.metrics.values())) for k in sum_keys}
                        self._log("info", "METRICS_SUM", "지표 합계(억원 / 일부항목은 억원기준)", **sums)
                        if sums.get("revenue", 0.0) == 0.0:
                            self._log("error", "METRICS_ZERO", "매출액 합계가 0 → 계정 매핑/단위 파싱 재검토 필요")

                    # ---- 연동 포인트: 주식수/시가총액/BPS ----
                    shares_out = None
                    try:
                        # 만주 → 주
                        shares_out = get_share_count(self.ticker) * 10000  # type: ignore[name-defined]
                    except Exception:
                        shares_out = None

                    price_now = self.current_price or None
                    bps_now = None
                    if self.metrics and shares_out:
                        latest_y = max(self.metrics.keys())
                        eq_억원 = self.metrics[latest_y].get("total_equity", 0.0) or 0.0
                        if eq_억원 > 0:
                            bps_now = (eq_억원 * 1e8) / float(shares_out)  # 원/주

                    # ---- DCF 계산(시장가중/실효 CoD 반영) ----
                    self.dcf = dcf_model.value(
                        self.metrics or {}, beta=self.beta, years=self.years, window=min(self.years, 5),
                        shares_out=shares_out, price_now=price_now, log=self._log
                    )
                    self._log("info", "ASSUMPTIONS", "DCF 파라미터", years=self.years, window=min(self.years, 5))
                    if self.dcf is None:
                        self._log("warning", "DCF", "DCF 계산 실패 또는 데이터 부족")
                    else:
                        self._log("info", "DCF", "DCF 계산 완료", EV_억원=round(self.dcf["enterprise_value"], 0))

                    # ---- S-RIM 계산(BPS/지속계수 반영) ----
                    sr = SRIMModel()
                    parts = sr.roe_parts(self.metrics or {})
                    if not parts:
                        self._log("warning", "SRIM", "ROE 분해 결과 없음")
                    sr_window = min(self.years, max(2, len(parts))) if parts else 2
                    self.srim = sr.value(
                        self.metrics or {}, parts, beta=self.beta, window=sr_window,
                        bps_now=bps_now, persistence=0.6, log=self._log
                    )
                    self._log("info", "ASSUMPTIONS", "SRIM 파라미터", window=sr_window, bps_now=bps_now)
                    if self.srim is None:
                        self._log("warning", "SRIM", "S-RIM 계산 실패 또는 데이터 부족")
                    else:
                        self._log("info", "SRIM", "S-RIM 계산 완료", IV=self.srim.get("intrinsic_value"))
                except Exception as ex:
                    self._log("error", "VALUATION", "밸류에이션 계산 중 예외", error=str(ex), tb=traceback.format_exc())
            else:
                self._log("warning", "DART", "corp_code 미확보 → DART 재무 수집 생략")
        else:
            self._log("info", "DART", "DART 비사용 경로")

        self._log("info", "COLLECT", "수집 종료")

    # ---------- 차트 ----------
    def build_charts(self):
        try:
            if self.price_df is not None and not self.price_df.empty:
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=self.price_df.index,
                    open=self.price_df.get("시가"),
                    high=self.price_df.get("고가"),
                    low=self.price_df.get("저가"),
                    close=self.price_df.get("종가"),
                    name="주가",
                ))
                fig.update_layout(title=f"{self.ticker_name} ({self.ticker}) 주가", height=420)
                self.fig_price = fig
                self._log("info", "CHART", "가격 차트 생성")
            else:
                self._log("warning", "CHART", "가격 데이터 없음으로 차트 생략")
        except Exception as ex:
            self._log("error", "CHART", "가격 차트 생성 실패", error=str(ex), tb=traceback.format_exc())

        try:
            if self.inv_df is not None and not self.inv_df.empty:
                valid = [c for c in ["개인", "기관합계", "외국인", "기타법인"] if c in self.inv_df.columns]
                if valid:
                    fig1 = go.Figure()
                    for c in valid:
                        fig1.add_trace(go.Scatter(x=self.inv_df.index, y=self.inv_df[c], mode="lines+markers", name=c))
                    fig1.update_layout(title="투자자별 순매수 추이", height=360, hovermode="x unified")
                    self.fig_flow = fig1

                    cum = self.inv_df[valid].fillna(0).cumsum()
                    fig2 = go.Figure()
                    for c in valid:
                        fig2.add_trace(go.Scatter(x=cum.index, y=cum[c], mode="lines", name=c))
                    fig2.update_layout(title="투자자별 누적 순매수", height=360, hovermode="x unified")
                    self.fig_flow_cum = fig2
                    self._log("info", "CHART", "수급 차트 생성", cols=valid)
                else:
                    self._log("warning", "CHART", "투자자 컬럼 없음", available=list(self.inv_df.columns))
            else:
                self._log("warning", "CHART", "투자자 데이터 없음으로 차트 생략")
        except Exception as ex:
            self._log("error", "CHART", "수급 차트 생성 실패", error=str(ex), tb=traceback.format_exc())

    # ---------- 내보내기 ----------
    def export(self) -> Path | None:
        out_dir = Path("./reports"); out_dir.mkdir(exist_ok=True)
        base = f"report_{self.ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        html_path = out_dir / f"{base}.html"

        # ---------- HTML (항상 생성, 풀 콘텐츠) ----------
        try:
            parts = []
            parts.append("""
            <html><head><meta charset="utf-8">
            <style>
              body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Noto Sans KR',Arial,sans-serif;margin:24px;}
              h1,h2,h3{margin:8px 0;}
              .kpi{display:flex;flex-wrap:wrap;gap:12px;margin:8px 0 16px;}
              .kpi div{background:#f6f6f8;padding:10px 12px;border-radius:10px;}
              table{border-collapse:collapse;width:100%;font-size:14px;}
              td,th{border:1px solid #eee;padding:6px 8px;text-align:right;}
              th:first-child,td:first-child{text-align:left;}
              .section{margin-top:24px;}
              .caption{color:#777;font-size:12px;margin-top:-10px;}
            </style>
            </head><body>
            """)
            parts.append(f"<h2>종합 보고서: {self.ticker_name} ({self.ticker})</h2>")
            parts.append(f"<p>생성일: {datetime.now():%Y-%m-%d %H:%M}</p>")

            # KPI
            parts.append('<div class="kpi">')
            if self.current_price:
                parts.append(f"<div>확정 종가: <b>{self.current_price:,.0f}원</b></div>")
            if self._price_info:
                parts.append(f"<div>최근일 잠정: <b>{self._price_info['latest_close']:,.0f}원</b></div>")
            if self.dcf:
                parts.append(f"<div>DCF EV(억원): <b>{self.dcf['enterprise_value']:,.0f}</b></div>")
                parts.append(f"<div>WACC: <b>{self.dcf['wacc']:.2%}</b></div>")
            if self.srim:
                parts.append(f"<div>S-RIM 적정가(원/주): <b>{self.srim['intrinsic_value']:,.0f}</b></div>")
            parts.append("</div>")

            # 차트(인터랙티브)
            figs = [f for f in [self.fig_price, self.fig_flow, self.fig_flow_cum] if f is not None]
            if figs:
                parts.append('<div class="section"><h3>차트</h3>')
                first = True
                for fig in figs:
                    html_fig = pio.to_html(fig, full_html=False, include_plotlyjs=True if first else False,
                                           config={"displaylogo": False, "responsive": True})
                    parts.append(html_fig); first = False
                parts.append("</div>")

            # 재무 표
            if self.metrics:
                df = pd.DataFrame(self.metrics).T.round(0)
                parts.append('<div class="section"><h3>재무 지표(요약)</h3>')
                parts.append(df.to_html(border=0, justify="right"))
                parts.append("</div>")

            # 수급 표
            if self.inv_df is not None and not self.inv_df.empty:
                tmp = self.inv_df.copy(); tmp.index = tmp.index.strftime('%Y-%m-%d')
                parts.append('<div class="section"><h3>투자자별 순매수(일별)</h3>')
                parts.append('<div class="caption">최근 60영업일</div>')
                parts.append(tmp.tail(60).to_html(border=0))
                parts.append("</div>")

            # 디테일 JSON
            if self.dcf or self.srim:
                parts.append('<div class="section"><h3>밸류에이션 상세</h3>')
                if self.dcf:  parts.append(f"<pre>{pd.Series(self.dcf).to_string()}</pre>")
                if self.srim: parts.append(f"<pre>{pd.Series(self.srim).to_string()}</pre>")
                parts.append("</div>")

            parts.append("</body></html>")
            html_path.write_text("".join(parts), encoding="utf-8")
            self._log("info", "EXPORT", "HTML 저장 완료", path=str(html_path))
        except Exception as ex:
            self._log("error", "EXPORT", "HTML 저장 실패", error=str(ex), tb=traceback.format_exc())
            return None

        # ---------- PDF (kaleido 있을 때만 이미지 삽입) ----------
        if REPORTLAB_AVAILABLE and KALEIDO_AVAILABLE:
            pdf_path = out_dir / f"{base}.pdf"
            try:
                from reportlab.lib.pagesizes import A4
                from reportlab.pdfgen import canvas as pdf_canvas
                from reportlab.lib.units import cm
                c = pdf_canvas.Canvas(str(pdf_path), pagesize=A4)
                w, h = A4; y = h - 2 * cm

                # (선택) 한글 폰트 등록
                try:
                    from reportlab.pdfbase import pdfmetrics
                    from reportlab.pdfbase.ttfonts import TTFont
                    if Path("NanumGothic.ttf").exists():
                        pdfmetrics.registerFont(TTFont("NanumGothic", "NanumGothic.ttf"))
                        font = "NanumGothic"
                    else:
                        font = "Helvetica"
                except Exception:
                    font = "Helvetica"

                c.setFont(font, 16); c.drawString(2*cm, y, f"종합 보고서: {self.ticker_name} ({self.ticker})"); y -= 0.9*cm
                c.setFont(font, 10); c.drawString(2*cm, y, f"생성일: {datetime.now():%Y-%m-%d %H:%M}"); y -= 0.6*cm
                if self.current_price is not None:
                    c.drawString(2*cm, y, f"확정 종가: {self.current_price:,.0f}원"); y -= 0.6*cm
                if self._price_info:
                    c.drawString(2*cm, y, f"최근일 잠정: {self._price_info['latest_close']:,.0f}원 ({self._price_info['latest_date']})"); y -= 0.6*cm
                if self.dcf:
                    c.drawString(2*cm, y, f"DCF EV(억원): {self.dcf['enterprise_value']:,.0f} / WACC {self.dcf['wacc']:.2%}"); y -= 0.6*cm
                if self.srim:
                    c.drawString(2*cm, y, f"S-RIM 적정가: {self.srim['intrinsic_value']:,.0f}원/주"); y -= 0.8*cm

                def draw_fig(fig, yy):
                    img = out_dir / f"{base}_{np.random.randint(1e9)}.png"
                    pio.write_image(fig, str(img), width=1000, height=520, scale=2)
                    h_img = 9.2*cm
                    c.drawImage(str(img), 1.5*cm, yy - h_img, width=w - 3*cm, height=h_img, preserveAspectRatio=True)
                    try: img.unlink()
                    except Exception: pass
                    return yy - (h_img + 0.6*cm)

                for fig in [self.fig_price, self.fig_flow, self.fig_flow_cum]:
                    if fig is None: continue
                    if y < 12*cm: c.showPage(); y = h - 2*cm; c.setFont(font, 10)
                    y = draw_fig(fig, y)

                c.showPage(); c.save()
                self._log("info", "EXPORT", "PDF 저장 완료", path=str(pdf_path))
                return pdf_path
            except Exception as ex:
                self._log("error", "EXPORT", "PDF 저장 실패", error=str(ex), tb=traceback.format_exc())
                return html_path

        # kaleido 없으면 HTML만 반환
        return html_path

    # ---------- 렌더 ----------
    def render(self):
        for m in self.warnings:
            st.warning(m)

        st.subheader("📌 개요")
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("종목명", self.ticker_name)
        with c2: st.metric("종목코드", self.ticker)
        with c3: st.metric("분석기간(영업일)", f"{self.days}")
        with c4:
            if self._price_info:
                st.metric("현재가(확정)", f"{self.current_price:,.0f}원" if self.current_price else "N/A")
                st.caption(
                    f"최근일(잠정) 종가: {self._price_info['latest_close']:,.0f}원 "
                    f"({self._price_info['latest_date']}) · "
                    f"확정 종가: {self._price_info['settled_close']:,.0f}원 "
                    f"({self._price_info['settled_date']})"
                )
            else:
                st.metric("현재가", f"{self.current_price:,.0f}원" if self.current_price else "N/A")

        st.markdown("---")
        st.subheader("📈 가격 추이")
        if self.fig_price: st.plotly_chart(self.fig_price, use_container_width=True)
        else: st.info("가격 데이터를 불러오지 못했습니다.")

        st.markdown("---")
        st.subheader("👥 투자자 수급")
        if self.fig_flow:
            col = st.columns(2)
            with col[0]: st.plotly_chart(self.fig_flow, use_container_width=True)
            with col[1]:
                if self.fig_flow_cum: st.plotly_chart(self.fig_flow_cum, use_container_width=True)
        else:
            st.info("투자자별 순매수 데이터를 불러오지 못했습니다.")

        st.markdown("---")
        st.subheader("💰 밸류에이션 요약 (DCF/S-RIM)")
        if self.dcf or self.srim:
            c1, c2, c3 = st.columns(3)
            with c1:
                if self.dcf:
                    st.metric("DCF EV(억원)", f"{self.dcf['enterprise_value']:,.0f}")
                    st.caption(f"WACC {self.dcf['wacc']:.2%} | g {self.dcf['growth']:.2%} | margin {self.dcf['margin']:.2%}")
                else:
                    st.metric("DCF EV(억원)", "N/A")
            with c2:
                if self.srim:
                    st.metric("S-RIM 적정가(원/주)", f"{self.srim['intrinsic_value']:,.0f}")
                    st.caption(f"ROE {self.srim['sustainable_roe']:.2%} | r {self.srim['required_return']:.2%}")
                else:
                    st.metric("S-RIM 적정가", "N/A")
            with c3:
                if self.srim and self.current_price:
                    fair = self.srim["intrinsic_value"]
                    pct = (fair - self.current_price) / self.current_price * 100
                    st.metric("현재가 대비", f"{pct:+.1f}%")
                else:
                    st.metric("현재가 대비", "N/A")
            with st.expander("DCF 상세"):
                st.json(self.dcf or {"info": "데이터 없음"})
            with st.expander("S-RIM 상세"):
                st.json(self.srim or {"info": "데이터 없음"})
        else:
            st.info("밸류에이션을 계산할 충분한 데이터가 없습니다.")

        st.markdown("---")
        st.subheader("📊 재무 지표(요약)")
        if self.metrics:
            df = pd.DataFrame(self.metrics).T.round(0)
            st.dataframe(df.style.format("{:,.0f}"), use_container_width=True)
        else:
            st.info("재무 데이터가 없습니다.")

        st.markdown("---")
        st.subheader("📎 부록")
        st.caption("원천 데이터 일부를 확인할 수 있습니다.")
        with st.expander("가격 데이터"):
            if self.price_df is not None and not self.price_df.empty:
                tmp = self.price_df.copy(); tmp.index = tmp.index.strftime("%Y-%m-%d")
                st.dataframe(tmp, use_container_width=True)
            else:
                st.write("N/A")
        with st.expander("투자자별 순매수(일별)"):
            if self.inv_df is not None and not self.inv_df.empty:
                tmp = self.inv_df.copy(); tmp.index = tmp.index.strftime("%Y-%m-%d")
                st.dataframe(tmp, use_container_width=True)
            else:
                st.write("N/A")

        st.markdown("---")
        with st.expander("🔧 진단 로그 (수집/계산 과정)", expanded=True):
            if self.logs:
                df = pd.DataFrame(self.logs)
                if "context" not in df.columns:
                    df["context"] = ""
                def _fmt_ctx(x):
                    if x is None: return ""
                    if isinstance(x, float) and (math.isnan(x) or np.isnan(x)): return ""
                    if isinstance(x, dict): return "; ".join(f"{k}={v}" for k, v in x.items())
                    return str(x)
                df["context"] = df["context"].apply(_fmt_ctx)
                order = [c for c in ["ts", "level", "stage", "message", "context"] if c in df.columns]
                st.dataframe(df[order], use_container_width=True, height=260)
            else:
                st.write("로그가 없습니다.")


# =============================
# Streamlit 진입점(메인에서 호출)
# =============================

def run():
    # rerun에도 보고서를 유지하기 위해 세션 상태 사용
    if "rpt" not in st.session_state:
        st.session_state.rpt = None
    if "report_ready" not in st.session_state:
        st.session_state.report_ready = False

    st.title("📊 주식 분석 (종합 보고서)")

    # 메인 앱 사이드바를 침범하지 않도록, 본문 상단 컨트롤 패널 사용
    with st.container():
        with st.form("controls"):
            st.subheader("⚙️ 분석 설정")
            c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1])
            with c1:
                ticker = st.text_input("종목 코드", value="005930", help="예: 005930 (삼성전자)")
            with c2:
                days = st.slider("분석 기간(영업일)", 7, 180, 90, step=1)
            with c3:
                beta = st.number_input("베타", value=1.0, min_value=0.1, max_value=3.0, step=0.1)
            with c4:
                years = st.selectbox("재무 반영 연수", [3, 5, 7], index=1)

            with st.expander("고급 설정", expanded=False):
                use_dart = st.checkbox("DART 사용", value=bool(DART_KEY), help="config.ini에 키가 있어야 활성화됩니다.")

            submitted = st.form_submit_button("📄 종합 보고서 생성")

    # 제출 시 새 보고서 생성 → 세션에 저장
    if submitted:
        rpt = ReportBuilder(ticker=ticker, days=days, beta=beta, years=years, use_dart=use_dart)
        if rpt.validate():
            with st.spinner("데이터 수집 중..."):
                rpt.collect()
            with st.spinner("차트 구성 중..."):
                rpt.build_charts()
            st.session_state.rpt = rpt
            st.session_state.report_ready = True
        else:
            for e in rpt.errors:
                st.error(e)

    # 제출 여부와 관계없이, 세션의 보고서를 항상 렌더
    if st.session_state.report_ready and st.session_state.rpt:
        rpt = st.session_state.rpt
        rpt.render()

        st.markdown("---")
        st.subheader("🖨️ 보고서 내보내기")
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("PDF/HTML 저장", key="export_btn", use_container_width=True):
                out = rpt.export()
                if out and out.exists():
                    st.success(f"저장 완료: {out.name}")
                    try:
                        st.download_button("다운로드", data=open(out, "rb").read(), file_name=out.name, key=f"dl_{out.name}")
                    except Exception:
                        st.info("다운로드 제한 시, 보고서 폴더의 파일을 직접 확인하세요.")
                else:
                    st.error("내보내기에 실패했습니다.")
        with col2:
            st.caption("ReportLab + kaleido 설치 시 PDF, 미설치 시 HTML로 자동 저장됩니다. HTML은 브라우저 인쇄로 PDF 저장 가능.")
    else:
        st.info("상단의 설정을 입력하고 ‘종합 보고서 생성’을 눌러주세요.")


if __name__ == "__main__":
    run()
