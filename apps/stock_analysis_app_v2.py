"stock_analysis_app_v2 (리팩터링 + 진단 로그 강화)\n- 메인 앱의 사이드바를 사용하지 않는 서브앱 버전\n- 상단 컨트롤 패널(본문 영역)에서 1회 입력 → 단일 종합 보고서 렌더\n- ReportBuilder 클래스로 수집/검증/시각화/내보내기 일원화\n- DART 기반 재무/밸류에이션(옵션), PyKrx 기반 가격/수급\n- PDF(ReportLab+Kaleido) 또는 HTML 자동 내보내기\n- 🔧 어디서 실패했는지 알 수 있도록 단계별 진단 로그 출력\n\n주의: set_page_config는 메인에서만 호출합니다.\n"

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path
import time
import pickle
import os
import configparser
import warnings
import traceback
import math
import re
from datetime import timezone

warnings.filterwarnings("ignore")

# =============================\n# 환경 설정/외부 라이브러리 체크\n# =============================
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

# OpenDartReader는 배포 버전에 따라 생성자 접근 방식이 다릅니다.\n# - 어떤 환경: from OpenDartReader import OpenDartReader; OpenDartReader(api_key)\n# - 다른 환경: import OpenDartReader; OpenDartReader.OpenDartReader(api_key)
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

# =============================\n# 캐시/헬퍼
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


# =============================\n# DART 수집 & 모델 (견고화)
# =============================
class DartDataCollector:
    def __init__(self, api_key: str | None, log=None):
        self.api_key = api_key
        self._log = log
        if api_key and DART_AVAILABLE and ODR_CTOR is not None:
            try:
                self.dart = ODR_CTOR(api_key)  # 배포 방식 차이를 흡수한 생성자
                if self._log: self._log("info", "DART_INIT", "OpenDartReader 초기화 성공")
            except Exception as ex:
                if self._log: self._log("error", "DART_INIT", "OpenDartReader 초기화 실패", error=str(ex), tb=traceback.format_exc())
                self.dart = None
        else:
            if self._log: self._log("warning", "DART_INIT", "DART 사용 불가 (키 없음/미설치)")
            self.dart = None

    def company(self, ticker: str) -> dict | None:
        """DART 회사 기본 정보 조회 (여러 폴백 적용).
        1) dart.company(티커)
        2) dart.find_corp_code(티커)
        3) pykrx 종목명 → dart.company_by_name(종목명)
        """
        if not self.dart:
            if self._log: self._log("warning", "DART_COMPANY", "DART 핸들 없음")
            return None
        # 1) company by ticker (신규 버전 제공)
        try:
            info = self.dart.company(str(ticker))
            if isinstance(info, dict):
                cc = info.get("corp_code") or info.get("corpcode")
                nm = info.get("corp_name") or info.get("corpname")
                if cc:
                    if self._log: self._log("info", "DART_COMPANY", "company() 성공", corp_code=cc, corp_name=nm)
                    return {"corp_code": str(cc), "corp_name": str(nm or ticker), "stock_code": str(ticker)}
        except Exception as ex:
            if self._log: self._log("warning", "DART_COMPANY", "company() 실패", error=str(ex))
        # 2) 직접 corp_code 찾기
        try:
            cc = self.dart.find_corp_code(str(ticker))
            if cc:
                try:
                    nm = get_market_ticker_name(str(ticker))
                except Exception:
                    nm = str(ticker)
                if self._log: self._log("info", "DART_COMPANY", "find_corp_code() 성공", corp_code=cc, corp_name=nm)
                return {"corp_code": str(cc), "corp_name": str(nm), "stock_code": str(ticker)}
        except Exception as ex:
            if self._log: self._log("warning", "DART_COMPANY", "find_corp_code() 실패", error=str(ex))
        # 3) 종목명으로 조회
        try:
            nm = get_market_ticker_name(str(ticker))
            byname = self.dart.company_by_name(str(nm))
            if hasattr(byname, "empty") and not byname.empty:
                row = byname.iloc[0]
                cc = row.get("corp_code") or row.get("corpcode")
                name = row.get("corp_name") or row.get("corpname") or nm
                if cc:
                    if self._log: self._log("info", "DART_COMPANY", "company_by_name() 성공", corp_code=cc, corp_name=name)
                    return {"corp_code": str(cc), "corp_name": str(name), "stock_code": str(ticker)}
        except Exception as ex:
            if self._log: self._log("warning", "DART_COMPANY", "company_by_name() 실패", error=str(ex))
        if self._log: self._log("warning", "DART_COMPANY", "모든 매칭 실패", ticker=ticker)
        return None

    def fin_map(self, corp_code: str, years: int = 5) -> dict[int, pd.DataFrame]:
        if not self.dart:
            if self._log: self._log("warning", "DART_FS", "DART 핸들 없음")
            return {}
        out = {}
        this_year = datetime.now().year
        # 사업(11011) → 반기(11012) → 1분기(11013) → 3분기(11014) 순서 폴백
        reprt_codes = ["11011", "11012", "11013", "11014"]
        for y in range(this_year - years, this_year):
            got = False
            for rc in reprt_codes:
                try:
                    fs = self.dart.finstate(str(corp_code), y, reprt_code=rc)
                    if fs is not None and not getattr(fs, "empty", False):
                        out[y] = fs
                        if self._log: self._log("info", "DART_FS", "재무제표 수집", year=y, reprt_code=rc, rows=len(fs))
                        got = True
                        break
                    else:
                        if self._log: self._log("debug", "DART_FS", "빈 재무제표", year=y, reprt_code=rc)
                except Exception as ex:
                    if self._log: self._log("warning", "DART_FS", "재무제표 조회 실패", year=y, reprt_code=rc, error=str(ex))
            if not got and self._log:
                self._log("warning", "DART_FS", "해당 연도 보고서 미확보", year=y)
        if not out and self._log:
            self._log("warning", "DART_FS", "수집된 재무제표가 없습니다.")
        return out


class DCFModel:
    def __init__(self):
        self.rfr = 0.035
        self.mrp = 0.06
        self.crp = 0.005
        self.tax = 0.25
        self.g = 0.025

    def wacc(self, beta: float, debt_ratio: float = 0.3) -> float:
        coe = self.rfr + beta * (self.mrp + self.crp)
        cod = self.rfr + 0.02
        return (1 - debt_ratio) * coe + debt_ratio * cod * (1 - self.tax)

    def _parse_amount(self, v):
        """DART 금액(문자/숫자)을 억원(float)으로 파싱"""
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
            digits = re.sub(r"[^0-9\.\-ről]", "", s)
            if digits in ("", "-", "."):
                return None
            s = digits
        try:
            return float(s) / 1e8
        except Exception:
            return None

    def _prefer_cfs(self, df: pd.DataFrame) -> pd.DataFrame:
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
        # 2) account_nm contains
        if names and "account_nm" in d.columns:
            m = pd.Series(False, index=d.index)
            for n in names:
                m = m | d["account_nm"].astype(str).str.contains(n, na=False)
            cand = d[m]
            if not cand.empty:
                return cand.iloc[0]
        return None

    def _amount_from_row(self, row: pd.Series) -> float:
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

                rev_row   = self._pick_row(is_df, names=["매출", "수익"], ids=["ifrs-full_Revenue", "ifrs_Revenue"])
                ebit_row  = self._pick_row(is_df, names=["영업이익"], ids=["ifrs-full_OperatingIncomeLoss"])
                net_row   = self._pick_row(is_df, names=["당기순이익", "분기순이익", "지배기업 소유주지분 순이익"],
                                           ids=["ifrs-full_ProfitLoss", "ifrs-full_ProfitLossAttributableToOwnersOfParent"])
                assets_row = self._pick_row(bs_df, names=["자산총계"], ids=["ifrs-full_Assets"])
                equity_row = self._pick_row(bs_df, names=["자본총계", "지배기업 소유주지분"],
                                            ids=["ifrs-full_Equity", "ifrs-full_EquityAttributableToOwnersOfParent"])
                ocf_row    = self._pick_row(cf_df, names=["영업활동현금흐름", "영업활동 현금흐름"],
                                            ids=["ifrs-full_CashFlowsFromUsedInOperatingActivities"])

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
                    def _log_row(tag, row):
                        if row is None:
                            log("warning", "METRIC_MATCH", f"{tag} 미발견", year=y)
                        else:
                            log("debug", "METRIC_MATCH", f"{tag} 매칭",
                                year=y,
                                account_id=str(row.get("account_id", "")) if row.get("account_id") else "",
                                account_nm=str(row.get("account_nm", "")) if row.get("account_nm") else "",
                                thstrm=str(row.get("thstrm_amount", "")) if row.get("thstrm_amount") else "")
                    _log_row("revenue", rev_row)
                    _log_row("ebit", ebit_row)
                    _log_row("net_income", net_row)
                    _log_row("assets", assets_row)
                    _log_row("equity", equity_row)
                    _log_row("operating_cf", ocf_row)

            except Exception as ex:
                if log:
                    log("error", "METRICS", "연도별 지표 추출 실패", year=y, error=str(ex))
        return m

    def value(self, metrics: dict[int, dict], beta: float = 1.0, years: int = 5, window=None) -> dict | None:
        if len(metrics) < 3:
            return None
        if window is None:
            window = min(years, len(metrics))
        ys = sorted(metrics.keys())
        latest = ys[-1]
        # 성장률: 윈도우 내 연속 연도만 사용
        revs = [metrics[y]["revenue"] for y in ys[-window:]]
        g_list = []
        for i in range(1, len(revs)):
            if revs[i-1] > 0 and revs[i] > 0:
                g_list.append(revs[i]/revs[i-1] - 1)
        g = float(np.mean(g_list)) if g_list else 0.05
        g = max(-0.1, min(0.3, g))
        # 마진: 윈도우 평균
        m_list = []
        for y in ys[-window:]:
            rev = metrics[y]["revenue"]
            if rev > 0:
                m_list.append(metrics[y]["ebit"]/rev)
        margin = float(np.mean(m_list)) if m_list else 0.1
        margin = max(0.0, min(0.5, margin))
        # WACC
        w = self.wacc(beta)
        # FCF 프로젝트
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
        ev = sum(r["pv_fcf"] for r in rows) + pv_tv
        return {
            "enterprise_value": ev,
            "fcf_rows": rows,
            "terminal_value": tv,
            "wacc": w,
            "growth": g,
            "margin": margin,
            "assumptions": {"years": years, "terminal_growth": self.g, "tax": self.tax, "beta": beta},
        }


class SRIMModel:
    def __init__(self):
        self.rfr = 0.035
        self.mrp = 0.06
        self.crp = 0.005

    def roe_parts(self, metrics: dict[int, dict]) -> dict[int, dict]:
        out = {}
        for y, m in metrics.items():
            if m["total_equity"] > 0 and m["revenue"] > 0 and m["total_assets"] > 0:
                net_margin = m["net_income"] / m["revenue"]
                at = m["revenue"] / m["total_assets"]
                em = m["total_assets"] / m["total_equity"]
                roe = net_margin * at * em
                out[y] = {"roe": roe, "net_margin": net_margin, "asset_turnover": at, "equity_multiplier": em}
        return out

    def value(self, metrics: dict[int, dict], parts: dict[int, dict], beta: float = 1.0, window=3, current_bps: float | None = None) -> dict | None:
        if len(parts) < 2:
            return None
        rr = self.rfr + beta * (self.mrp + self.crp)
        window = min(window, len(parts))
        recent = [parts[y]["roe"] for y in sorted(parts.keys())[-window:]]
        s_roe = float(np.mean(recent)) if recent else 0.1
        s_roe = max(0.0, min(0.5, s_roe))
        payout = 0.3
        retain = 1 - payout
        g = s_roe * retain

        if current_bps is None or current_bps <= 0:
            return None  # BPS 없이는 계산 불가

        if s_roe <= rr:
            iv = current_bps
            excess = 0.0
        else:
            excess = s_roe - rr
            if rr <= g:
                iv = current_bps * 2
            else:
                iv = current_bps + (excess * current_bps) / (rr - g)
        return {
            "intrinsic_value": iv,
            "sustainable_roe": s_roe,
            "required_return": rr,
            "growth_rate": g,
            "excess_roe": excess,
            "current_bps": current_bps,
            "roe_components": parts,
        }


# =============================\n# ReportBuilder (+ 진단 로그)
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
        self.fig_price = None
        self.fig_flow = None
        self.fig_flow_cum = None
        self.comprehensive_analysis: str | None = None  # 종합 분석 결과

    def _generate_comprehensive_analysis(self) -> str:
        """
        수집된 데이터를 바탕으로 주가, 수급, 밸류에이션을 종합적으로 분석합니다.
        """
        analysis_points = []
        positive_signals = 0

        # 1. 주가 수준 분석
        if self.price_df is not None and not self.price_df.empty and self.current_price is not None:
            recent_high = self.price_df['고가'].max()
            recent_low = self.price_df['저가'].min()
            if recent_high > recent_low:
                position = (self.current_price - recent_low) / (recent_high - recent_low)
                if position < 0.2:  # 하위 20% 수준
                    analysis_points.append(f"현재 주가는 최근 {self.days}일 가격대 중 하위 20% 수준으로, 단기적인 가격 매력이 있습니다.")
                    positive_signals += 1
                elif position > 0.8: # 상위 20% 수준
                    analysis_points.append(f"현재 주가는 최근 {self.days}일 가격대 중 상위 20% 수준으로, 단기 과열에 대한 주의가 필요합니다.")
                else:
                    analysis_points.append(f"현재 주가는 최근 {self.days}일 가격대의 약 {position:.0%} 지점에 위치해 있습니다.")

        # 2. 수급 분석
        if self.inv_df is not None and len(self.inv_df) > 5:
            # 최근 5일간의 순매수 합계
            last_5d_net = self.inv_df.tail(5).sum()
            foreign_net = last_5d_net.get('외국인', 0)
            inst_net = last_5d_net.get('기관합계', 0)
            retail_net = last_5d_net.get('개인', 0)
            
            if foreign_net > 0 and inst_net > 0 and retail_net < 0:
                analysis_points.append("최근 5일간 외국인과 기관의 동반 순매수세가 유입되는 가운데 개인은 매도하여, 수급이 매우 긍정적입니다.")
                positive_signals += 2 # 동반 매수는 강력한 신호
            elif foreign_net > 0 and retail_net < 0:
                analysis_points.append("최근 외국인의 매수세가 유입되고 개인이 매도하는 경향을 보여, 수급이 긍정적입니다.")
                positive_signals += 1
            elif inst_net > 0 and retail_net < 0:
                analysis_points.append("최근 기관의 매수세가 유입되고 개인이 매도하는 경향을 보여, 수급이 양호합니다.")
                positive_signals += 1
            elif foreign_net < 0 and inst_net < 0 and retail_net > 0:
                analysis_points.append("최근 외국인과 기관이 동반 순매도하고 개인이 매수하는 경향을 보여, 단기 수급 부담이 매우 큽니다.")

        # 3. 밸류에이션 분석
        is_undervalued = False
        if self.srim and self.srim.get('intrinsic_value') and self.current_price:
            iv = self.srim['intrinsic_value']
            undervalued_pct = (iv - self.current_price) / self.current_price
            if undervalued_pct > 0.25:  # 25% 이상 저평가
                analysis_points.append(f"S-RIM 모델 기준 적정주가는 약 {iv:,.0f}원으로, 현재보다 {undervalued_pct:.1%} 높은 수준의 저평가 상태입니다.")
                is_undervalued = True
                positive_signals += 1
            elif undervalued_pct < -0.25:
                analysis_points.append(f"S-RIM 모델 기준 적정주가는 약 {iv:,.0f}원으로, 현재보다 {abs(undervalued_pct):.1%} 낮은 수준의 고평가 상태입니다.")

        # 4. 종합 의견 및 추천
        if not analysis_points:
            return "분석할 데이터가 부족하여 종합 의견을 도출할 수 없습니다."

        recommendation = "중립"
        if is_undervalued and positive_signals >= 2:
            recommendation = "긍정적 매수 검토"
        elif positive_signals >= 2:
            recommendation = "긍정적"
        elif positive_signals == 0 and not is_undervalued:
            recommendation = "보수적 접근 필요"
        
        summary = "\n".join(f"- {point}" for point in analysis_points)
        summary += f"\n\n**종합 의견:** 위의 분석들을 종합해 볼 때, 현 시점 투자 매력도에 대한 의견은 **'{recommendation}'** 입니다."
        
        return summary

    def run_comprehensive_analysis(self):
        self._log("info", "COMP_ANALYSIS", "종합 분석 시작")
        try:
            self.comprehensive_analysis = self._generate_comprehensive_analysis()
            self._log("info", "COMP_ANALYSIS", "종합 분석 완료")
        except Exception as e:
            self._log("error", "COMP_ANALYSIS", "종합 분석 중 오류 발생", error=str(e))
            self.comprehensive_analysis = "종합 분석 중 오류가 발생했습니다."

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
        # 현재가
        try:
            px = _latest_prices(self.ticker, log=self._log)
            if px:
                self.current_price = px["settled_close"]
                self._price_info = px
            else:
                self._log("warning", "PRICE_NOW", "가격 정보 산출 실패")
        except Exception as ex:
            self._log("error", "PRICE_NOW", "현재가 조회 예외", error=str(ex), tb=traceback.format_exc())
        
        # DART
        if self.use_dart and DART_KEY:
            dart = DartDataCollector(DART_KEY, log=self._log)
            self.company = dart.company(self.ticker)
            corp = (self.company or {}).get("corp_code")
            if corp:
                fs_map = dart.fin_map(corp, years=self.years)
                try:
                    dcf_model = DCFModel()
                    self.metrics = dcf_model.extract_metrics(fs_map, log=self._log)
                    if not self.metrics:
                        self._log("warning", "METRICS", "재무 지표 추출 결과 없음")
                    else:
                        sums = {k: sum((v.get(k, 0.0) or 0.0) for v in self.metrics.values()) 
                                for k in ["revenue", "ebit", "net_income", "total_assets", "total_equity", "operating_cf"]}
                        self._log("info", "METRICS_SUM", "지표 합계(억원)", **sums)
                        if sums["revenue"] == 0.0:
                            self._log("error", "METRICS_ZERO", "매출액 합계가 0 → 파싱 실패 가능(계정명/ID/하이픈/단위)")
                    
                    self.dcf = dcf_model.value(self.metrics or {}, beta=self.beta, years=self.years, window=min(self.years, 5))
                    self._log("info", "ASSUMPTIONS", "DCF 파라미터", years=self.years, window=min(self.years, 5))
                    if self.dcf is None:
                        self._log("warning", "DCF", "DCF 계산 실패 또는 데이터 부족(기저 매출=0)")
                    else:
                        self._log("info", "DCF", "DCF 계산 완료", EV_억원=round(self.dcf["enterprise_value"], 0))
                    
                    calculated_bps = None
                    try:
                        cap_df = stock.get_market_cap_by_date(s, e, self.ticker)
                        if not cap_df.empty:
                            num_shares = cap_df['상장주식수'].iloc[-1]
                            self._log("info", "SHARES", "상장주식수 확인", count=num_shares)
                            if self.metrics:
                                latest_year = sorted(self.metrics.keys())[-1]
                                latest_equity = self.metrics[latest_year].get("total_equity")
                                if latest_equity and latest_equity > 0 and num_shares > 0:
                                    calculated_bps = (latest_equity * 100_000_000) / num_shares
                                    self._log("info", "BPS_CALC", "BPS 계산 완료", bps=round(calculated_bps, 2), equity_억원=latest_equity, shares=num_shares)
                                else:
                                    self._log("warning", "BPS_CALC", "BPS 계산 불가 (자본총계 또는 주식수 없음)")
                        else:
                            self._log("warning", "SHARES", "시가총액/상장주식수 데이터 없음")
                    except Exception as ex:
                        self._log("error", "BPS_CALC", "BPS 계산 중 예외 발생", error=str(ex))

                    sr = SRIMModel(); parts = sr.roe_parts(self.metrics or {})
                    if not parts:
                        self._log("warning", "SRIM", "ROE 분해 결과 없음")
                    sr_window = min(self.years, max(2, len(parts)))
                    self.srim = sr.value(self.metrics or {}, parts, beta=self.beta, window=sr_window, current_bps=calculated_bps)
                    self._log("info", "ASSUMPTIONS", "SRIM 파라미터", window=sr_window, bps=calculated_bps)
                    if self.srim is None:
                        self._log("warning", "SRIM", "S-RIM 계산 실패 (BPS 데이터 부족 가능성)")
                    else:
                        self._log("info", "SRIM", "S-RIM 계산 완료", IV=self.srim.get("intrinsic_value"))
                except Exception as ex:
                    self._log("error", "VALUATION", "밸류에이션 계산 중 예외", error=str(ex), tb=traceback.format_exc())
            else:
                self._log("warning", "DART", "corp_code 미확보 → DART 재무 수집 생략")
        else:
            self._log("info", "DART", "DART 비사용 경로")
        
        # 종합 분석 기능 호출
        self.run_comprehensive_analysis()
        
        self._log("info", "COLLECT", "수집 종료")

    # ---------- 차트 ----------
    def build_charts(self):
        price_ok = self.price_df is not None and not self.price_df.empty
        inv_ok = self.inv_df is not None and not self.inv_df.empty

        def set_rangebreaks(fig, df):
            if df is None or df.empty:
                return
            all_days = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
            missing_days = all_days.difference(df.index)
            fig.update_xaxes(rangebreaks=[dict(values=missing_days.strftime('%Y-%m-%d'))])

        try:
            if price_ok:
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=self.price_df.index,
                    open=self.price_df.get("시가"),
                    high=self.price_df.get("고가"),
                    low=self.price_df.get("저가"),
                    close=self.price_df.get("종가"),
                    name="주가",
                    increasing=dict(line_color="#D60000", fillcolor="#D60000"),
                    decreasing=dict(line_color="#0051D6", fillcolor="#0051D6"),
                ))
                fig.update_layout(
                    title=f"{self.ticker_name} ({self.ticker}) 주가",
                    height=400,
                    xaxis_rangeslider_visible=False
                )
                set_rangebreaks(fig, self.price_df)
                self.fig_price = fig
                self._log("info", "CHART", "가격 차트 생성")
            else:
                self._log("warning", "CHART", "가격 데이터 없음으로 차트 생략")
        except Exception as ex:
            self._log("error", "CHART", "가격 차트 생성 실패", error=str(ex), tb=traceback.format_exc())

        try:
            if inv_ok:
                valid = [c for c in ["개인", "기관합계", "외국인", "기타법인"] if c in self.inv_df.columns]
                if valid:
                    fig2 = go.Figure()
                    cum = self.inv_df[valid].fillna(0).cumsum()
                    for c in valid:
                        fig2.add_trace(go.Scatter(x=cum.index, y=cum[c], mode="lines", name=c))
                    fig2.update_layout(title="투자자별 누적 순매수", height=300, hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                    set_rangebreaks(fig2, self.inv_df)
                    self.fig_flow_cum = fig2

                    fig1 = go.Figure()
                    for c in valid:
                        fig1.add_trace(go.Bar(x=self.inv_df.index, y=self.inv_df[c], name=c))
                    fig1.update_layout(title=f"투자자별 순매수 추이 - {self.days}일", height=300, hovermode="x unified", barmode='relative', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                    set_rangebreaks(fig1, self.inv_df)
                    self.fig_flow = fig1

                    self._log("info", "CHART", "수급 차트 생성", cols=valid)
                else:
                    self._log("warning", "CHART", "투자자 컬럼 없음", available=list(self.inv_df.columns))
            else:
                self._log("warning", "CHART", "투자자 데이터 없음으로 차트 생략")
        except Exception as ex:
            self._log("error", "CHART", "수급 차트 생성 실패", error=str(ex), tb=traceback.format_exc())

    # ---------- 내보내기 ----------
    def export(self) -> Path | None:
        out_dir = Path("./reports");
        out_dir.mkdir(exist_ok=True)
        base = f"report_{self.ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        html_path = out_dir / f"{base}.html"

        try:
            parts = []
            parts.append("""
            <html><head><meta charset=\"utf-8\">
            <style>
              body{font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Noto Sans KR', Arial, sans-serif; margin:24px;}
              h1,h2,h3{margin: 8px 0;}
              .kpi{display:flex; gap:16px; margin: 8px 0 16px;}
              .kpi div{background:#f6f6f8; padding:10px 12px; border-radius:10px;}
              table{border-collapse:collapse; width:100%; font-size:14px;}
              td,th{border:1px solid #eee; padding:6px 8px; text-align:right;}
              th:first-child, td:first-child{text-align:left;}
              .section{margin-top:24px;}
            </style>
            </head><body>
            """)
            parts.append(f"<h2>종합 보고서: {self.ticker_name} ({self.ticker})</h2>")
            parts.append(f"<p>생성일: {datetime.now():%Y-%m-%d %H:%M}</p>")
            parts.append('<div class="kpi">')
            parts.append(
                f"<div>현재가: <b>{self.current_price:,.0f}원</b></div>" if self.current_price else "<div>현재가: N/A</div>")
            if self.dcf:
                parts.append(f"<div>DCF EV(억원): <b>{self.dcf['enterprise_value']:,.0f}</b></div>")
                parts.append(f"<div>WACC: <b>{self.dcf['wacc']:.2%}</b></div>")
            if self.srim:
                parts.append(f"<div>S-RIM 적정가(원/주): <b>{self.srim['intrinsic_value']:,.0f}</b></div>")
            parts.append("</div>")

            if self.comprehensive_analysis:
                parts.append(f'<div class="section"><h3>종합 분석</h3><p>{self.comprehensive_analysis.replace("\n", "<br>")}</p></div>')

            figs = [f for f in [self.fig_price, self.fig_flow, self.fig_flow_cum] if f is not None]
            if figs:
                parts.append('<div class="section"><h3>차트</h3>')
                first = True
                for fig in figs:
                    html_fig = pio.to_html(fig, full_html=False, include_plotlyjs=True if first else False,
                                           config={"displaylogo": False, "responsive": True})
                    parts.append(html_fig)
                    first = False
                parts.append("</div>")

            if self.metrics:
                df = pd.DataFrame(self.metrics).T.round(0)
                parts.append('<div class="section"><h3>재무 지표(요약)</h3>')
                parts.append(df.to_html(border=0, justify="right"))
                parts.append("</div>")

            if self.inv_df is not None and not self.inv_df.empty:
                tmp = self.inv_df.copy();
                tmp.index = tmp.index.strftime('%Y-%m-%d')
                parts.append('<div class="section"><h3>투자자별 순매수(일별)</h3>')
                parts.append(tmp.tail(60).to_html(border=0))
                parts.append("</div>")

            if self.dcf or self.srim:
                parts.append('<div class="section"><h3>밸류에이션 상세</h3>')
                if self.dcf:
                    parts.append(f"<pre>{pd.Series(self.dcf).to_string()}</pre>")
                if self.srim:
                    parts.append(f"<pre>{pd.Series(self.srim).to_string()}</pre>")
                parts.append("</div>")

            parts.append("</body></html>")
            html_path.write_text("".join(parts), encoding="utf-8")
            self._log("info", "EXPORT", "HTML 저장 완료", path=str(html_path))
        except Exception as ex:
            self._log("error", "EXPORT", "HTML 저장 실패", error=str(ex), tb=traceback.format_exc())
            return None

        if REPORTLAB_AVAILABLE and KALEIDO_AVAILABLE:
            pdf_path = out_dir / f"{base}.pdf"
            try:
                from reportlab.lib.pagesizes import A4
                from reportlab.pdfgen import canvas as pdf_canvas
                from reportlab.lib.units import cm
                c = pdf_canvas.Canvas(str(pdf_path), pagesize=A4)
                w, h = A4;
                y = h - 2 * cm

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

                c.setFont(font, 16);
                c.drawString(2 * cm, y, f"종합 보고서: {self.ticker_name} ({self.ticker})");
                y -= 0.9 * cm
                c.setFont(font, 10);
                c.drawString(2 * cm, y, f"생성일: {datetime.now():%Y-%m-%d %H:%M}");
                y -= 0.6 * cm
                if self.current_price is not None:
                    c.drawString(2 * cm, y, f"현재가: {self.current_price:,.0f}원");
                    y -= 0.6 * cm
                if self.dcf:
                    c.drawString(2 * cm, y,
                                 f"DCF EV(억원): {self.dcf['enterprise_value']:,.0f} / WACC {self.dcf['wacc']:.2%}");
                    y -= 0.6 * cm
                if self.srim:
                    c.drawString(2 * cm, y, f"S-RIM 적정가(원/주): {self.srim['intrinsic_value']:,.0f}");
                    y -= 0.8 * cm
                
                if self.comprehensive_analysis:
                    from reportlab.platypus import SimpleDocTemplate, Paragraph
                    from reportlab.lib.styles import getSampleStyleSheet
                    styles = getSampleStyleSheet()
                    p = Paragraph(self.comprehensive_analysis.replace('\n', '<br/>'), styles['BodyText'])
                    p.wrapOn(c, w - 4*cm, h)
                    p.drawOn(c, 2*cm, y - p.height)
                    y -= (p.height + 0.6*cm)

                def draw_fig(fig, yy):
                    img = out_dir / f"{base}_{np.random.randint(1e9)}.png"
                    pio.write_image(fig, str(img), width=1000, height=520, scale=2)
                    h_img = 9.2 * cm
                    c.drawImage(str(img), 1.5 * cm, yy - h_img, width=w - 3 * cm, height=h_img,
                                preserveAspectRatio=True)
                    try:
                        img.unlink()
                    except Exception:
                        pass
                    return yy - (h_img + 0.6 * cm)

                for fig in [self.fig_price, self.fig_flow, self.fig_flow_cum]:
                    if fig is None: continue
                    if y < 12 * cm: c.showPage(); y = h - 2 * cm; c.setFont(font, 10)
                    y = draw_fig(fig, y)

                c.showPage();
                c.save()
                self._log("info", "EXPORT", "PDF 저장 완료", path=str(pdf_path))
                return pdf_path
            except Exception as ex:
                self._log("error", "EXPORT", "PDF 저장 실패", error=str(ex), tb=traceback.format_exc())
                return html_path

        return html_path

    # ---------- 렌더 (컴포넌트화) ----------
    def render_overview(self):
        st.subheader("📌 개요 및 밸류에이션")
        
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            st.markdown(f"### {self.ticker_name} ({self.ticker})")
        with c2:
            st.metric("현재가", f"{self.current_price:,.0f}원" if self.current_price else "N/A")
        with c3:
            if self.srim and self.current_price:
                fair = self.srim['intrinsic_value']
                pct = (fair - self.current_price) / self.current_price * 100
                st.metric("현재가 대비", f"{pct:+.1f}%")
            else:
                st.metric("현재가 대비", "N/A")

        st.markdown("---")

        if self.dcf or self.srim:
            c1, c2 = st.columns(2)
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
        else:
            st.info("밸류에이션을 계산할 충분한 데이터가 없습니다.")

        st.markdown("---")
        st.subheader("🔍 종합 분석 (Gemini 분석)")
        if self.comprehensive_analysis:
            st.markdown(self.comprehensive_analysis, unsafe_allow_html=True)
        else:
            st.info("종합 분석 정보가 없습니다.")

    def render_charts(self):
        st.subheader(f"📈 종합 차트 ({self.days}일)")
        if self.fig_price:
            st.plotly_chart(self.fig_price, use_container_width=True)
        if self.fig_flow_cum:
            st.plotly_chart(self.fig_flow_cum, use_container_width=True)
        if self.fig_flow:
            st.plotly_chart(self.fig_flow, use_container_width=True)

        if not self.fig_price and not self.fig_flow:
             st.info("차트 데이터를 불러오지 못했습니다.")

    def render_financials(self):
        st.subheader("📊 재무 지표(요약)")
        if self.metrics:
            df = pd.DataFrame(self.metrics).T.round(0)
            st.dataframe(df.style.format("{:,.0f}"), use_container_width=True)
        else:
            st.info("재무 데이터가 없습니다.")

    def render_appendix(self):
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
        with st.expander("DCF/S-RIM 상세 정보"):
            st.json({"DCF": self.dcf or "데이터 없음", "SRIM": self.srim or "데이터 없음"})

    def render_logs(self):
        with st.expander("🔧 진단 로그 (수집/계산 과정)", expanded=True):
            if self.logs:
                df = pd.DataFrame(self.logs)
                if "context" not in df.columns:
                    df["context"] = ""
                def _fmt_ctx(x):
                    if x is None:
                        return ""
                    if isinstance(x, float) and (math.isnan(x) or np.isnan(x)):
                        return ""
                    if isinstance(x, dict):
                        return "; ".join(f"{k}={v}" for k, v in x.items())
                    return str(x)
                df["context"] = df["context"].apply(_fmt_ctx)
                order = [c for c in ["ts", "level", "stage", "message", "context"] if c in df.columns]
                st.dataframe(df[order], use_container_width=True, height=240)
            else:
                st.write("로그가 없습니다.")


# =============================\n# Streamlit 진입점(메인에서 호출)
# =============================
@st.cache_data
def get_stock_list():
    """로컬 stock_list.csv 파일에서 종목 리스트를 불러옵니다."""
    file_path = "./apps/stock_list.csv"
    if not os.path.exists(file_path):
        st.error(f"'{file_path}' 파일이 없습니다. 먼저 `update_stock_list.py`를 실행하여 종목 리스트를 생성해주세요.")
        return []
    try:
        df = pd.read_csv(file_path)
        # 이름으로 정렬
        df = df.sort_values(by="name")
        # "종목명 (코드)" 형식으로 포맷. 코드는 6자리로 맞춤
        formatted_list = [f"{row['name']} ({str(row['code']).zfill(6)})".replace("\n", " ") for index, row in df.iterrows()]
        return formatted_list
    except Exception as e:
        st.error(f"'{file_path}' 파일 로딩 중 오류 발생.")
        st.exception(e)
        return []

def run():
    # --- 상태 초기화 ---
    if "report_ready" not in st.session_state:
        st.session_state.report_ready = False
    if "rpt" not in st.session_state:
        st.session_state.rpt = None
    if "aux_rpt" not in st.session_state:
        st.session_state.aux_rpt = None

    st.title("📊 주식 분석 (종합 보고서)")

    # --- 데이터 로딩 ---
    stock_list_for_ui = get_stock_list()

    # --- 컨트롤 폼 ---
    with st.form("analysis_controls"):
        st.subheader("⚙️ 분석 설정")
        
        c1, c2, c3, c4 = st.columns([2, 1, 1, 1])

        with c1:
            if not stock_list_for_ui:
                st.warning("종목 리스트를 불러올 수 없습니다.")
                selected_stock = None
            else:
                # 기본 선택값으로 '삼성전자' 설정
                try:
                    samsung_str = next(s for s in stock_list_for_ui if s.startswith("삼성전자"))
                    default_index = stock_list_for_ui.index(samsung_str)
                except (StopIteration, ValueError):
                    default_index = 0
                
                selected_stock = st.selectbox(
                    "종목 선택",
                    options=stock_list_for_ui,
                    index=default_index,
                    help="목록에서 분석할 종목을 선택하세요. 드롭다운을 열고 종목명이나 코드를 입력하여 검색할 수 있습니다."
                )
        
        with c2:
            days = st.slider("메인 기간(일)", 7, 730, 90, step=1)
        with c3:
            beta = st.number_input("베타", value=1.0, min_value=0.1, max_value=3.0, step=0.1)
        with c4:
            years = st.selectbox("재무 연수", [3, 5, 7], index=1)

        with st.expander("고급 설정"):
            use_dart = st.checkbox("DART 재무분석 사용", value=bool(DART_KEY), help="config.ini에 키가 있어야 활성화됩니다.")
            use_aux_chart = st.checkbox("보조차트 활성화", help="오른쪽에 다른 기간의 보조차트를 함께 표시합니다.")
            aux_days = st.slider("보조차트 기간(일)", 7, 365, 60, step=1, disabled=not use_aux_chart)

        submitted = st.form_submit_button("📄 종합 보고서 생성")

    # --- 보고서 생성 및 렌더링 ---
    if submitted:
        if selected_stock:
            match = re.search(r'\((\d{6})\)', selected_stock)
            ticker = match.group(1) if match else ""
            
            st.session_state.report_ready = False
            main_rpt = ReportBuilder(ticker=ticker, days=days, beta=beta, years=years, use_dart=use_dart)
            
            if main_rpt.validate():
                with st.spinner("메인 보고서 데이터 수집 중..."):
                    main_rpt.collect()
                    main_rpt.build_charts()
                st.session_state.rpt = main_rpt
                st.session_state.report_ready = True
            else:
                for e in main_rpt.errors:
                    st.error(f"메인 보고서 오류: {e}")
                st.session_state.rpt = None

            if use_aux_chart and st.session_state.get("report_ready"):
                aux_rpt = ReportBuilder(ticker=ticker, days=aux_days, beta=beta, years=years, use_dart=False)
                if aux_rpt.validate():
                    with st.spinner(f"{aux_days}일 보조 보고서 생성 중..."):
                        aux_rpt.collect()
                        aux_rpt.build_charts()
                    st.session_state.aux_rpt = aux_rpt
                else:
                    for e in aux_rpt.errors:
                        st.warning(f"보조 보고서 오류: {e}")
            else:
                st.session_state.aux_rpt = None
        else:
            st.error("종목을 선택해주세요.")
            st.session_state.report_ready = False

    if st.session_state.get("report_ready") and st.session_state.get("rpt"):
        main_rpt = st.session_state.rpt
        aux_rpt = st.session_state.get("aux_rpt")

        main_rpt.render_overview()

        if aux_rpt:
            col1, col2 = st.columns(2)
            with col1:
                main_rpt.render_charts()
            with col2:
                aux_rpt.render_charts()
        else:
            main_rpt.render_charts()

        st.markdown("---")
        main_rpt.render_financials()
        st.markdown("---")
        main_rpt.render_appendix()
        st.markdown("---")
        main_rpt.render_logs()

        st.markdown("---")
        st.subheader("🖨️ 보고서 내보내기")
        if st.button("PDF/HTML 저장"):
            out = main_rpt.export()
            if out and out.exists():
                st.success(f"저장 완료: {out.name}")
                with open(out, "rb") as f:
                    st.download_button("다운로드", f, file_name=out.name)
            else:
                st.error("내보내기에 실패했습니다.")
    else:
        st.info("분석할 종목을 선택한 후 ‘종합 보고서 생성’ 버튼을 눌러주세요.")


if __name__ == "__main__":
    run()
