"""
apps/stock_analysis_app.py
주식 분석 통합 애플리케이션 - 최종 수정 버전
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import pickle
import os
from pathlib import Path
import configparser
import warnings
warnings.filterwarnings('ignore')

is_debugging = False

config = configparser.ConfigParser()
config.read('config.ini')
dart_key = config['DART']['key']

# PyKrx 임포트 (에러 처리 포함)
try:
    from pykrx import stock
    from pykrx.stock import get_market_trading_value_by_investor, get_market_ticker_list, get_market_ticker_name
    PYKRX_AVAILABLE = True
except ImportError:
    PYKRX_AVAILABLE = False
    st.error("❌ PyKrx 라이브러리가 설치되지 않았습니다. 'pip install pykrx'로 설치해주세요.")

try:
    import OpenDartReader
    DART_AVAILABLE = True
except ImportError:
    DART_AVAILABLE = False


class DartDataCollector:
    """DART API를 활용한 재무데이터 수집"""

    def __init__(self, api_key=None):
        self.api_key = api_key
        if api_key and DART_AVAILABLE:
            self.dart = OpenDartReader.OpenDartReader(api_key)
        else:
            self.dart = None

    def get_company_info(self, ticker):
        """종목 코드로 회사 정보 조회"""
        if not self.dart:
            return None

        try:
            # 종목 코드로 회사 정보 검색
            corp_list = self.dart.list()
            company = corp_list[corp_list['stock_code'] == ticker]

            if not company.empty:
                return {
                    'corp_code': company.iloc[0]['corp_code'],
                    'corp_name': company.iloc[0]['corp_name'],
                    'stock_code': ticker
                }
        except Exception as e:
            st.error(f"회사 정보 조회 실패: {e}")
        return None

    def get_financial_statements(self, corp_code, years=5):
        """재무제표 데이터 수집"""
        if not self.dart:
            return None

        try:
            current_year = datetime.now().year
            financial_data = {}

            for year in range(current_year - years, current_year):
                try:
                    # 연결재무제표 우선, 없으면 별도재무제표
                    fs_data = self.dart.finstate(corp_code, year, reprt_code='11011')
                    if fs_data is not None and not fs_data.empty:
                        financial_data[year] = fs_data
                        st.success(f"✅ {year}년 재무제표 수집 완료")
                    else:
                        st.warning(f"⚠️ {year}년 재무제표 없음")
                except Exception as e:
                    st.warning(f"⚠️ {year}년 데이터 수집 실패: {e}")

            return financial_data
        except Exception as e:
            st.error(f"재무제표 수집 실패: {e}")
        return None


class DCFModel:
    """DCF 밸류에이션 모델"""

    def __init__(self):
        self.risk_free_rate = 0.035  # 한국 10년 국고채
        self.market_premium = 0.06  # 시장 위험 프리미엄
        self.country_risk = 0.005  # 국가 위험 프리미엄
        self.tax_rate = 0.25  # 한국 법인세율
        self.terminal_growth = 0.025  # 영구성장률

    def calculate_wacc(self, beta, debt_ratio=0.3):
        """가중평균자본비용 계산"""
        cost_of_equity = self.risk_free_rate + beta * (self.market_premium + self.country_risk)
        cost_of_debt = self.risk_free_rate + 0.02  # 신용스프레드

        wacc = (1 - debt_ratio) * cost_of_equity + debt_ratio * cost_of_debt * (1 - self.tax_rate)
        return wacc

    def extract_financial_metrics(self, financial_data):
        """재무제표에서 주요 지표 추출"""
        metrics = {}

        for year, fs_data in financial_data.items():
            try:
                # 손익계산서 데이터 추출
                income_stmt = fs_data[fs_data['sj_div'] == 'IS']

                revenue = self._find_account_value(income_stmt, ['매출액', '수익(매출액)'])
                ebit = self._find_account_value(income_stmt, ['영업이익'])
                net_income = self._find_account_value(income_stmt, ['당기순이익'])

                # 재무상태표 데이터 추출
                balance_sheet = fs_data[fs_data['sj_div'] == 'BS']
                total_assets = self._find_account_value(balance_sheet, ['자산총계'])
                total_equity = self._find_account_value(balance_sheet, ['자본총계'])

                # 현금흐름표 데이터 추출
                cash_flow = fs_data[fs_data['sj_div'] == 'CF']
                operating_cf = self._find_account_value(cash_flow, ['영업활동현금흐름'])

                metrics[year] = {
                    'revenue': revenue,
                    'ebit': ebit,
                    'net_income': net_income,
                    'total_assets': total_assets,
                    'total_equity': total_equity,
                    'operating_cf': operating_cf
                }

            except Exception as e:
                st.warning(f"{year}년 지표 추출 실패: {e}")
                continue

        return metrics

    def _find_account_value(self, data, account_names):
        """계정과목명으로 값 찾기"""
        for name in account_names:
            found = data[data['account_nm'].str.contains(name, na=False)]
            if not found.empty:
                try:
                    return float(found.iloc[0]['thstrm_amount']) / 100000000  # 억원 단위
                except:
                    continue
        return 0

    def calculate_dcf(self, metrics, beta=1.0, forecast_years=5):
        """DCF 계산"""
        if len(metrics) < 3:
            return None

        years = sorted(metrics.keys())
        latest_year = years[-1]

        # 성장률 계산 (최근 3년 평균)
        revenue_growth_rates = []
        for i in range(len(years) - 2):
            if metrics[years[i + 1]]['revenue'] > 0 and metrics[years[i]]['revenue'] > 0:
                growth = (metrics[years[i + 1]]['revenue'] / metrics[years[i]]['revenue']) - 1
                revenue_growth_rates.append(growth)

        avg_growth = np.mean(revenue_growth_rates) if revenue_growth_rates else 0.05
        avg_growth = max(-0.1, min(0.3, avg_growth))  # -10% ~ 30% 범위 제한

        # 영업이익률 계산
        operating_margins = []
        for year in years[-3:]:
            if metrics[year]['revenue'] > 0:
                margin = metrics[year]['ebit'] / metrics[year]['revenue']
                operating_margins.append(margin)

        avg_margin = np.mean(operating_margins) if operating_margins else 0.1
        avg_margin = max(0, min(0.5, avg_margin))  # 0% ~ 50% 범위 제한

        # WACC 계산
        wacc = self.calculate_wacc(beta)

        # FCF 예측
        base_revenue = metrics[latest_year]['revenue']
        fcf_projections = []

        for year in range(1, forecast_years + 1):
            projected_revenue = base_revenue * ((1 + avg_growth) ** year)
            projected_ebit = projected_revenue * avg_margin
            projected_nopat = projected_ebit * (1 - self.tax_rate)

            # 간단화된 FCF 계산 (투자 등 제외)
            fcf = projected_nopat * 0.8  # 보수적 가정
            pv_fcf = fcf / ((1 + wacc) ** year)

            fcf_projections.append({
                'year': year,
                'revenue': projected_revenue,
                'ebit': projected_ebit,
                'fcf': fcf,
                'pv_fcf': pv_fcf
            })

        # 터미널 가치 계산
        terminal_fcf = fcf_projections[-1]['fcf'] * (1 + self.terminal_growth)
        terminal_value = terminal_fcf / (wacc - self.terminal_growth)
        pv_terminal = terminal_value / ((1 + wacc) ** forecast_years)

        # 기업가치 계산
        total_pv_fcf = sum([proj['pv_fcf'] for proj in fcf_projections])
        enterprise_value = total_pv_fcf + pv_terminal

        return {
            'enterprise_value': enterprise_value,
            'fcf_projections': fcf_projections,
            'terminal_value': terminal_value,
            'wacc': wacc,
            'growth_rate': avg_growth,
            'operating_margin': avg_margin,
            'assumptions': {
                'forecast_years': forecast_years,
                'terminal_growth': self.terminal_growth,
                'tax_rate': self.tax_rate,
                'beta': beta
            }
        }


class SRIMModel:
    """S-RIM 밸류에이션 모델"""

    def __init__(self):
        self.risk_free_rate = 0.035
        self.market_premium = 0.06
        self.country_risk = 0.005

    def calculate_roe_components(self, metrics):
        """ROE 듀폰 분해"""
        roe_data = {}

        for year, data in metrics.items():
            if data['total_equity'] > 0 and data['revenue'] > 0 and data['total_assets'] > 0:
                # ROE = Net Margin × Asset Turnover × Equity Multiplier
                net_margin = data['net_income'] / data['revenue']
                asset_turnover = data['revenue'] / data['total_assets']
                equity_multiplier = data['total_assets'] / data['total_equity']
                roe = net_margin * asset_turnover * equity_multiplier

                roe_data[year] = {
                    'roe': roe,
                    'net_margin': net_margin,
                    'asset_turnover': asset_turnover,
                    'equity_multiplier': equity_multiplier
                }

        return roe_data

    def calculate_srim(self, metrics, roe_data, beta=1.0, forecast_years=5):
        """S-RIM 계산"""
        if len(roe_data) < 2:
            return None

        # 요구수익률 계산
        required_return = self.risk_free_rate + beta * (self.market_premium + self.country_risk)

        # 지속가능 ROE 추정 (최근 3년 평균)
        recent_roes = [data['roe'] for data in list(roe_data.values())[-3:]]
        sustainable_roe = np.mean(recent_roes)
        sustainable_roe = max(0, min(0.5, sustainable_roe))  # 0% ~ 50% 제한

        # 배당성향 추정 (보수적으로 30% 가정)
        payout_ratio = 0.3
        retention_ratio = 1 - payout_ratio
        growth_rate = sustainable_roe * retention_ratio

        # 현재 BPS 계산
        latest_year = max(metrics.keys())
        # 이 부분은 주식수 정보가 필요하므로 간단화
        current_bps = 50000  # 임시값, 실제로는 총자본/발행주식수

        # S-RIM 계산
        if sustainable_roe <= required_return:
            intrinsic_value = current_bps
        else:
            excess_roe = sustainable_roe - required_return
            if required_return <= growth_rate:
                intrinsic_value = current_bps * 2  # 간단한 프리미엄
            else:
                intrinsic_value = current_bps + (excess_roe * current_bps) / (required_return - growth_rate)

        return {
            'intrinsic_value': intrinsic_value,
            'sustainable_roe': sustainable_roe,
            'required_return': required_return,
            'growth_rate': growth_rate,
            'excess_roe': excess_roe,
            'current_bps': current_bps,
            'roe_components': roe_data
        }


class TradingDataManager:
    """투자자별 매매 데이터 관리 클래스"""

    def __init__(self, cache_dir="./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.data_file = self.cache_dir / "trading_data.pkl"
        self.meta_file = self.cache_dir / "data_meta.pkl"

    def load_cached_data(self):
        """캐시된 데이터 로드"""
        try:
            if self.data_file.exists():
                with open(self.data_file, 'rb') as f:
                    data = pickle.load(f)

                with open(self.meta_file, 'rb') as f:
                    meta = pickle.load(f)

                return data, meta
            else:
                return pd.DataFrame(), {}
        except Exception as e:
            st.error(f"❌ 캐시 로드 실패: {e}")
            return pd.DataFrame(), {}

    def save_data(self, data, meta):
        """데이터 저장"""
        try:
            with open(self.data_file, 'wb') as f:
                pickle.dump(data, f)

            with open(self.meta_file, 'wb') as f:
                pickle.dump(meta, f)

        except Exception as e:
            st.error(f"❌ 데이터 저장 실패: {e}")

    def get_missing_dates(self, existing_data, start_date, end_date):
        """누락된 날짜 찾기"""
        if existing_data.empty:
            date_range = pd.bdate_range(start_date, end_date)
            return [d.strftime("%Y%m%d") for d in date_range]

        existing_dates = set(existing_data['date'].dt.strftime("%Y%m%d").unique())
        target_dates = pd.bdate_range(start_date, end_date)
        target_date_strs = [d.strftime("%Y%m%d") for d in target_dates]

        missing_dates = [d for d in target_date_strs if d not in existing_dates]
        return missing_dates

    def fetch_trading_data(self, date_str, ticker, max_retries=3):
        """특정 날짜와 종목의 거래 데이터 조회"""
        if not PYKRX_AVAILABLE:
            return None

        for attempt in range(max_retries):
            try:
                df = get_market_trading_value_by_investor(date_str, date_str, ticker)
                if df is not None and not df.empty:
                    df = df.reset_index()
                    df['date'] = pd.to_datetime(date_str)
                    df['ticker'] = ticker
                    return df
                else:
                    return None
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    print(f"❌ {date_str}, {ticker} 조회 실패: {e}")
                    return None
        return None

    def update_data(self, tickers, start_date, end_date, progress_callback=None):
        """점진적 데이터 업데이트"""
        existing_data, meta = self.load_cached_data()
        missing_dates = self.get_missing_dates(existing_data, start_date, end_date)

        if not missing_dates:
            st.info("📅 모든 데이터가 최신 상태입니다.")
            return existing_data

        st.info(f"📥 {len(missing_dates)}일의 신규 데이터를 수집합니다.")

        new_records = []
        total_requests = len(missing_dates) * len(tickers)
        current_request = 0

        if progress_callback:
            progress_bar = progress_callback.progress(0)
            status_text = progress_callback.empty()

        for date_str in missing_dates:
            for ticker in tickers:
                current_request += 1

                if progress_callback:
                    progress = current_request / total_requests
                    progress_bar.progress(progress)
                    status_text.text(f"수집 중: {date_str}, {ticker} ({current_request}/{total_requests})")

                df = self.fetch_trading_data(date_str, ticker)
                if df is not None:
                    new_records.append(df)

                time.sleep(1.0)

        if new_records:
            new_data = pd.concat(new_records, ignore_index=True)
            if not existing_data.empty:
                updated_data = pd.concat([existing_data, new_data], ignore_index=True)
            else:
                updated_data = new_data

            updated_data = updated_data.drop_duplicates(['date', 'ticker', '투자자구분']).reset_index(drop=True)

            meta.update({
                'last_update': datetime.now(),
                'total_records': len(updated_data),
                'date_range': (updated_data['date'].min(), updated_data['date'].max())
            })

            self.save_data(updated_data, meta)
            return updated_data
        else:
            return existing_data


def load_simple_cache():
    """간단한 캐시 데이터 로드"""
    cache_file = "simple_trading_cache.pkl"
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    return {}


def save_simple_cache(data):
    """간단한 캐시 데이터 저장"""
    cache_file = "simple_trading_cache.pkl"
    with open(cache_file, "wb") as f:
        pickle.dump(data, f)


def get_simple_trading_data(ticker, start_date, end_date):
    """캐시를 활용한 간단한 거래 데이터 수집"""
    if not PYKRX_AVAILABLE:
        st.error("❌ PyKrx 라이브러리가 필요합니다.")
        return pd.DataFrame()

    cache = load_simple_cache()
    cache_key = f"{ticker}_{start_date}_{end_date}"

    if cache_key in cache:
        st.success(f"✅ {ticker} 캐시 데이터 사용")
        return cache[cache_key]

    try:
        st.info(f"📥 {ticker} 데이터 수집 중...")

        # 🔧 일별 데이터를 개별적으로 수집하는 방식으로 변경
        date_range = pd.bdate_range(start_date, end_date)
        daily_data = []

        st.write(f"🔍 **{len(date_range)}일의 데이터를 개별 수집 중...**")

        for i, date in enumerate(date_range):
            date_str = date.strftime("%Y%m%d")
            try:
                # 하루씩 데이터 수집
                daily_result = stock.get_market_trading_value_by_investor(date_str, date_str, ticker)

                if not daily_result.empty:
                    # 데이터가 transpose된 형태이므로 올바르게 처리
                    # Index가 투자자구분, Columns가 거래타입
                    if '순매수' in daily_result.columns:
                        # 순매수 컬럼만 추출하고 transpose
                        day_data = daily_result['순매수'].to_frame().T
                        day_data.index = [date]  # 날짜를 index로 설정
                        daily_data.append(day_data)

                # 진행상황 표시 (매 5일마다)
                if (i + 1) % 5 == 0:
                    st.write(f"  진행: {i + 1}/{len(date_range)} 일 완료")

            except Exception as e:
                st.write(f"  ⚠️ {date_str} 데이터 수집 실패: {e}")
                continue

            # API 제한을 위한 대기
            time.sleep(0.5)

        if daily_data:
            # 모든 일별 데이터를 합치기
            combined_data = pd.concat(daily_data, ignore_index=False)
            combined_data = combined_data.fillna(0)

            st.write("🔍 **최종 데이터 구조:**")
            st.write("- **Index (날짜):**", f"{type(combined_data.index)}")
            st.write("- **날짜 범위:**", f"{combined_data.index.min()} ~ {combined_data.index.max()}")
            st.write("- **컬럼들 (투자자별):**", list(combined_data.columns))
            st.write("- **데이터 형태:**", combined_data.shape)
            st.write("- **첫 번째 행:**")
            st.dataframe(combined_data.head())

            cache[cache_key] = combined_data
            save_simple_cache(cache)
            st.success(f"✅ {ticker} 일별 데이터 수집 완료!")
            return combined_data
        else:
            st.error("❌ 수집된 데이터가 없습니다.")

    except Exception as e:
        st.error(f"❌ {ticker} 데이터 수집 실패: {e}")

        # 🔧 원본 함수 결과 상세 분석
        st.info("🔍 **원본 함수 결과 분석:**")
        try:
            test_data = stock.get_market_trading_value_by_investor(start_date, end_date, ticker)
            st.write("**원본 데이터 형태:**")
            st.write("- **Index:**", test_data.index.tolist())
            st.write("- **Columns:**", test_data.columns.tolist())
            st.write("- **Values:**")
            st.dataframe(test_data)

            # transpose 시도
            st.write("**Transpose 결과:**")
            transposed = test_data.T
            st.dataframe(transposed)

        except Exception as debug_e:
            st.error(f"디버깅 실패: {debug_e}")

    return pd.DataFrame()


def create_investor_trend_chart(data, ticker_name):
    """투자자별 순매수 추이 차트"""
    if data.empty:
        return None

    fig = go.Figure()

    colors = {
        '개인': '#FF6B6B',
        '기관합계': '#4ECDC4',
        '외국인': '#45B7D1',
        '기타법인': '#96CEB4'
    }

    # 데이터 구조 확인: '투자자구분' 컬럼이 있으면 상세 분석, 없으면 간단 분석
    if '투자자구분' in data.columns:
        # 상세 분석 데이터 (pivot 필요)
        valid_investors = ['개인', '기관합계', '외국인']
        filtered_data = data[data['투자자구분'].isin(valid_investors)]

        if filtered_data.empty:
            return None

        # 날짜 컬럼 찾기
        date_col = None
        for col in ['date', '날짜', 'Date', 'DATE']:
            if col in filtered_data.columns:
                date_col = col
                break

        if date_col is None:
            st.error("❌ 상세 분석: 날짜 컬럼을 찾을 수 없습니다.")
            st.write("**사용 가능한 컬럼들:**", list(filtered_data.columns))
            return None

        pivot_data = filtered_data.pivot(index=date_col, columns='투자자구분', values='순매수')
        pivot_data = pivot_data.fillna(0)

        for investor in pivot_data.columns:
            fig.add_trace(go.Scatter(
                x=pivot_data.index,
                y=pivot_data[investor],
                mode='lines+markers',
                name=investor,
                line=dict(color=colors.get(investor, '#999999'), width=2),
                marker=dict(size=6)
            ))
    else:
        # 간단 분석 데이터 (PyKrx에서 반환 - index가 날짜)
        dates = data.index  # index가 날짜

        if is_debugging:
            # 실제 컬럼 확인 및 디버깅
            st.write("🔍 **차트 생성 디버깅:**")
            st.write("- **전체 컬럼:**", list(data.columns))

        # 유효한 투자자 구분 찾기 (더 유연하게)
        valid_investors = ['개인', '기관합계', '외국인', '기타법인']
        available_investors = []

        for col in data.columns:
            if col in valid_investors:
                available_investors.append(col)

        if is_debugging:
            st.write("- **발견된 투자자 구분:**", available_investors)

        if not available_investors:
            st.warning("⚠️ 유효한 투자자 구분을 찾을 수 없습니다.")
            st.write("- **사용 가능한 모든 컬럼:**", list(data.columns))

            # 컬럼명에 투자자 관련 키워드가 있는지 확인
            possible_investors = []
            investor_keywords = ['개인', '기관', '외국', '기타', 'individual', 'institution', 'foreign']
            for col in data.columns:
                for keyword in investor_keywords:
                    if keyword in col:
                        possible_investors.append(col)

            if possible_investors:
                st.info(f"🔍 투자자 관련 키워드가 포함된 컬럼들: {possible_investors}")
                # 이 컬럼들을 사용해서 차트 생성 시도
                for col in possible_investors:
                    if not data[col].isna().all():
                        fig.add_trace(go.Scatter(
                            x=dates,
                            y=data[col],
                            mode='lines+markers',
                            name=col,
                            line=dict(width=2),
                            marker=dict(size=6)
                        ))
            else:
                return None
        else:
            # 차트에 데이터 추가
            for investor in available_investors:
                if investor in data.columns and not data[investor].isna().all():
                    fig.add_trace(go.Scatter(
                        x=dates,
                        y=data[investor],
                        mode='lines+markers',
                        name=investor,
                        line=dict(color=colors.get(investor, '#999999'), width=2),
                        marker=dict(size=6)
                    ))
                    st.write(f"- **{investor} 데이터 추가됨:** {len(data[investor])}개 포인트")

    fig.update_layout(
        title=f"{ticker_name} 투자자별 순매수 추이",
        xaxis_title="날짜",
        yaxis_title="순매수 (원)",
        height=500,
        hovermode='x unified'
    )

    return fig


def create_cumulative_chart(data, ticker_name):
    """누적 순매수 차트"""
    if data.empty:
        return None

    colors = {
        '개인': '#FF6B6B',
        '기관합계': '#4ECDC4',
        '외국인': '#45B7D1',
        '기타법인': '#96CEB4'
    }

    fig = go.Figure()

    # 데이터 구조 확인: '투자자구분' 컬럼이 있으면 상세 분석, 없으면 간단 분석
    if '투자자구분' in data.columns:
        # 상세 분석 데이터 (pivot 필요)
        valid_investors = ['개인', '기관합계', '외국인', '기타법인']
        filtered_data = data[data['투자자구분'].isin(valid_investors)]

        if filtered_data.empty:
            return None

        # 날짜 컬럼 찾기
        date_col = None
        for col in ['date', '날짜', 'Date', 'DATE']:
            if col in filtered_data.columns:
                date_col = col
                break

        if date_col is None:
            st.error("❌ 상세 분석: 날짜 컬럼을 찾을 수 없습니다.")
            return None

        pivot_data = filtered_data.pivot(index=date_col, columns='투자자구분', values='순매수')
        pivot_data = pivot_data.fillna(0)
        cumulative = pivot_data.cumsum()

        for investor in cumulative.columns:
            fig.add_trace(go.Scatter(
                x=cumulative.index,
                y=cumulative[investor],
                mode='lines',
                name=investor,
                line=dict(color=colors.get(investor, '#999999'), width=3),
                fill='tonexty' if investor != cumulative.columns[0] else 'tozeroy'
            ))
    else:
        # 간단 분석 데이터 (PyKrx에서 반환 - index가 날짜)
        dates = data.index  # index가 날짜

        # 유효한 투자자 구분 찾기
        valid_investors = ['개인', '기관합계', '외국인', '기타법인']
        available_investors = [col for col in data.columns if col in valid_investors]

        if not available_investors:
            # 투자자 관련 키워드가 있는 컬럼 찾기
            investor_keywords = ['개인', '기관', '외국', '기타']
            possible_investors = []
            for col in data.columns:
                for keyword in investor_keywords:
                    if keyword in col:
                        possible_investors.append(col)

            if possible_investors:
                available_investors = possible_investors
            else:
                st.warning("⚠️ 누적 차트: 유효한 투자자 구분을 찾을 수 없습니다.")
                return None

        # 선택된 투자자들의 데이터만 추출하고 누적 계산
        investor_data = data[available_investors].fillna(0)
        cumulative = investor_data.cumsum()

        for investor in available_investors:
            fig.add_trace(go.Scatter(
                x=dates,
                y=cumulative[investor],
                mode='lines',
                name=investor,
                line=dict(color=colors.get(investor, '#999999'), width=3),
                fill='tonexty' if investor != available_investors[0] else 'tozeroy'
            ))

    fig.update_layout(
        title=f"{ticker_name} 누적 순매수",
        xaxis_title="날짜",
        yaxis_title="누적 순매수 (원)",
        height=500,
        hovermode='x unified'
    )

    return fig


def basic_stock_analysis():
    """기본 주식 분석 탭"""
    st.header("📈 기본 주식 분석")

    col1, col2 = st.columns(2)

    with col1:
        # 종목 코드 입력
        ticker = st.text_input("종목 코드 입력", value="005930", help="예: 005930 (삼성전자)")

    with col2:
        # 기간 설정
        days = st.selectbox("분석 기간", [7, 14, 30, 60, 90], index=2)

    if st.button("📊 분석 시작", type="primary"):
        if not PYKRX_AVAILABLE:
            st.error("❌ PyKrx 라이브러리가 설치되어 있지 않습니다.")
            return

        try:
            # 종목명 조회
            ticker_name = get_market_ticker_name(ticker)
            if not ticker_name:
                st.error("❌ 유효하지 않은 종목 코드입니다.")
                return

            st.success(f"✅ {ticker_name} ({ticker}) 분석을 시작합니다.")

            # 날짜 설정
            end_date = datetime.now() - timedelta(days=1)
            start_date = end_date - timedelta(days=days)

            # 기본 주가 데이터
            with st.spinner("주가 데이터 수집 중..."):
                price_data = stock.get_market_ohlcv_by_date(
                    start_date.strftime("%Y%m%d"),
                    end_date.strftime("%Y%m%d"),
                    ticker
                )

            if not price_data.empty:
                if is_debugging:
                    # 데이터 구조 확인 및 디버깅
                    st.write("📋 데이터 컬럼 확인:", list(price_data.columns))

                # 주가 차트
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=price_data.index,
                    open=price_data['시가'],
                    high=price_data['고가'],
                    low=price_data['저가'],
                    close=price_data['종가'],
                    name="주가"
                ))

                fig.update_layout(
                    title=f"{ticker_name} ({ticker}) 주가 차트",
                    xaxis_title="날짜",
                    yaxis_title="가격 (원)",
                    height=500
                )

                st.plotly_chart(fig, use_container_width=True)

                # 기본 통계
                current_price = price_data['종가'].iloc[-1]
                prev_price = price_data['종가'].iloc[-2] if len(price_data) > 1 else current_price
                change = current_price - prev_price
                change_pct = (change / prev_price) * 100 if prev_price != 0 else 0

                # 컬럼명 확인하여 안전하게 접근
                volume_col = None
                amount_col = None

                for col in price_data.columns:
                    if '거래량' in col or 'Volume' in col:
                        volume_col = col
                    elif '거래대금' in col or 'Amount' in col or '금액' in col:
                        amount_col = col

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("현재가", f"{current_price:,.0f}원", f"{change:+.0f}원")
                with col2:
                    st.metric("등락률", f"{change_pct:+.2f}%")
                with col3:
                    if volume_col:
                        st.metric("거래량", f"{price_data[volume_col].iloc[-1]:,.0f}주")
                    else:
                        st.metric("거래량", "N/A")
                with col4:
                    if amount_col:
                        st.metric("거래대금", f"{price_data[amount_col].iloc[-1]:,.0f}원")
                    else:
                        # 거래대금이 없으면 거래량 * 종가로 근사치 계산
                        if volume_col:
                            approx_amount = price_data[volume_col].iloc[-1] * current_price
                            st.metric("거래대금(추정)", f"{approx_amount:,.0f}원")
                        else:
                            st.metric("거래대금", "N/A")

                # 상세 데이터 테이블
                with st.expander("📋 상세 데이터 보기"):
                    display_data = price_data.copy()
                    display_data.index = display_data.index.strftime('%Y-%m-%d')

                    # 안전한 포맷팅 (존재하는 컬럼만)
                    format_dict = {}
                    for col in display_data.columns:
                        if col in ['시가', '고가', '저가', '종가']:
                            format_dict[col] = '{:,.0f}'
                        elif '거래량' in col or 'Volume' in col:
                            format_dict[col] = '{:,.0f}'
                        elif '거래대금' in col or 'Amount' in col or '금액' in col:
                            format_dict[col] = '{:,.0f}'

                    st.dataframe(display_data.style.format(format_dict))

        except Exception as e:
            st.error(f"❌ 데이터 수집 중 오류가 발생했습니다: {e}")


def investor_trading_analysis():
    """투자자별 매매동향 분석 탭"""
    st.header("👥 투자자별 매매동향 분석")

    # 분석 모드 선택
    analysis_mode = st.radio("분석 모드 선택", ["간단 분석", "상세 분석"], horizontal=True)

    if analysis_mode == "간단 분석":
        simple_investor_analysis()
    else:
        detailed_investor_analysis()


def simple_investor_analysis():
    """간단한 투자자 분석"""
    col1, col2, col3 = st.columns(3)

    with col1:
        # 종목 선택
        stocks = {
            '삼성전자': '005930',
            'SK하이닉스': '000660',
            'NAVER': '035420',
            '카카오': '035720',
            'LG에너지솔루션': '373220',
            '셀트리온': '068270',
            '현대차': '005380'
        }

        selected_stock = st.selectbox("종목 선택", list(stocks.keys()))
        ticker = stocks[selected_stock]

    with col2:
        # 기간 설정
        days = st.selectbox("분석 기간", [7, 14, 30, 60, 120, 150], index=2)

    with col3:
        # 캐시 관리
        if st.button("🗑️ 캐시 삭제"):
            cache_file = "simple_trading_cache.pkl"
            if os.path.exists(cache_file):
                os.remove(cache_file)
                st.success("캐시가 삭제되었습니다.")

    # 데이터 수집 및 표시
    if st.button("📊 투자자 분석 시작", type="primary"):
        end_date = datetime.now() - timedelta(days=1)
        start_date = end_date - timedelta(days=days)
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")

        with st.spinner("데이터를 가져오는 중..."):
            data = get_simple_trading_data(ticker, start_str, end_str)

        if not data.empty:
            # 📋 데이터 구조 디버깅 (PyKrx 특성 반영)
            with st.expander("🔍 데이터 구조 확인 (디버깅)", expanded=is_debugging):
                st.write("**데이터 형태:** PyKrx의 get_market_trading_value_by_investor 결과")
                st.write("**Index (날짜):**", f"{data.index.name if data.index.name else 'DatetimeIndex'} - {type(data.index)}")
                st.write("**날짜 범위:**", f"{data.index.min()} ~ {data.index.max()}")
                st.write("**컬럼들 (투자자별):**", list(data.columns))

                # 투자자 구분 컬럼 확인
                valid_investors = ['개인', '기관합계', '외국인']
                found_investors = [col for col in data.columns if col in valid_investors]
                st.write("**발견된 투자자 구분:**", found_investors)

                # 기타 컬럼들
                other_cols = [col for col in data.columns if col not in valid_investors]
                st.write("**기타 컬럼들:**", other_cols)

                st.write("**데이터 샘플:**")
                st.dataframe(data.head())

            # 차트 표시
            col1, col2 = st.columns(2)

            with col1:
                chart = create_investor_trend_chart(data, selected_stock)
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
                else:
                    st.warning("⚠️ 추이 차트를 생성할 수 없습니다.")

            with col2:
                cumulative_chart = create_cumulative_chart(data, selected_stock)
                if cumulative_chart:
                    st.plotly_chart(cumulative_chart, use_container_width=True)
                else:
                    st.warning("⚠️ 누적 차트를 생성할 수 없습니다.")

            # 요약 통계 (PyKrx 데이터 - 간단 분석)
            st.subheader("📈 기간 요약")
            col1, col2, col3, col4 = st.columns(4)

            # PyKrx 데이터에서 직접 합계 계산
            with col1:
                individual_sum = data['개인'].sum() if '개인' in data.columns else 0
                st.metric("개인 순매수", f"{individual_sum:,.0f}원")
            with col2:
                institutional_sum = data['기관합계'].sum() if '기관합계' in data.columns else 0
                st.metric("기관 순매수", f"{institutional_sum:,.0f}원")
            with col3:
                foreign_sum = data['외국인'].sum() if '외국인' in data.columns else 0
                st.metric("외국인 순매수", f"{foreign_sum:,.0f}원")
            with col4:
                buy_sum = data['매수'].sum() if '매수' in data.columns else 0
                st.metric("총 거래대금", f"{buy_sum:,.0f}원")

            # 데이터 테이블 (PyKrx 데이터 - index가 날짜)
            with st.expander("📋 상세 데이터 보기"):
                display_data = data.copy()

                # PyKrx 데이터는 index가 날짜이므로 index를 포맷팅
                if hasattr(display_data.index, 'strftime'):
                    display_data.index = display_data.index.strftime('%Y-%m-%d')

                # 안전한 포맷팅 (투자자별 수치 데이터)
                format_dict = {}
                for col in display_data.columns:
                    if any(keyword in col for keyword in ['매도', '매수', '순매수', '금액', '개인', '기관', '외국인', '기타']):
                        format_dict[col] = "{:,.0f}"

                st.dataframe(display_data.style.format(format_dict), use_container_width=True)


def detailed_investor_analysis():
    """상세한 투자자 분석"""
    # 사이드바 설정
    with st.sidebar:
        st.subheader("⚙️ 상세 분석 설정")

        # 날짜 범위 설정
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "시작일",
                value=datetime.now() - timedelta(days=30),
                max_value=datetime.now()
            )
        with col2:
            end_date = st.date_input(
                "종료일",
                value=datetime.now() - timedelta(days=1),
                max_value=datetime.now()
            )

        # 종목 선택
        st.subheader("📈 분석 종목")

        popular_stocks = {
            '삼성전자': '005930',
            'SK하이닉스': '000660',
            'LG에너지솔루션': '373220',
            'NAVER': '035420',
            '카카오': '035720',
            '셀트리온': '068270',
            '현대차': '005380',
            'POSCO홀딩스': '005490'
        }

        selected_stocks = st.multiselect(
            "종목 선택",
            options=list(popular_stocks.keys()),
            default=['삼성전자', 'SK하이닉스']
        )

        if st.button("🔄 데이터 업데이트", type="primary"):
            st.session_state.force_update = True

    if not selected_stocks:
        st.warning("⚠️ 분석할 종목을 선택해주세요.")
        return

    tickers = [popular_stocks[name] for name in selected_stocks]

    # 데이터 관리자 초기화
    data_manager = TradingDataManager()

    # 데이터 로드/업데이트
    if 'detailed_trading_data' not in st.session_state or st.session_state.get('force_update', False):
        with st.spinner("데이터를 수집하고 있습니다..."):
            progress_container = st.container()
            trading_data = data_manager.update_data(
                tickers,
                start_date,
                end_date,
                progress_callback=progress_container
            )
            st.session_state.detailed_trading_data = trading_data
            st.session_state.force_update = False
    else:
        trading_data = st.session_state.detailed_trading_data

    if trading_data.empty:
        st.error("❌ 데이터가 없습니다. 날짜 범위와 종목을 확인해주세요.")
        return

    # 데이터 정보 표시
    with st.expander("📋 데이터 정보", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("총 레코드", f"{len(trading_data):,}")
        with col2:
            st.metric("분석 기간", f"{len(trading_data['date'].unique())}일")
        with col3:
            st.metric("종목 수", len(trading_data['ticker'].unique()))
        with col4:
            st.metric("마지막 업데이트", trading_data['date'].max().strftime('%Y-%m-%d'))

        # 🔍 투자자 구분 확인 (디버깅용)
        st.write("**투자자 구분들:**", sorted(trading_data['투자자구분'].unique().tolist()))

    # 분석 결과 표시
    tab1, tab2, tab3 = st.tabs(["📊 개별 종목", "📋 종합 요약", "📈 비교 분석"])

    with tab1:
        selected_ticker_name = st.selectbox("분석할 종목 선택", options=selected_stocks)
        selected_ticker = popular_stocks[selected_ticker_name]

        ticker_data = trading_data[trading_data['ticker'] == selected_ticker]

        if not ticker_data.empty:
            # 유효한 투자자 구분만 필터링
            valid_investors = ['개인', '기관합계', '외국인']
            filtered_data = ticker_data[ticker_data['투자자구분'].isin(valid_investors)]

            if filtered_data.empty:
                st.warning("⚠️ 유효한 투자자 구분 데이터가 없습니다.")
                return

            pivot = filtered_data.pivot(index='date', columns='투자자구분', values='순매수')
            pivot = pivot.fillna(0)

            col1, col2 = st.columns(2)

            with col1:
                # 일별 추이
                fig1 = go.Figure()
                colors = {'개인': '#FF6B6B', '기관합계': '#4ECDC4', '외국인': '#45B7D1', '기타법인': '#96CEB4'}

                for investor in pivot.columns:
                    if investor in valid_investors:  # 추가 안전장치
                        fig1.add_trace(go.Scatter(
                            x=pivot.index,
                            y=pivot[investor],
                            mode='lines+markers',
                            name=investor,
                            line=dict(color=colors.get(investor, '#999999'), width=2)
                        ))

                fig1.update_layout(
                    title=f"{selected_ticker_name} 투자자별 순매수 추이",
                    xaxis_title="날짜",
                    yaxis_title="순매수 (원)",
                    height=400
                )

                st.plotly_chart(fig1, use_container_width=True)

            with col2:
                # 누적 추이
                cumulative = pivot.cumsum()
                fig2 = go.Figure()

                for investor in cumulative.columns:
                    if investor in valid_investors:  # 추가 안전장치
                        fig2.add_trace(go.Scatter(
                            x=cumulative.index,
                            y=cumulative[investor],
                            mode='lines',
                            name=investor,
                            line=dict(color=colors.get(investor, '#999999'), width=3)
                        ))

                fig2.update_layout(
                    title=f"{selected_ticker_name} 누적 순매수",
                    xaxis_title="날짜",
                    yaxis_title="누적 순매수 (원)",
                    height=400
                )

                st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        st.subheader("📋 기간별 투자자 순매수 요약")

        # 유효한 투자자 구분만 필터링
        valid_investors = ['개인', '기관합계', '외국인']
        filtered_summary_data = trading_data[trading_data['투자자구분'].isin(valid_investors)]

        if not filtered_summary_data.empty:
            summary = filtered_summary_data.groupby(['ticker', '투자자구분'])['순매수'].sum().reset_index()
            summary_pivot = summary.pivot(index='ticker', columns='투자자구분', values='순매수')
            summary_pivot = summary_pivot.fillna(0)

            # 종목명 추가
            ticker_names = {v: k for k, v in popular_stocks.items()}
            summary_pivot['종목명'] = summary_pivot.index.map(lambda x: ticker_names.get(x, x))
            summary_pivot = summary_pivot[['종목명'] + [col for col in summary_pivot.columns if col != '종목명']]

            st.dataframe(
                summary_pivot.style.format({col: "{:,.0f}" for col in summary_pivot.columns if col != '종목명'}),
                use_container_width=True
            )
        else:
            st.warning("⚠️ 유효한 투자자 구분 데이터가 없습니다.")

    with tab3:
        st.subheader("📈 종목별 외국인 순매수 비교")

        # 유효한 외국인 데이터만 필터링
        foreign_data = trading_data[
            (trading_data['투자자구분'] == '외국인') |
            (trading_data['투자자구분'] == '외국인투자자')
        ].copy()

        if not foreign_data.empty:
            fig = go.Figure()

            for ticker in tickers:
                ticker_data = foreign_data[foreign_data['ticker'] == ticker]
                if not ticker_data.empty:
                    ticker_name = [k for k, v in popular_stocks.items() if v == ticker][0]
                    fig.add_trace(go.Scatter(
                        x=ticker_data['date'],
                        y=ticker_data['순매수'],
                        mode='lines+markers',
                        name=ticker_name,
                        line=dict(width=2)
                    ))

            fig.update_layout(
                title="외국인 투자자 순매수 비교",
                xaxis_title="날짜",
                yaxis_title="순매수 (원)",
                height=500,
                hovermode='x unified'
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ 외국인 투자자 데이터가 없습니다.")


def display_valuation_results(company_info, dcf_result, srim_result, metrics, ticker):
    """밸류에이션 결과 표시"""

    # 현재 주가 조회
    current_price = get_current_stock_price(ticker)

    # 요약 결과
    st.header("📊 적정주가 분석 결과")

    # 메인 결과 카드
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        dcf_per_share = dcf_result['enterprise_value'] * 100 / 594  # 임시 주식수 (삼성전자)
        st.metric(
            "DCF 적정주가",
            f"{dcf_per_share:,.0f}원",
            f"{((dcf_per_share - current_price) / current_price * 100):+.1f}%" if current_price else None
        )

    with col2:
        srim_price = srim_result['intrinsic_value']
        st.metric(
            "S-RIM 적정주가",
            f"{srim_price:,.0f}원",
            f"{((srim_price - current_price) / current_price * 100):+.1f}%" if current_price else None
        )

    with col3:
        avg_price = (dcf_per_share + srim_price) / 2
        st.metric(
            "평균 적정주가",
            f"{avg_price:,.0f}원",
            f"{((avg_price - current_price) / current_price * 100):+.1f}%" if current_price else None
        )

    with col4:
        if current_price:
            st.metric(
                "현재 주가",
                f"{current_price:,.0f}원",
                "기준가격"
            )

    # 상세 분석 탭
    tab1, tab2, tab3, tab4 = st.tabs(["📈 DCF 분석", "💎 S-RIM 분석", "📊 재무 지표", "📋 상세 데이터"])

    with tab1:
        display_dcf_analysis(dcf_result)

    with tab2:
        display_srim_analysis(srim_result)

    with tab3:
        display_financial_metrics(metrics)

    with tab4:
        display_detailed_data(company_info, dcf_result, srim_result)


def display_dcf_analysis(dcf_result):
    """DCF 분석 결과 표시"""
    st.subheader("🔮 DCF 분석 상세")

    # 주요 가정
    col1, col2 = st.columns(2)

    with col1:
        st.info("**주요 가정**")
        st.write(f"• WACC: {dcf_result['wacc']:.2%}")
        st.write(f"• 매출 성장률: {dcf_result['growth_rate']:.2%}")
        st.write(f"• 영업이익률: {dcf_result['operating_margin']:.2%}")
        st.write(f"• 영구성장률: {dcf_result['assumptions']['terminal_growth']:.2%}")

    with col2:
        st.success("**기업가치**")
        st.write(f"• 기업가치: {dcf_result['enterprise_value']:,.0f}억원")
        st.write(f"• 터미널가치: {dcf_result['terminal_value']:,.0f}억원")
        st.write(f"• 예측기간: {dcf_result['assumptions']['forecast_years']}년")

    # FCF 예측 차트
    if dcf_result['fcf_projections']:
        fcf_data = pd.DataFrame(dcf_result['fcf_projections'])

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=fcf_data['year'],
            y=fcf_data['fcf'],
            name='예상 FCF',
            marker_color='lightblue'
        ))

        fig.update_layout(
            title="자유현금흐름 예측",
            xaxis_title="연도",
            yaxis_title="FCF (억원)",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)


def display_srim_analysis(srim_result):
    """S-RIM 분석 결과 표시"""
    st.subheader("💎 S-RIM 분석 상세")

    col1, col2 = st.columns(2)

    with col1:
        st.info("**ROE 분석**")
        st.write(f"• 지속가능 ROE: {srim_result['sustainable_roe']:.2%}")
        st.write(f"• 요구수익률: {srim_result['required_return']:.2%}")
        st.write(f"• 초과수익률: {srim_result['excess_roe']:.2%}")
        st.write(f"• 성장률: {srim_result['growth_rate']:.2%}")

    with col2:
        st.success("**밸류에이션**")
        st.write(f"• 현재 BPS: {srim_result['current_bps']:,.0f}원")
        st.write(f"• 내재가치: {srim_result['intrinsic_value']:,.0f}원")

    # ROE 트렌드 차트
    if srim_result['roe_components']:
        roe_df = pd.DataFrame(srim_result['roe_components']).T

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=roe_df.index,
            y=roe_df['roe'] * 100,
            mode='lines+markers',
            name='ROE',
            line=dict(color='green', width=3)
        ))

        fig.update_layout(
            title="ROE 추이",
            xaxis_title="연도",
            yaxis_title="ROE (%)",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)


def display_financial_metrics(metrics):
    """재무 지표 표시"""
    st.subheader("📊 주요 재무 지표")

    # 데이터프레임 생성
    df = pd.DataFrame(metrics).T
    df = df.round(0)

    # 지표별 차트
    col1, col2 = st.columns(2)

    with col1:
        # 매출액 추이
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['revenue'],
            mode='lines+markers',
            name='매출액',
            line=dict(color='blue', width=3)
        ))

        fig.update_layout(
            title="매출액 추이",
            xaxis_title="연도",
            yaxis_title="매출액 (억원)",
            height=300
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # 영업이익 추이
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['ebit'],
            mode='lines+markers',
            name='영업이익',
            line=dict(color='green', width=3)
        ))

        fig.update_layout(
            title="영업이익 추이",
            xaxis_title="연도",
            yaxis_title="영업이익 (억원)",
            height=300
        )

        st.plotly_chart(fig, use_container_width=True)

    # 재무 데이터 테이블
    st.subheader("📋 재무 데이터 상세")
    st.dataframe(
        df.style.format("{:,.0f}"),
        use_container_width=True
    )


def get_current_stock_price(ticker):
    """현재 주가 조회 (PyKrx 활용)"""
    try:
        if PYKRX_AVAILABLE:
            today = datetime.now().strftime("%Y%m%d")
            yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")

            price_data = stock.get_market_ohlcv_by_date(yesterday, today, ticker)
            if not price_data.empty:
                return price_data['종가'].iloc[-1]
    except:
        pass
    return None


def display_detailed_data(company_info, dcf_result, srim_result):
    """상세 데이터 표시"""
    st.subheader("📋 분석 상세 정보")

    # 회사 정보
    st.info("**회사 정보**")
    st.json(company_info)

    # DCF 상세
    with st.expander("DCF 상세 데이터"):
        st.json(dcf_result)

    # S-RIM 상세
    with st.expander("S-RIM 상세 데이터"):
        st.json(srim_result)


# 3. 적정주가 분석 탭 함수
def valuation_analysis():
    """적정주가 분석 탭"""
    st.header("💰 적정주가 분석 (DCF + S-RIM)")

    if not DART_AVAILABLE:
        st.error("❌ OpenDartReader 라이브러리가 설치되지 않았습니다.")
        st.info("터미널에서 다음 명령어로 설치해주세요: `pip install opendartreader`")
        return

    # DART API 키 입력
    api_key = dart_key
    # with st.expander("🔑 DART API 설정", expanded=True):
        # api_key = st.text_input(
        #     "DART API 키",
        #     type="password",
        #     help="DART 홈페이지(https://opendart.fss.or.kr)에서 발급받으세요"
        # )
        #
        # if not api_key:
        #     st.warning("⚠️ DART API 키를 입력해주세요.")
        #     st.info("💡 DART API 키 발급: https://opendart.fss.or.kr > 인증키 신청")
        #     return

    # 분석 설정
    col1, col2, col3 = st.columns(3)

    with col1:
        ticker = st.text_input("종목 코드", value="005930", help="예: 005930 (삼성전자)")

    with col2:
        beta = st.number_input("베타", value=1.0, min_value=0.1, max_value=3.0, step=0.1)

    with col3:
        analysis_years = st.selectbox("분석 기간", [3, 5, 7], index=1)

    if st.button("📊 적정주가 분석 시작", type="primary"):

        with st.spinner("데이터 수집 및 분석 중..."):
            # 데이터 수집기 초기화
            dart_collector = DartDataCollector(api_key)

            # 1. 회사 정보 조회
            st.info("🔍 회사 정보 조회 중...")
            company_info = dart_collector.get_company_info(ticker)

            if not company_info:
                st.error("❌ 종목 코드에 해당하는 회사를 찾을 수 없습니다.")
                return

            st.success(f"✅ {company_info['corp_name']} 정보 조회 완료")

            # 2. 재무제표 수집
            st.info("📋 재무제표 수집 중...")
            financial_data = dart_collector.get_financial_statements(
                company_info['corp_code'],
                years=analysis_years
            )

            if not financial_data:
                st.error("❌ 재무제표 데이터를 수집할 수 없습니다.")
                return

            # 3. DCF 분석
            st.info("💹 DCF 분석 중...")
            dcf_model = DCFModel()

            # 재무 지표 추출
            metrics = dcf_model.extract_financial_metrics(financial_data)

            if not metrics:
                st.error("❌ 재무 지표를 추출할 수 없습니다.")
                return

            # DCF 계산
            dcf_result = dcf_model.calculate_dcf(metrics, beta)

            # 4. S-RIM 분석
            st.info("📈 S-RIM 분석 중...")
            srim_model = SRIMModel()

            # ROE 분해
            roe_data = srim_model.calculate_roe_components(metrics)

            # S-RIM 계산
            srim_result = srim_model.calculate_srim(metrics, roe_data, beta)

        # 결과 표시
        if dcf_result and srim_result:
            display_valuation_results(
                company_info,
                dcf_result,
                srim_result,
                metrics,
                ticker
            )
        else:
            st.error("❌ 밸류에이션 계산에 실패했습니다.")


def run():
    """메인 실행 함수"""
    # 🔧 st.set_page_config 제거 (main.py에서 설정하므로)
    st.title("📊 주식 분석 대시보드")

    if not PYKRX_AVAILABLE:
        st.error("❌ PyKrx 라이브러리가 설치되지 않았습니다.")
        st.info("터미널에서 다음 명령어로 설치해주세요: `pip install pykrx`")
        return

    # 탭 생성
    tab1, tab2, tab3 = st.tabs(["📈 기본 주식 분석", "👥 투자자별 매매동향(간략)", "💰 적정주가 분석"])

    with tab1:
        basic_stock_analysis()

    with tab2:
        investor_trading_analysis()

    with tab3:
        valuation_analysis()

    # 사용법 안내
    with st.expander("💡 사용법 안내"):
        st.markdown("""
        ### 📖 기본 주식 분석
        - 종목 코드를 입력하여 주가 차트와 기본 정보 확인
        - 캔들스틱 차트로 가격 추이 분석
        - 거래량, 거래대금 등 주요 지표 표시
        
        ### 👥 투자자별 매매동향 분석
        **간단 분석**
        - 빠른 단일 종목 분석
        - 캐시 활용으로 즉시 실행
        
        **상세 분석**  
        - 다중 종목 비교 분석
        - 장기간 데이터 수집 및 저장
        - 투자자별 상세 통계
        
        ### ⚡ 팁
        - 첫 실행 시 데이터 수집에 시간이 걸릴 수 있습니다
        - 캐시를 활용하여 재실행 시 빠른 로딩
        - 영업일 기준으로 데이터 제공
        """)


if __name__ == "__main__":
    run()
