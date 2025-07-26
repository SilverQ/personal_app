"""
apps/stock_analysis_app.py
주식 분석 통합 애플리케이션
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
import warnings
warnings.filterwarnings('ignore')

# PyKrx 임포트 (에러 처리 포함)
try:
    from pykrx import stock
    from pykrx.stock import get_market_trading_value_by_investor, get_market_ticker_list, get_market_ticker_name
    PYKRX_AVAILABLE = True
except ImportError:
    PYKRX_AVAILABLE = False
    st.error("❌ PyKrx 라이브러리가 설치되지 않았습니다. 'pip install pykrx'로 설치해주세요.")

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
        data = stock.get_market_trading_value_by_investor(start_date, end_date, ticker)

        if not data.empty:
            data = data.reset_index()
            cache[cache_key] = data
            save_simple_cache(cache)
            st.success(f"✅ {ticker} 데이터 수집 완료")
            return data
    except Exception as e:
        st.error(f"❌ {ticker} 데이터 수집 실패: {e}")

    return pd.DataFrame()

def create_investor_trend_chart(data, ticker_name):
    """투자자별 순매수 추이 차트"""
    if data.empty:
        return None

    fig = go.Figure()

    colors = {
        '개인': '#FF6B6B',
        '기관계': '#4ECDC4',
        '외국인': '#45B7D1',
        '기타법인': '#96CEB4'
    }

    for investor in data.columns[1:]:
        if investor in ['매도', '매수']:
            continue
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data[investor],
            mode='lines+markers',
            name=investor,
            line=dict(color=colors.get(investor, '#999999'), width=2),
            marker=dict(size=6)
        ))

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

    # 순매수 컬럼만 선택
    columns_to_plot = [col for col in data.columns if col not in ['매도', '매수'] and col != data.index.name]

    fig = go.Figure()

    colors = {
        '개인': '#FF6B6B',
        '기관계': '#4ECDC4',
        '외국인': '#45B7D1',
        '기타법인': '#96CEB4'
    }

    for investor in columns_to_plot:
        cumulative = data[investor].cumsum()
        fig.add_trace(go.Scatter(
            x=data.index,
            y=cumulative,
            mode='lines',
            name=investor,
            line=dict(color=colors.get(investor, '#999999'), width=3),
            fill='tonexty' if investor != columns_to_plot[0] else 'tozeroy'
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
        days = st.selectbox("분석 기간", [7, 14, 30, 60], index=2)

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
            # 차트 표시
            col1, col2 = st.columns(2)

            with col1:
                chart = create_investor_trend_chart(data, selected_stock)
                if chart:
                    st.plotly_chart(chart, use_container_width=True)

            with col2:
                cumulative_chart = create_cumulative_chart(data, selected_stock)
                if cumulative_chart:
                    st.plotly_chart(cumulative_chart, use_container_width=True)

            # 요약 통계
            st.subheader("📈 기간 요약")
            col1, col2, col3, col4 = st.columns(4)

            # 안전한 컬럼 접근
            buy_col = None
            for col in data.columns:
                if '매수' in col and '순매수' not in col:
                    buy_col = col
                    break

            with col1:
                if '개인' in data.columns:
                    st.metric("개인 순매수", f"{data['개인'].sum():,.0f}원")
                else:
                    st.metric("개인 순매수", "N/A")
            with col2:
                if '기관계' in data.columns:
                    st.metric("기관 순매수", f"{data['기관계'].sum():,.0f}원")
                else:
                    st.metric("기관 순매수", "N/A")
            with col3:
                if '외국인' in data.columns:
                    st.metric("외국인 순매수", f"{data['외국인'].sum():,.0f}원")
                else:
                    st.metric("외국인 순매수", "N/A")
            with col4:
                if buy_col:
                    st.metric("총 거래대금", f"{data[buy_col].sum():,.0f}원")
                else:
                    st.metric("총 거래대금", "N/A")

            # 데이터 테이블
            with st.expander("📋 상세 데이터 보기"):
                display_data = data.copy()
                if hasattr(display_data.index, 'strftime'):
                    display_data.index = display_data.index.strftime('%Y-%m-%d')

                # 안전한 포맷팅
                format_dict = {}
                for col in display_data.columns:
                    if any(keyword in col for keyword in ['매도', '매수', '순매수', '금액']):
                        format_dict[col] = "{:,.0f}"

                st.dataframe(display_data.style.format(format_dict))

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

    # 분석 결과 표시
    tab1, tab2, tab3 = st.tabs(["📊 개별 종목", "📋 종합 요약", "📈 비교 분석"])

    with tab1:
        selected_ticker_name = st.selectbox("분석할 종목 선택", options=selected_stocks)
        selected_ticker = popular_stocks[selected_ticker_name]

        ticker_data = trading_data[trading_data['ticker'] == selected_ticker]

        if not ticker_data.empty:
            pivot = ticker_data.pivot(index='date', columns='투자자구분', values='순매수')
            pivot = pivot.fillna(0)

            col1, col2 = st.columns(2)

            with col1:
                # 일별 추이
                fig1 = go.Figure()
                colors = {'개인': '#FF6B6B', '기관계': '#4ECDC4', '외국인': '#45B7D1', '기타법인': '#96CEB4'}

                for investor in pivot.columns:
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

        summary = trading_data.groupby(['ticker', '투자자구분'])['순매수'].sum().reset_index()
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

    with tab3:
        st.subheader("📈 종목별 외국인 순매수 비교")

        foreign_data = trading_data[trading_data['투자자구분'] == '외국인'].copy()
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

def run():
    """메인 실행 함수"""
    st.title("📊 주식 분석 대시보드")

    if not PYKRX_AVAILABLE:
        st.error("❌ PyKrx 라이브러리가 설치되지 않았습니다.")
        st.info("터미널에서 다음 명령어로 설치해주세요: `pip install pykrx`")
        return

    # 탭 생성
    tab1, tab2 = st.tabs(["📈 기본 주식 분석", "👥 투자자별 매매동향"])

    with tab1:
        basic_stock_analysis()

    with tab2:
        investor_trading_analysis()

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
