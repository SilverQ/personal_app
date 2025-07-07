# app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pykrx import stock
from pykrx import bond
from pykrx.stock import get_market_trading_value_by_investor, get_exhaustion_rates_of_foreign_investment
import plotly.express as px
import io
import logging  # 추가

# urllib3의 디버그 로그 비활성화
logging.getLogger("urllib3").setLevel(logging.WARNING)


# 앱 실행 함수
def run():
    st.title("📊 투자자별 매매동향 대시보드 (PyKrx 기반)")

    # st.set_page_config(page_title="투자자별 매매동향 대시보드", layout="wide")
    col_ctrl, col_vis = st.columns([1, 3])  # 1:3 비율로 좌우 분할
    with col_ctrl:
        # 1) 날짜 선택
        today = datetime.today()
        default_date = today - timedelta(days=1)
        date = st.date_input("조회 기준일", value=default_date, max_value=today)

        # 2) 외국인 소진율 범위 슬라이더
        ownership_min, ownership_max = st.slider(
            "외국인 소진율 범위 (%)",
            0, 100, (10, 50)
        )

        # 3) 외국인 최근 순매수 종목만 필터
        filter_foreign_buy = st.checkbox("외국인 최근 순매수 종목만", value=False)

        # 4) 개인 최근 순매도 종목만 필터
        filter_individual_sell = st.checkbox("개인 최근 순매도 종목만", value=False)

        # 5) 조회 버튼
        if st.button("데이터 조회"):
            with col_ctrl:
                with st.spinner("데이터 조회 중..."):

                    # 투자자별 매매동향 가져오기
                    str_date = date.strftime("%Y%m%d")
                    df_trading = get_market_trading_value_by_investor(
                        str_date, str_date, "ALL"
                    ).reset_index()
                    # st.write(df_trading.columns.tolist())  # ['index','개인','외국인','기관'] 같은 항목이 나올 겁니다.
                    # [0: "투자자구분", 1: "매도", 2: "매수", 3: "순매수"]
                    df_trading.columns = [
                        "종목코드", "종목명", "개인", "기관", "외국인", "기타"
                    ]

                    # 외국인 지분율 데이터 가져오기
                    df_ownership = stock.get_stock_foreign_ownership(str_date).reset_index()
                    df_ownership.columns = ["종목코드", "종목명", "상장주식수", "외국인지분한도", "외국인보유수량", "외국인소진율"]

                    # 병합
                    df_merged = pd.merge(df_trading, df_ownership[["종목코드", "외국인소진율"]], on="종목코드", how="left")

                    # 외국인 소진율 범위 필터
                    df_filtered = df_merged[
                        (df_merged["외국인소진율"] >= ownership_min) &
                        (df_merged["외국인소진율"] <= ownership_max)
                    ]

                    # 외국인 순매수 필터
                    if filter_foreign_buy:
                        df_filtered = df_filtered[df_filtered["외국인"] > 0]

                    # 개인 순매도 필터
                    if filter_individual_sell:
                        df_filtered = df_filtered[df_filtered["개인"] < 0]

                    if df_filtered.empty:
                        st.warning("조건을 만족하는 종목이 없습니다.")
                    else:
                        # 시각화: 외국인 순매수금액 상위 20종목 기준
                        df_vis = df_filtered.sort_values(by="외국인", ascending=False).head(20)

                        fig = px.bar(
                            df_vis,
                            x="종목명",
                            y="외국인",
                            color="외국인소진율",
                            color_continuous_scale="Blues",
                            title="상위 20종목 외국인 순매수 & 소진율"
                        )
                        fig.update_layout(xaxis_title="종목명", yaxis_title="외국인 순매수금액")

                        st.plotly_chart(fig, use_container_width=True)

                        st.subheader("📋 필터링된 데이터")
                        st.dataframe(df_filtered, use_container_width=True)

                        # 데이터 다운로드
                        csv_data = df_filtered.to_csv(index=False).encode("utf-8-sig")
                        st.download_button(
                            label="📥 CSV 다운로드",
                            data=csv_data,
                            file_name=f"filtered_stocks_{str_date}.csv",
                            mime="text/csv",
                        )

                        # 시각화 이미지 다운로드
                        img_bytes = fig.to_image(format="png")
                        st.download_button(
                            label="🖼️ 그래프 이미지 다운로드",
                            data=img_bytes,
                            file_name=f"chart_{str_date}.png",
                            mime="image/png",
                        )
