import os
import time
import numpy as np
import pandas as pd
import requests as re
import streamlit as st
from apps.currency_dict import *
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


# 앱 실행 함수
def run():
    st.title("환율 데이터 수집 및 차트")

    # 크롤링 함수
    def market_index_crawling(key, start_date=None, pages=500):
        date = []
        value = []
        for i in range(1, pages + 1):
            url = re.get(hana_million_nat_url_dict[key] + '&page=%s' % i)
            html = BeautifulSoup(url.content, 'html.parser')
            tbody = html.find('tbody').find_all('tr')

            for tr in tbody:
                temp_date = tr.find('td', {'class': 'date'}).text.replace('.', '-').strip()
                temp_value = float(tr.find('td', {'class': 'num'}).text.strip().replace(',', ''))

                # 날짜 형식을 통일하기 위해 temp_date를 Timestamp 형식으로 변환
                temp_date = pd.to_datetime(temp_date)

                if start_date and temp_date <= start_date:
                    return pd.DataFrame(value, index=date, columns=[key.split()[1]])

                date.append(temp_date)
                value.append(temp_value)

        data = pd.DataFrame(value, index=date, columns=[key.split()[1]])
        return data

    # 데이터 수집 로직 (전체 또는 최신 데이터만 수집)
    def collect_currency_data():
        data_file_path = os.path.join(os.path.dirname(__file__), 'naver_currency.csv')

        # 데이터 파일이 존재할 경우: 최신 데이터만 가져오기
        if os.path.exists(data_file_path):
            try:
                currency_rate = pd.read_csv(data_file_path, index_col='date')
                currency_rate.index = pd.to_datetime(currency_rate.index, format="%Y-%m-%d")
                st.success("데이터 파일을 성공적으로 불러왔습니다.")
                st.dataframe(currency_rate.head())  # 데이터의 일부분을 화면에 표시
                last_collected_date = currency_rate.index.max()

                # 모든 키에 대해 데이터를 먼저 다운로드
                new_data = pd.DataFrame()
                for key in hana_million_nat_url_dict.keys():
                    tmp = market_index_crawling(key, start_date=last_collected_date)
                    if not tmp.empty:
                        if new_data.empty:
                            new_data = tmp
                        else:
                            new_data = pd.merge(new_data, tmp, left_index=True, right_index=True, how='outer')

                # 새로운 데이터가 있으면 기존 데이터와 연결
                if not new_data.empty:
                    currency_rate = pd.concat([new_data, currency_rate]).sort_index().drop_duplicates()
                    st.write(f"최신 데이터를 추가로 수집했습니다. (마지막 수집일: {new_data.index.max()})")
                else:
                    st.write("새로운 데이터가 없습니다.")

                currency_rate.to_csv(data_file_path, index_label='date')
                st.write(f"최신 데이터를 추가로 수집했습니다. (마지막 수집일: {last_collected_date})")

                return currency_rate
            except Exception as e:
                st.error(f"데이터 읽기 실패: {e}")
                return None

        else:
            st.warning("데이터 파일이 존재하지 않습니다. 새로 데이터를 수집 중입니다...")
            currency_rate = pd.DataFrame()
            for i, key in enumerate(hana_million_nat_url_dict.keys()):
                tmp = market_index_crawling(key)

                if currency_rate.empty:
                    currency_rate = tmp
                else:
                    currency_rate = pd.merge(currency_rate, tmp, left_index=True, right_index=True, how='inner')

            currency_rate.to_csv(data_file_path, index_label='date')
            st.write("모든 데이터를 새로 수집했습니다.")

        return currency_rate

    # 차트를 저장하고 표시하는 함수
    def generate_and_save_chart(df, file_name):
        fig, axs = plt.subplots(len(df.columns) // 4 + 1, 4, figsize=(20, 15))
        for i, col in enumerate(df.columns):
            axs[i // 4, i % 4].plot(df.index, df[col])
            axs[i // 4, i % 4].set_title(col)
            start, end = axs[i // 4, i % 4].get_xlim()
            axs[i // 4, i % 4].xaxis.set_ticks(np.linspace(start, end, 6))
        plt.tight_layout()
        plt.savefig(file_name)
        return file_name

    # 선택한 기간만큼 데이터를 필터링하는 함수
    def filter_data_by_period(currency_rate, period):
        today = datetime.today()

        if period == '3개월':
            start_date = today - timedelta(days=90)
        elif period == '1년':
            start_date = today - timedelta(days=365)
        elif period == '3년':
            start_date = today - timedelta(days=3 * 365)
        else:
            start_date = pd.to_datetime(currency_rate.index.min())

        filtered_data = currency_rate.loc[currency_rate.index >= str(start_date.date())]
        return filtered_data

    # 데이터가 이미 수집되었는지 확인
    if 'currency_rate' not in st.session_state:
        st.session_state['currency_rate'] = None

    # 데이터 수집 버튼
    if st.button("데이터 수집"):
        with st.spinner("데이터를 수집 중입니다... 잠시만 기다려주세요."):
            currency_rate = collect_currency_data()
            st.session_state['currency_rate'] = currency_rate
            st.success("데이터 수집이 완료되었습니다!")

    # 데이터가 수집된 경우에만 기간 선택과 차트 생성 표시
    if st.session_state['currency_rate'] is not None:
        currency_rate = st.session_state['currency_rate']

        # 기간 선택 옵션 추가
        period_options = ["3개월", "1년", "3년", "전체"]
        selected_period = st.selectbox("기간을 선택하세요:", period_options)

        # 선택한 기간만큼 데이터 필터링
        filtered_data = filter_data_by_period(currency_rate, selected_period)

        # 차트 생성 및 저장
        st.write(f"{selected_period} 데이터를 기준으로 차트를 생성 중입니다...")
        chart_file = generate_and_save_chart(filtered_data, f"currency_{selected_period}.png")

        # 사용자가 클릭 시 차트 표시
        if st.button(f'{selected_period} 차트 보기'):
            st.image(chart_file, caption=f"{selected_period} 환율 차트")
    else:
        st.info("먼저 데이터를 수집하세요.")
