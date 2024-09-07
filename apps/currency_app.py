import os
import time
import numpy as np
import pandas as pd
import requests as re
import streamlit as st

from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 환율 정보 URL 매핑
hana_million_nat_url_dict = {
    '미국 USD': 'http://finance.naver.com/marketindex/exchangeDailyQuote.nhn?marketindexCd=FX_USDKRW',
    '유럽 EUR': 'http://finance.naver.com/marketindex/exchangeDailyQuote.nhn?marketindexCd=FX_EURKRW',
    '일본 JPY': 'http://finance.naver.com/marketindex/exchangeDailyQuote.nhn?marketindexCd=FX_JPYKRW',
    '중국 CNY': 'http://finance.naver.com/marketindex/exchangeDailyQuote.nhn?marketindexCd=FX_CNYKRW',
    '영국 GBP': 'https://finance.naver.com/marketindex/exchangeDailyQuote.nhn?marketindexCd=FX_GBPKRW',
    '캐나다 CAD': 'https://finance.naver.com/marketindex/exchangeDailyQuote.nhn?marketindexCd=FX_CADKRW',
    '스위스 CHF': 'https://finance.naver.com/marketindex/exchangeDailyQuote.nhn?marketindexCd=FX_CHFKRW',
    '홍콩 HKD': 'https://finance.naver.com/marketindex/exchangeDailyQuote.nhn?marketindexCd=FX_HKDKRW',
    '스웨덴 SEK': 'https://finance.naver.com/marketindex/exchangeDailyQuote.nhn?marketindexCd=FX_SEKKRW',
    '호주 AUD': 'https://finance.naver.com/marketindex/exchangeDailyQuote.nhn?marketindexCd=FX_AUDKRW',
    'UAE AED': 'https://finance.naver.com/marketindex/exchangeDailyQuote.nhn?marketindexCd=FX_AEDKRW',
    '바레인 BHD': 'https://finance.naver.com/marketindex/exchangeDailyQuote.nhn?marketindexCd=FX_BHDKRW',
    '체코 CZK': 'https://finance.naver.com/marketindex/exchangeDailyQuote.nhn?marketindexCd=FX_CZKKRW',
    '덴마크 DKK': 'https://finance.naver.com/marketindex/exchangeDailyQuote.nhn?marketindexCd=FX_DKKKRW',
    '헝가리 HUF': 'https://finance.naver.com/marketindex/exchangeDailyQuote.nhn?marketindexCd=FX_HUFKRW',
    '인도네시아 IDR': 'https://finance.naver.com/marketindex/exchangeDailyQuote.nhn?marketindexCd=FX_IDRKRW',
    '쿠웨이트 KWD': 'https://finance.naver.com/marketindex/exchangeDailyQuote.nhn?marketindexCd=FX_KWDKRW',
    '멕시코 MXN': 'https://finance.naver.com/marketindex/exchangeDailyQuote.nhn?marketindexCd=FX_MXNKRW',
    '노르웨이 NOK': 'https://finance.naver.com/marketindex/exchangeDailyQuote.nhn?marketindexCd=FX_NOKKRW',
    '뉴질랜드 NZD': 'https://finance.naver.com/marketindex/exchangeDailyQuote.nhn?marketindexCd=FX_NZDKRW',
    '폴란드 PLN': 'https://finance.naver.com/marketindex/exchangeDailyQuote.nhn?marketindexCd=FX_PLNKRW',
    '러시아 RUB': 'https://finance.naver.com/marketindex/exchangeDailyQuote.nhn?marketindexCd=FX_RUBKRW',
    '사우디 SAR': 'https://finance.naver.com/marketindex/exchangeDailyQuote.nhn?marketindexCd=FX_SARKRW',
    '싱가포르 SGD': 'https://finance.naver.com/marketindex/exchangeDailyQuote.nhn?marketindexCd=FX_SGDKRW',
    '태국 THB': 'https://finance.naver.com/marketindex/exchangeDailyQuote.nhn?marketindexCd=FX_THBKRW',
    '튀르키에 TRY': 'https://finance.naver.com/marketindex/exchangeDailyQuote.nhn?marketindexCd=FX_TRYKRW',
    '남아공 ZAR': 'https://finance.naver.com/marketindex/exchangeDailyQuote.nhn?marketindexCd=FX_ZARKRW'
}


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


# 데이터 수집 로직 자동 실행 (전체 또는 최신 데이터만 수집)
def collect_currency_data():
    data_file_path = 'data/naver_currency.csv'

    # 데이터 파일이 존재할 경우: 최신 데이터만 가져오기
    if os.path.exists(data_file_path):
        currency_rate = pd.read_csv(data_file_path, index_col='date')

        try:
            currency_rate.index = pd.to_datetime(currency_rate.index, format="%Y-%m-%d")
        except Exception as e:
            st.error(f"날짜 형식 변환 실패: {e}")
            return None

        last_collected_date = currency_rate.index.max()

        for key in hana_million_nat_url_dict.keys():
            tmp = market_index_crawling(key, start_date=last_collected_date)
            if not tmp.empty:
                currency_rate = pd.concat([tmp, currency_rate], axis=0)

        currency_rate.to_csv(data_file_path, index_label='date')
        st.write(f"최신 데이터를 추가로 수집했습니다. (마지막 수집일: {last_collected_date})")

    else:
        for i, key in enumerate(hana_million_nat_url_dict.keys()):
            tmp = market_index_crawling(key)

            if i == 0:
                currency_rate = tmp
            else:
                currency_rate = pd.merge(currency_rate, tmp, left_index=True, right_index=True, how='inner')

        currency_rate.to_csv(data_file_path, index_label='date')
        st.write("모든 데이터를 수집했습니다.")

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


# 앱 실행 함수
def run():
    st.title("환율 데이터 수집 및 차트")

    st.write("데이터를 자동으로 수집 중입니다...")
    currency_rate = collect_currency_data()
    if currency_rate is not None:
        st.success("데이터 수집 완료!")

        # 기간 선택 옵션 추가
        period_options = ["3개월", "1년", "3년", "전체"]
        selected_period = st.selectbox("기간을 선택하세요:", period_options)

        # 선택한 기간만큼 데이터 필터링
        filtered_data = filter_data_by_period(currency_rate, selected_period)

        # 차트 생성 및 저장
        st.write(f"{selected_period} 데이터를 기준으로 차트를 생성 중입니다...")
        chart_file = generate_and_save_chart(filtered_data, f"currency_{selected_period}.png")
        st.success(f"{selected_period} 기간 차트 생성 완료!")

        # 사용자가 클릭 시 차트 표시
        if st.button(f'{selected_period} 차트 보기'):
            st.image(chart_file, caption=f"{selected_period} 환율 차트")
