import os
import time
import numpy as np
import pandas as pd
import requests as re
import streamlit as st

from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

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
    # 'WTI': 'http://finance.naver.com/marketindex/exchangeDailyQuote.nhn?marketindexCd=OIL_CL&fdtc=2',
    # '국제 금': 'http://finance.naver.com/marketindex/exchangeDailyQuote.nhn?marketindexCd=CMDT_GC&fdtc=2'
    }
hana_million_nat = {
    'USD': '미국' , 'EUR': '유럽' , 'JPY': '일본' , 'GBP': '영국' ,
    'CAD': '캐나다', 'CHF': '스위스' , 'HKD': '홍콩', 'SEK': '스웨덴',
    'AUD': '호주', 'AED': 'UAE', 'BHD': '바레인' , 'CNY': '중국', 'CZK': '체코',
    'DKK': '덴마크', 'HUF': '헝가리', 'IDR': '인도네시아', 'KWD': '쿠웨이트',
    'MXN': '멕시코', 'NOK': '노르웨이', 'NZD': '뉴질랜드', 'PLN': '폴란드',
    'RUB': '러시아', 'SAR': '사우디', 'SGD': '싱가포르', 'THB': '태국',
    'TRY': '튀르키에', 'ZAR': '남아공'
    # , 'KRW': '한국'
    }
hana_million_nat_dtype = {
    'USD': np.float64, 'EUR': np.float64, 'JPY': np.float64, 'GBP': np.float64,
    'CAD': np.float64, 'CHF': np.float64, 'HKD': np.float64, 'SEK': np.float64,
    'AUD': np.float64, 'AED': np.float64, 'BHD': np.float64, 'CNY': np.float64,
    'DKK': np.float64, 'HUF': np.float64, 'IDR': np.float64, 'KWD': np.float64,
    'MXN': np.float64, 'NOK': np.float64, 'NZD': np.float64, 'PLN': np.float64,
    'RUB': np.float64, 'SAR': np.float64, 'SGD': np.float64, 'THB': np.float64,
    'TRY': np.float64, 'ZAR': np.float64, 'CZK': np.float64
    # , 'KRW': '한국'
    }

# # Streamlit 임포트는 터미널에서 실행할 때만 필요
# if __name__ != "__main__":
#     import streamlit as st


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
        # 파일에서 첫 번째 컬럼을 인덱스로 설정하고 날짜 형식으로 변환
        currency_rate = pd.read_csv(data_file_path, index_col='date')

        # 날짜 인덱스를 datetime 형식으로 변환 (문자열 형식일 경우)
        try:
            currency_rate.index = pd.to_datetime(currency_rate.index, format="%Y-%m-%d")
        except Exception as e:
            if __name__ != "__main__":
                st.error(f"날짜 형식 변환 실패: {e}")
            else:
                print(f"날짜 형식 변환 실패: {e}")
            return None

        last_collected_date = currency_rate.index.max()

        for i, key in enumerate(list(hana_million_nat.keys())):
            url_key = hana_million_nat[key] + ' ' + key
            tmp = market_index_crawling(url_key, start_date=last_collected_date)

            if not tmp.empty:  # 새로운 데이터가 있으면 병합
                currency_rate = pd.concat([tmp, currency_rate], axis=0)

        # CSV 파일 저장 시 index_label='date'로 인덱스에 이름을 부여
        currency_rate.to_csv(data_file_path, index_label='date')
        if __name__ != "__main__":
            st.write(f"최신 데이터를 추가로 수집했습니다. (마지막 수집일: {last_collected_date})")
        else:
            print(f"최신 데이터를 추가로 수집했습니다. (마지막 수집일: {last_collected_date})")

    # 데이터 파일이 없을 경우: 모든 데이터를 수집
    else:
        for i, key in enumerate(list(hana_million_nat.keys())):
            url_key = hana_million_nat[key] + ' ' + key
            tmp = market_index_crawling(url_key)

            if i == 0:
                currency_rate = tmp
            else:
                currency_rate = pd.merge(currency_rate, tmp, left_index=True, right_index=True, how='inner')

        # CSV 파일 저장 시 index_label='date'로 인덱스에 이름을 부여
        currency_rate.to_csv(data_file_path, index_label='date')
        if __name__ != "__main__":
            st.write("모든 데이터를 수집했습니다.")
        else:
            print("모든 데이터를 수집했습니다.")

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
    plt.savefig(f'data/{file_name}')
    return f'data/{file_name}'


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
        start_date = pd.to_datetime(currency_rate.index.min())  # 전체 데이터

    filtered_data = currency_rate.loc[currency_rate.index >= str(start_date.date())]
    return filtered_data


# # Streamlit UI 코드
# if __name__ != "__main__":
#     st.title("환율 데이터 수집 및 차트")
#
#     # 데이터 수집 자동 실행
#     st.write("데이터를 자동으로 수집 중입니다...")
#     currency_rate = collect_currency_data()
#     if currency_rate is not None:
#         st.success("데이터 수집 완료!")
#
#         # 기간 선택 옵션 추가
#         period_options = ["3개월", "1년", "3년", "전체"]
#         selected_period = st.selectbox("기간을 선택하세요:", period_options)
#
#         # 선택한 기간만큼 데이터 필터링
#         filtered_data = filter_data_by_period(currency_rate, selected_period)
#
#         # 차트 생성 및 저장
#         st.write(f"{selected_period} 데이터를 기준으로 차트를 생성 중입니다...")
#         chart_file = generate_and_save_chart(filtered_data, f"currency_{selected_period}.png")
#         st.success(f"{selected_period} 기간 차트 생성 완료!")
#
#         # 사용자가 클릭 시 차트 표시
#         if st.button(f'{selected_period} 차트 보기'):
#             st.image(chart_file, caption=f"{selected_period} 환율 차트")
#
# # PyCharm 또는 디버깅 모드에서는 실행만
# if __name__ == "__main__":
#     # 데이터 수집 및 차트 생성 (디버깅용)
#     currency_rate = collect_currency_data()
#     if currency_rate is not None:
#         period = "1년"  # 기본 디버깅용 기간 설정
#         filtered_data = filter_data_by_period(currency_rate, period)
#         print(f"수집된 데이터({period} 기준):\n", filtered_data.head())
#
#         # 차트 생성 및 파일 경로 출력
#         chart_file = generate_and_save_chart(filtered_data, f"currency_debug_{period}.png")
#         print(f"생성된 차트 파일: {chart_file}")
#
#     st.title("환율 데이터 수집 및 차트")

# 데이터 수집 자동 실행
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