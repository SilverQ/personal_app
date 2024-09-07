import streamlit as st
import pandas as pd
import numpy as np
import requests as re
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 데이터 수집 및 처리 관련 로직
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


# 앱 실행 함수
def run():
    st.title("환율 추이")

    # 데이터 수집 함수 (예시)
    def market_index_crawling(key, pages=1):
        date = []
        value = []
        for i in range(1, pages + 1):
            url = re.get(hana_million_nat_url_dict[key] + '&page=%s' % i)
            html = BeautifulSoup(url.content, 'html.parser')
            tbody = html.find('tbody').find_all('tr')

            for tr in tbody:
                temp_date = tr.find('td', {'class': 'date'}).text.replace('.', '-').strip()
                temp_value = float(tr.find('td', {'class': 'num'}).text.strip().replace(',', ''))
                temp_date = pd.to_datetime(temp_date)

                date.append(temp_date)
                value.append(temp_value)

        data = pd.DataFrame(value, index=date, columns=[key.split()[1]])
        return data

    # 환율 데이터를 선택하여 표시
    currency_options = list(hana_million_nat_url_dict.keys())
    selected_currency = st.selectbox("환율을 선택하세요:", currency_options)

    # 선택한 환율 데이터를 크롤링하여 표시
    if st.button(f"{selected_currency} 데이터 가져오기"):
        data = market_index_crawling(selected_currency)
        st.write(data)

        # 차트 생성 및 표시
        st.line_chart(data)
