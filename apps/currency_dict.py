import numpy as np

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