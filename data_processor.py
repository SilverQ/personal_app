import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


class DataProcessor:
    def __init__(self, data_file_path='data/naver_currency.csv'):
        self.data_file_path = data_file_path

    def collect_currency_data(self, crawling_instance):
        # 데이터 파일이 존재할 경우: 최신 데이터만 가져오기
        if os.path.exists(self.data_file_path):
            currency_rate = pd.read_csv(self.data_file_path, index_col='date')
            currency_rate.index = pd.to_datetime(currency_rate.index, format="%Y-%m-%d")
            last_collected_date = currency_rate.index.max()

            for key in list(crawling_instance.hana_million_nat_url_dict.keys()):
                tmp = crawling_instance.market_index_crawling(key, start_date=last_collected_date)
                if not tmp.empty:  # 새로운 데이터가 있으면 병합
                    currency_rate = pd.concat([tmp, currency_rate], axis=0)

            currency_rate.to_csv(self.data_file_path, index_label='date')

        else:
            # 데이터 파일이 없을 경우 모든 데이터를 수집
            for key in list(crawling_instance.hana_million_nat_url_dict.keys()):
                tmp = crawling_instance.market_index_crawling(key)
                if not tmp.empty:
                    if 'currency_rate' in locals():
                        currency_rate = pd.merge(currency_rate, tmp, left_index=True, right_index=True, how='inner')
                    else:
                        currency_rate = tmp
            currency_rate.to_csv(self.data_file_path, index_label='date')

        return currency_rate

    def filter_data_by_period(self, currency_rate, period):
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

    def generate_and_save_chart(self, df, file_name):
        fig, axs = plt.subplots(len(df.columns) // 4 + 1, 4, figsize=(20, 15))
        for i, col in enumerate(df.columns):
            axs[i // 4, i % 4].plot(df.index, df[col])
            axs[i // 4, i % 4].set_title(col)
            start, end = axs[i // 4, i % 4].get_xlim()
            axs[i // 4, i % 4].xaxis.set_ticks(np.linspace(start, end, 6))
        plt.tight_layout()
        plt.savefig(file_name)
        return file_name
