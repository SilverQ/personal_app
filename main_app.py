import streamlit as st

import mahalanobis_distance_app
from apps import (currency_app, dummy_app) # 각 앱을 모듈로 분리
from apps import stock_analysis_app_v3 as stock_analysis_app
# from apps import chat

# 🎯 와이드 레이아웃 설정 (이 부분 추가!)
st.set_page_config(
    page_title="SilverQ Main Application",
    page_icon="📊",
    layout="wide",  # ← 이것이 핵심!
    initial_sidebar_state="expanded"
)
#
# # 메인 페이지 제목
# st.title("SilverQ Main Application")

# 사이드바에서 앱 선택
st.sidebar.title("애플리케이션 선택")
app_choice = st.sidebar.radio("애플리케이션을 선택하세요:",
                              ["환율 추이", "주식 분석", "Find Outlier"
                                  # , "Ollama"
                                  , "더미 앱"])

# 선택한 앱에 따라 다른 기능을 보여줌
if app_choice == "환율 추이":
    currency_app.run()  # 환율 추이 앱을 실행
elif app_choice == "주식 분석":
    stock_analysis_app.run()  # 대화 앱을 실행
elif app_choice == "Find Outlier":
    mahalanobis_distance_app.run()  # 대화 앱을 실행
# elif app_choice == "Ollama":
#     # Ollama.run()  # 대화 앱을 실행
#     chat.run_app()  # 대화 앱을 실행
elif app_choice == "더미 앱":
    dummy_app.run()  # 더미 앱 실행

