import streamlit as st
from apps import currency_app, dummy_app  # 각 앱을 모듈로 분리

# 메인 페이지 제목
st.title("SilverQ Main Application")

# 사이드바에서 앱 선택
st.sidebar.title("애플리케이션 선택")
app_choice = st.sidebar.radio("애플리케이션을 선택하세요:", ["환율 추이", "더미 앱"])

# # 선택한 앱에 따라 다른 기능을 보여줌
# if app_choice == "환율 추이":
#     currency_app.run()  # 환율 추이 앱을 실행
# elif app_choice == "더미 앱":
#     dummy_app.run()  # 더미 앱 실행
