import os
import time
import socket
import subprocess
import streamlit as st


# 포트가 열려 있는지 확인하는 함수
def is_port_open(port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('localhost', port))
    sock.close()
    return result == 0


# 가상환경과 애플리케이션 경로 정의
apps = {
    "환율 추이": {
        "venv_path": ".\.venv\Scripts\python.exe",
        "script_path": ".\currency_app.py",
        "port": 8502
    },
    # "추천분류": {
    #     "venv_path": "/home/hdh/Projects/subclass_predict_hf/venv/bin/python",
    #     "script_path": "/home/hdh/Projects/0.streamlit/app_v3.py",
    #     "port": 8502
    # },
    # "R&D 다운로드": {
    #     "venv_path": "/home/hdh/Projects/web_crawler/venv/bin/python",
    #     "script_path": "/home/hdh/Projects/kipris_rnd/app.py",
    #     "port": 8503
    # },
    # # 필요한 만큼 추가
}

st.sidebar.title("애플리케이션 선택")
app_choice = st.sidebar.radio("애플리케이션을 선택하세요:", list(apps.keys()))

if app_choice:
    app_info = apps[app_choice]
    command = f"{app_info['venv_path']} -m streamlit run {app_info['script_path']} --server.port={app_info['port']} --server.address=0.0.0.0"
    print(command)

    # 서브 애플리케이션을 백그라운드에서 실행
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # 앱이 실행될 때까지 기다림
    timeout = 15  # 최대 대기 시간 (초)
    start_time = time.time()
    while time.time() - start_time < timeout:
        if is_port_open(app_info['port']):
            break
        time.sleep(1)

    # 포트가 열리지 않은 경우 오류 메시지 출력
    if not is_port_open(app_info['port']):
        st.error("앱이 실행되지 않았거나 포트 연결이 실패했습니다.")
    else:
        # iframe으로 서브 앱을 메인 앱 화면에 임베드
        iframe_url = f"http://localhost:{app_info['port']}"
        st.markdown(f'<iframe src="{iframe_url}" width="100%" height="600"></iframe>', unsafe_allow_html=True)

# 메인 페이지 제목
# st.title("Main Application")
# st.write(f"{app_choice} 실행 준비 완료.")
