import os
import requests
import git
import subprocess
import configparser
import requests


# DuckDNS IP 업데이트
def update_duckdns():
    config = configparser.ConfigParser()
    config.read('config.ini')
    domain = config['DuckDNS']['domain']
    token = config['DuckDNS']['token']
    url = f"https://www.duckdns.org/update?domains={domain}&token={token}"
    requests.get(url)


# GitHub 저장소 업데이트
def update_github_repo():
    repo_path = "D:\DL_work\personal_app"
    repo = git.Repo(repo_path)
    # # 작업 디렉토리 경로
    # print(repo.working_dir)
    # # Git 디렉토리 경로
    # print(repo.git_dir)
    # # 현재 활성 브랜치
    # print(repo.active_branch)
    # # 모든 브랜치 목록
    # for branch in repo.branches:
    #     print(branch)
    repo.remotes.origin.pull()


# Streamlit 서버 실행
def run_streamlit():
    streamlit_script = r"D:\DL_work\personal_app\main_app.py"
    command = [
        "streamlit", "run", streamlit_script,
        "--server.port=8501",
        "--server.address=0.0.0.0"
    ]
    print("Running command:", " ".join(command))
    subprocess.run(command, creationflags=subprocess.CREATE_NEW_CONSOLE)
    with open("streamlit_error.log", "w") as error_log:
        subprocess.Popen(command, stderr=error_log)


if __name__ == "__main__":
    update_duckdns()
    update_github_repo()
    run_streamlit()
