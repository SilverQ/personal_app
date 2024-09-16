import os
import requests
import git
import subprocess


# DuckDNS IP 업데이트
def update_duckdns():
    domain = "http://silverq.duckdns.org"
    token = "2b62b2c2-aaab-4bc6-9c5c-7f162f1835f1"
    url = f"https://www.duckdns.org/update?domains={domain}&token={token}&ip="
    requests.get(url)


# GitHub 저장소 업데이트
def update_github_repo():
    repo_path = "https://github.com/SilverQ/personal_app"
    repo = git.Repo(repo_path)
    repo.remotes.origin.pull()


# Streamlit 서버 실행
def run_streamlit():
    streamlit_script = "D:\DL_work\personal_app"
    subprocess.Popen(["streamlit", "run", streamlit_script])


if __name__ == "__main__":
    update_duckdns()
    update_github_repo()
    run_streamlit()
