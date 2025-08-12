import os
# import requests
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
    current_file_path = os.path.abspath(__file__)
    repo_path = os.path.dirname(current_file_path)  # 또는 상위 폴더 기준으로 조정 가능
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
    # 가상환경 python 경로 (예: venv가 D:\DL_work\personal_app\venv 인 경우)
    venv_python = os.path.join(os.path.dirname(__file__), "venv", "Scripts", "python.exe")
    streamlit_script = os.path.join(os.path.dirname(__file__), "main_app.py")
    command = [
        venv_python, "-m", "streamlit", "run", streamlit_script,
        "--server.port=8501",
        "--server.address=0.0.0.0"
    ]
    print("Running command:", " ".join(command))
    subprocess.run(command, creationflags=subprocess.CREATE_NEW_CONSOLE)
    with open("streamlit_error.log", "w") as error_log:
        subprocess.Popen(command, stderr=error_log)


# Ollama 백엔드 서버 실행
def run_ollama_backend():
    venv_path = r"D:\DL_work\personal_app\venv\Scripts\activate"
    ollama_script = r"D:\DL_work\personal_app\apps\Ollama.py"
    command = f'cmd /k "{venv_path} & python {ollama_script}"'
    print("Running command:", command)
    with open("ollama_backend_error.log", "w") as error_log:
        result = subprocess.run(command, stderr=subprocess.PIPE, shell=True, text=True)
        error_log.write(result.stderr)


if __name__ == "__main__":
    # update_duckdns()
    update_github_repo()
    run_streamlit()
    # run_ollama_backend()
