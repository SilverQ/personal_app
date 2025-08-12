#!/usr/bin/env python3
import os, sys, platform, subprocess, shlex, logging, time, shutil
from pathlib import Path
import configparser
import requests

ROOT = Path(__file__).resolve().parent
LOGDIR = ROOT / "logs"; LOGDIR.mkdir(exist_ok=True)
APP = ROOT / "main_app.py"           # Streamlit entry
VENV = ROOT / "venv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOGDIR / "launcher.log"), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("launcher")

def py_exe_from_venv() -> str:
    if platform.system() == "Windows":
        return str(VENV / "Scripts" / "python.exe")
    return str(VENV / "bin" / "python")

def run(cmd, cwd=None, env=None, check=True):
    if isinstance(cmd, str):
        cmd = shlex.split(cmd)
    log.info("$ %s", " ".join(cmd))
    p = subprocess.run(cmd, cwd=cwd, env=env, text=True, capture_output=True)
    if p.stdout: log.info(p.stdout.strip())
    if p.stderr: log.warning(p.stderr.strip())
    if check and p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)} (rc={p.returncode})")
    return p

def ensure_venv():
    if VENV.exists():
        return
    py = shutil.which("python3") or shutil.which("python") or sys.executable
    if not py:
        raise RuntimeError("python 실행 파일을 찾을 수 없습니다.")
    run([py, "-m", "venv", str(VENV)])

def pip_install(reqs: list[str]):
    pip = [py_exe_from_venv(), "-m", "pip"]
    run(pip + ["install", "--upgrade", "pip", "setuptools", "wheel"])
    req_txt = ROOT / "requirements.txt"
    if req_txt.exists():
        run(pip + ["install", "-r", str(req_txt)])
    if reqs:
        run(pip + ["install"] + reqs)

def update_duckdns():
    cfg = configparser.ConfigParser()
    cfg.read(ROOT / "config.ini")
    if "DuckDNS" not in cfg:
        return
    domain = cfg["DuckDNS"].get("domain")
    token = cfg["DuckDNS"].get("token")
    if not domain or not token:
        return
    url = f"https://www.duckdns.org/update?domains={domain}&token={token}&ip="
    try:
        r = requests.get(url, timeout=10)
        log.info("DuckDNS: %s %s", r.status_code, r.text.strip())
    except Exception as e:
        log.warning("DuckDNS 업데이트 실패: %s", e)

def _copy_into(src: Path, dst: Path):
    for item in src.iterdir():
        if item.name in ("venv", "logs", ".git"):
            continue
        target = dst / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            if target.exists():
                target.unlink()
            shutil.copy2(item, target)

def _clean_project_except(dst: Path, keep=("venv", "logs")):
    for item in dst.iterdir():
        if item.name in keep:
            continue
        if item.is_dir():
            shutil.rmtree(item, ignore_errors=True)
        else:
            try: item.unlink()
            except FileNotFoundError: pass

def update_repo():
    if (ROOT / ".git").exists():
        # 기존 git 레포 → 최신화
        run("git fetch --all", cwd=ROOT)
        try:
            br = os.environ.get("GIT_BRANCH") or \
                 run("git rev-parse --abbrev-ref origin/HEAD", cwd=ROOT).stdout.strip().split("/")[-1]
        except Exception:
            br = "main"
        run(f"git checkout {br}", cwd=ROOT)
        run(f"git reset --hard origin/{br}", cwd=ROOT)
    else:
        # 최초 배포: GIT_REMOTE 로 클론한 뒤, 현재 디렉터리에 반영
        remote = os.environ.get("GIT_REMOTE")
        if not remote:
            log.warning("현재 폴더가 git repo가 아니고, GIT_REMOTE가 설정되지 않았습니다. 업데이트를 건너뜁니다.")
            return
        parent = ROOT.parent
        tmpdir = parent / ("repo_" + str(int(time.time())))
        run(["git", "clone", remote, str(tmpdir)], cwd=parent)
        _clean_project_except(ROOT, keep=("venv", "logs"))
        _copy_into(tmpdir, ROOT)
        shutil.rmtree(tmpdir, ignore_errors=True)
        log.info("리포지토리 초기 동기화 완료")

def run_streamlit():
    py = py_exe_from_venv()
    if not Path(py).exists():
        raise RuntimeError(f"venv python 경로가 없습니다: {py}")
    cmd = [
        py, "-m", "streamlit", "run", str(APP),
        "--server.address=0.0.0.0",
        "--server.port", os.environ.get("PORT", "8501"),
        "--server.headless=true",
    ]
    out = open(LOGDIR / "streamlit.out", "a", encoding="utf-8")
    err = open(LOGDIR / "streamlit.err", "a", encoding="utf-8")
    log.info("Streamlit 시작…")
    p = subprocess.Popen(cmd, stdout=out, stderr=err, cwd=str(ROOT))
    log.info("PID=%s", p.pid)
    return p

if __name__ == "__main__":
    # try:
    #     update_duckdns()
    # except Exception as e:
    #     log.warning("DuckDNS 스킵: %s", e)

    try:
        update_repo()
    except Exception as e:
        log.error("Git 업데이트 실패: %s", e)

    try:
        ensure_venv()
        pip_install(["streamlit", "plotly", "kaleido", "reportlab", "pykrx", "OpenDartReader", "requests"])
    except Exception as e:
        log.error("패키지 설치 실패: %s", e)
        sys.exit(1)

    proc = run_streamlit()
    log.info("로그 파일: %s / %s", LOGDIR / "streamlit.out", LOGDIR / "streamlit.err")
