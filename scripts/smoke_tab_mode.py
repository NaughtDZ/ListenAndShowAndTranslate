"""冒烟测试：真实入口 ``main.py --tab`` 能不能起来、端口能不能通、握手能不能成。

单元测试是 monkeypatch 掉服务端启动的，端到端脚本用的是自己 new 出来的
``TabAudioServer``——**这两条都没覆盖"真实入口 + 真实配置"**。
这个脚本补上：用 pythonw 起真正的字幕进程，然后像扩展一样连上去握手。

会短暂弹出控制窗（约 10 秒）后自动杀掉。

用法::

    .venv\\Scripts\\python.exe scripts\\smoke_tab_mode.py
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from app.audio.ws_client import MiniWSClient  # noqa: E402
from app.config import AppConfig  # noqa: E402


def port_open(port: int) -> bool:
    s = socket.socket()
    s.settimeout(0.5)
    try:
        s.connect(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def kill_tree() -> int:
    import psutil

    killed = 0
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmd = " ".join(proc.info["cmdline"] or [])
            if "main.py" in cmd and "--tab" in cmd:
                proc.kill()
                killed += 1
        except Exception:  # noqa: BLE001
            continue
    return killed


def main() -> int:
    cfg = AppConfig.load()
    port = int(cfg.tab_audio.port)
    token = cfg.tab_audio.token

    py = Path(sys.executable)
    pythonw = py.with_name("pythonw.exe")
    exe = pythonw if pythonw.exists() else py

    kill_tree()  # 先清掉可能残留的
    print(f"启动 {exe.name} main.py --tab（端口 {port}）…")
    proc = subprocess.Popen(
        [str(exe), str(ROOT / "main.py"), "--tab"],
        cwd=str(ROOT),
        close_fds=True,
    )

    results: list[tuple[bool, str]] = []
    try:
        deadline = time.time() + 25
        while time.time() < deadline and not port_open(port):
            time.sleep(0.5)
        listening = port_open(port)
        results.append((listening, f"端口 {port} 已监听（说明 TabAudioServer 起来了）"))
        if not listening:
            return report(results, proc)

        # 像扩展一样握手：控制角色
        cli = MiniWSClient(f"ws://127.0.0.1:{port}/lst/tab").connect()
        cli.send_json(
            {
                "type": "hello",
                "protocol": 1,
                "role": "control",
                "token": token,
                "format": {"rate": 48000, "channels": 2, "dtype": "float32"},
                "tab": {"id": 1, "title": "冒烟测试标签页"},
                "browser": {"name": "Edge"},
                "capturing": False,
            }
        )
        reply = cli.recv_json(timeout=5)
        results.append(
            (reply.get("type") == "welcome", f"握手成功：{reply}")
        )

        # 音频角色再连一条，并送一段真 PCM（进程是否真的在消费音频）
        audio = MiniWSClient(f"ws://127.0.0.1:{port}/lst/tab").connect()
        audio.send_json(
            {
                "type": "hello",
                "protocol": 1,
                "role": "audio",
                "token": token,
                "format": {"rate": 48000, "channels": 2, "dtype": "float32"},
                "tab": {"id": 1, "title": "冒烟测试标签页"},
                "capturing": True,
            }
        )
        audio.recv_json(timeout=5)
        import numpy as np

        t = np.arange(960, dtype=np.float64) / 48000
        mono = (np.sin(2 * np.pi * 440 * t) * 0.2).astype(np.float32)
        pcm = np.stack([mono, mono], axis=1).reshape(-1).astype(np.float32).tobytes()
        for _ in range(10):
            audio.send_binary(pcm)
            time.sleep(0.02)
        time.sleep(0.5)
        results.append((True, "两条连接（控制 + 音频）可以共存，PCM 已送入（未抛异常）"))
        cli.close()
        audio.close()
    finally:
        killed = kill_tree()
        print(f"已清理 main.py --tab 进程 {killed} 个")

    return report(results, proc)


def report(results: list[tuple[bool, str]], proc) -> int:
    print("\n判定：")
    for ok, label in results:
        print(f"   {'✅' if ok else '❌'} {label}")
    passed = sum(1 for ok, _ in results if ok)
    print(f"\n{passed}/{len(results)} 项通过")

    log = ROOT / "data" / "logs" / "lst.log"
    if log.exists():
        print("\n日志末尾：")
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines()[-12:]:
            print(f"   {line}")
    return 0 if passed == len(results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
