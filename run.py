#!/usr/bin/env python3
"""
Entry point duy nhất để khởi động dự án.

Cách dùng:
    python run.py                    # Backend only (port: server.port trong configs/app.yaml)
    python run.py --with-ui          # Backend + Chainlit UI (port: server.ui_port)
    python run.py --port 8002        # Đổi port backend
    python run.py --reload           # Hot-reload (dev mode)
    python run.py --with-ui --ui-port 8080
"""

import argparse
import subprocess
import sys
import time
import os

from app.core.config import get_settings


def parse_args():
    server = get_settings().server
    parser = argparse.ArgumentParser(description="MultiAgents RAG FnB — startup")
    parser.add_argument("--host", default=server.host, help=f"Backend host (default: {server.host})")
    parser.add_argument("--port", type=int, default=server.port, help=f"Backend port (default: {server.port})")
    parser.add_argument("--reload", action="store_true", help="Enable hot-reload (dev)")
    parser.add_argument("--with-ui", action="store_true", help="Also start Chainlit UI")
    parser.add_argument("--ui-port", type=int, default=server.ui_port, help=f"Chainlit port (default: {server.ui_port})")
    return parser.parse_args()


def check_port(port: int) -> bool:
    """Trả về True nếu port còn trống."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) != 0


def free_port(port: int) -> None:
    """Kill process đang giữ port (Linux)."""
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True, text=True,
        )
        pids = result.stdout.strip().split()
        if pids:
            for pid in pids:
                os.kill(int(pid), 9)
            time.sleep(1)
            print(f"[OK] Released port {port} (killed PID {', '.join(pids)})")
    except Exception as exc:
        print(f"[WARN] Could not free port {port}: {exc}")


def check_neo4j() -> bool:
    """Kiểm tra Neo4j có sẵn sàng không trước khi start."""
    try:
        from neo4j import GraphDatabase

        settings = get_settings()
        uri = settings.neo4j.uri
        user = settings.neo4j.user
        password = settings.neo4j.password
        max_retries = settings.server.neo4j_startup_retries
        delay = settings.server.neo4j_startup_retry_delay_seconds

        for attempt in range(1, max_retries + 1):
            try:
                driver = GraphDatabase.driver(uri, auth=(user, password))
                driver.verify_connectivity()
                driver.close()
                print(f"[OK] Neo4j connected: {uri}")
                return True
            except Exception as exc:
                print(f"[WARN] Neo4j attempt {attempt}/{max_retries}: {exc}")
                if attempt < max_retries:
                    time.sleep(delay)

        return False

    except ImportError:
        print("[WARN] neo4j package not found, skipping connectivity check.")
        return True


def start_chainlit(api_base_url: str, port: int) -> subprocess.Popen:
    env = os.environ.copy()
    env["CHAINLIT_API_BASE_URL"] = api_base_url

    proc = subprocess.Popen(
        [sys.executable, "-m", "chainlit", "run", "chainlit_app.py", "--port", str(port)],
        env=env,
    )
    print(f"[OK] Chainlit UI started → http://localhost:{port}")
    return proc


def main():
    args = parse_args()

    print("=" * 50)
    print("  MultiAgents RAG FnB")
    print("=" * 50)

    # 1. Check Neo4j
    neo4j_ok = check_neo4j()
    if not neo4j_ok:
        print("[ERROR] Cannot connect to Neo4j. Start Neo4j first:")
        print("  docker start neo4j-fnb")
        print("  # hoặc: docker run --name neo4j-fnb -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password -d neo4j:5")
        sys.exit(1)

    # 2. Check port
    if not check_port(args.port):
        print(f"[WARN] Port {args.port} đang bị chiếm. Đang giải phóng...")
        free_port(args.port)
        if not check_port(args.port):
            print(f"[ERROR] Không thể giải phóng port {args.port}. Dùng --port để chọn port khác.")
            sys.exit(1)

    # 3. Start Chainlit (nếu --with-ui)
    ui_proc = None
    if args.with_ui:
        if not check_port(args.ui_port):
            print(f"[WARN] UI port {args.ui_port} bị chiếm. Đang giải phóng...")
            free_port(args.ui_port)
        api_base = f"http://localhost:{args.port}"
        ui_proc = start_chainlit(api_base_url=api_base, port=args.ui_port)
        print(f"[INFO] API:  http://localhost:{args.port}/docs")
        print(f"[INFO] UI:   http://localhost:{args.ui_port}")
    else:
        print(f"[INFO] API:  http://localhost:{args.port}/docs")
        print(f"[INFO] Tip:  python run.py --with-ui  để mở Chainlit")

    print("=" * 50)

    # 3. Start uvicorn (blocking)
    import uvicorn
    try:
        uvicorn.run(
            "app.main:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
        )
    finally:
        if ui_proc:
            ui_proc.terminate()


if __name__ == "__main__":
    main()
