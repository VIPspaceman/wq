# -*- coding: utf-8 -*-
"""独立测试 BRAIN check 端点。复用项目的认证逻辑。"""
import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from config.settings import AppSettings
from brain.authentication import create_authenticated_session

TEST_ALPHA_ID = "omWLgVKv"

def main():
    settings = AppSettings.from_env()
    print("[1] 用项目认证登录...")
    session = create_authenticated_session(
        settings.credentials, settings.endpoints
    )
    print("[2] 登录成功")

    print(f"\n=== GET /alphas/{TEST_ALPHA_ID}/check ===")
    url = f"{settings.endpoints.base_url}/alphas/{TEST_ALPHA_ID}/check"
    r = session.get(url)
    print(f"status = {r.status_code}")
    try:
        data = r.json()
        print(f"response = {json.dumps(data, indent=2, ensure_ascii=False)[:2000]}")
    except Exception:
        print(f"text = {r.text[:500]}")

if __name__ == "__main__":
    main()
