"""DeepTravel E2E —— 测试脚本只发 HTTP 请求
server 用环境变量 MOCK_LLM=true 在命令行启动
"""
import os, sys, time, json, requests

PORT = 8002
BASE = f"http://127.0.0.1:{PORT}"

def wait_server(timeout=60):
    for i in range(timeout):
        try:
            r = requests.get(f"{BASE}/docs", timeout=1)
            return r.status_code == 200
        except:
            time.sleep(0.5)
    return False

def parse_sse(text):
    events = []
    for block in text.strip().split("\n\n"):
        for line in block.strip().split("\n"):
            if line.startswith("data: "):
                try: events.append(json.loads(line[6:]))
                except: pass
    return events

def plan():
    print("\n[1/5] POST /api/plan ... ", end="", flush=True)
    t0 = time.time()
    r = requests.post(f"{BASE}/api/plan",
        json={"user_input": "帮我规划成都4天3晚情侣游，预算5000", "mock": True},
        timeout=60)
    elapsed = time.time() - t0
    assert r.status_code == 200, f"HTTP {r.status_code}: {r.text[:200]}"
    events = parse_sse(r.text)
    tid = next((e["thread_id"] for e in events if e.get("type") == "start"), None)
    nodes = [e["node"] for e in events if e.get("type") == "node"]
    has_done = any(e.get("type") == "done" for e in events)
    assert tid, "缺 thread_id"
    assert has_done, "缺 done"
    for n in ["coordinator","itinerary","budget","safety","review","integrate"]:
        assert n in nodes, f"缺节点 {n}: {nodes}"
    plan_len = next((len(e.get("final_plan","")) for e in events if e.get("type") == "done"), 0)
    print(f"✅ tid={tid[:14]}.. nodes=6 plan_len={plan_len} {elapsed:.1f}s")
    return tid

def sessions():
    print("[2/5] GET /api/sessions ... ", end="", flush=True)
    r = requests.get(f"{BASE}/api/sessions", timeout=5)
    assert r.status_code == 200
    data = r.json()
    count = data.get("count", 0)
    sess = data.get("sessions", [])
    assert count >= 1 and len(sess) >= 1
    print(f"✅ count={count}")

def get_state(tid):
    print(f"[3/5] GET sessions/{tid[:10]}... ... ", end="", flush=True)
    r = requests.get(f"{BASE}/api/sessions/{tid}", timeout=5)
    assert r.status_code == 200, f"HTTP {r.status_code}"
    data = r.json()
    assert data.get("thread_id") == tid
    assert data.get("last_node") in ("integrate", "end", None, "")
    print(f"✅ last_node={data.get('last_node','?')} plan={'有' if data.get('final_plan') else '无'}")

def adjust(tid):
    print("[4/5] POST /api/adjust ... ", end="", flush=True)
    t0 = time.time()
    r = requests.post(f"{BASE}/api/adjust",
        json={"thread_id": tid, "modify_request": "预算改3000，成都3天2晚", "mock": True},
        timeout=60)
    elapsed = time.time() - t0
    assert r.status_code == 200, f"HTTP {r.status_code}: {r.text[:200]}"
    events = parse_sse(r.text)
    has_done = any(e.get("type") == "done" for e in events)
    adjust_from = next((e.get("start_node") for e in events if e.get("type") == "start_adjust"), "?")
    assert has_done, "缺 done"
    print(f"✅ adjust_from={adjust_from} events={len(events)} {elapsed:.1f}s")

def delete(tid):
    print(f"[5/5] DELETE sessions/{tid[:10]}... ... ", end="", flush=True)
    r = requests.delete(f"{BASE}/api/sessions/{tid}", timeout=5)
    assert r.status_code == 200
    print("✅")

if __name__ == "__main__":
    print("="*60)
    print("DeepTravel E2E (test client)")
    print("请先用 MOCK_LLM=true python -m uvicorn src.server:app --port 8002 启动 server")
    print("="*60)
    print("\n等待 server ... ", end="", flush=True)
    assert wait_server(60), "server 没启动"
    print("✅")

    tid = plan()
    sessions()
    get_state(tid)
    adjust(tid)
    delete(tid)

    print("\n" + "="*60)
    print("🎉 5/5 测试通过")
    print("="*60)
