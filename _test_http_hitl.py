"""HTTP 层 HITL 完整 E2E 测试"""
import os, sys, time, json, subprocess, urllib.request, urllib.error
from pathlib import Path

# 清理 checkpointer DB
Path("data/checkpoints.sqlite").exists() and Path("data/checkpoints.sqlite").unlink()

PORT = 8900
BASE = f"http://127.0.0.1:{PORT}"

# ---- 1. 启动 server ----
env = os.environ.copy()
env["MOCK_LLM"] = "true"
env["PYTHONPATH"] = "src"

proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "src.server:app", "--host", "127.0.0.1", "--port", str(PORT)],
    cwd="E:\\py_project\\DeepTravelV",
    env=env,
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
)
print("🚀 server starting...")

# 等待 server 就绪
for _ in range(30):
    try:
        r = urllib.request.urlopen(f"{BASE}/docs", timeout=1)
        if r.status == 200:
            break
    except Exception:
        time.sleep(0.3)
else:
    print("❌ server 没起来")
    proc.kill()
    sys.exit(1)
print("✅ server ready on", BASE)

# ---- SSE 收集器 ----
def sse_collect(url, method="POST", data=None, timeout=20):
    events = []
    body = json.dumps(data).encode() if data else b""
    req = urllib.request.Request(url, data=body, method=method)
    req.add_header("Content-Type", "application/json")
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        current = []
        for raw_line in resp:
            line = raw_line.decode("utf-8").rstrip("\r\n")
            if line.startswith("data: "):
                current.append(line[6:])
            elif line == "" and current:
                try:
                    events.append(json.loads("\n".join(current)))
                except Exception:
                    pass
                current = []
            if events and events[-1].get("type") in ("done", "error"):
                break
        return events
    except urllib.error.HTTPError as e:
        return [{"type": "error", "message": e.read().decode()}]
    except Exception as e:
        return [{"type": "error", "message": str(e)}]

# ============ 测试 1: 正常 plan ============
print("\n" + "="*60)
print("测试 1: 正常 plan（全 APPROVED）")
print("="*60)

events = sse_collect(f"{BASE}/api/plan", data={"user_input": "成都3天游", "mock": True})
for e in events:
    t = e.get("type")
    if t == "node":
        print(f"  ✅ {e.get('node', '?'):12s} → {e.get('next_node', '')}")
    elif t == "done":
        print(f"  📦 final_plan = {len(e.get('final_plan',''))} chars")
    elif t == "error":
        print(f"  ❌ {e}"); proc.kill(); sys.exit(1)

assert any(e.get("type") == "done" and e.get("final_plan") for e in events), "正常 plan 应该有 final_plan"
print("✅ 通过")

# ============ 测试 2: 触发 HITL + approve 恢复 ============
print("\n" + "="*60)
print("测试 2: plan(initial_revision_count=1) → REVISE 累积 → HITL → approve 恢复")
print("="*60)

# 2a. 触发 HITL 的 plan
print("\n[2a] plan revision_count=1 → review REVISE → 累积到 2 → 超限 → HITL")
events2 = sse_collect(f"{BASE}/api/plan", data={
    "user_input": "成都3天游",
    "mock": True,
    "initial_revision_count": 1,
})

thread_id = None
hitl_start = None
for e in events2:
    t = e.get("type")
    if t == "start":
        thread_id = e.get("thread_id")
        print(f"  🧵 thread_id = {thread_id}")
    elif t == "node":
        tag = ""
        if e.get("hitl_stage") or e.get("hitl_status"):
            tag = f" HITL=[{e.get('hitl_stage','')}|{e.get('hitl_reason','')[:30]}]"
        print(f"  📌 {e.get('node', '?'):12s} → {e.get('next_node', '')}{tag}")
    elif t == "hitl_start":
        hitl_start = e
        print(f"  🔴 hitl_start! stage={e.get('hitl_stage')}, reason={e.get('hitl_reason')}")
    elif t == "done":
        print(f"  📦 done, hitl_triggered={e.get('hitl_triggered')}")
    elif t == "error":
        print(f"  ❌ {e}")

assert thread_id, "应该有 thread_id"
assert hitl_start or any(e.get("type") == "done" and e.get("hitl_triggered") for e in events2), \
    "应该触发 HITL"
print("✅ HITL 正确触发")

# 2b. approve 恢复
print(f"\n[2b] POST /api/hitl/{thread_id} → approve")
events3 = sse_collect(f"{BASE}/api/hitl/{thread_id}", data={"decision": "approve", "note": "方案OK，批准"})

final_plan = ""
for e in events3:
    t = e.get("type")
    if t == "hitl_decision":
        print(f"  🟢 决策接收: {e.get('decision')}")
    elif t == "node":
        print(f"  🚀 {e.get('node', '?'):12s} → {e.get('next_node', '')}")
    elif t == "done":
        final_plan = e.get("final_plan", "")
        print(f"  📦 done! final_plan={len(final_plan)} chars, total_ms={e.get('total_ms')}")
    elif t == "error":
        print(f"  ❌ {e}")

assert final_plan, "approve 后应该有 final_plan"
print("✅ approve 恢复成功")

# ============ 测试 3: revise 恢复 ============
print("\n" + "="*60)
print("测试 3: 触发 HITL → revise 恢复")
print("="*60)

# 3a. 再触发一次 HITL（新 plan）
print("\n[3a] plan revision_count=1 → HITL...")
events4 = sse_collect(f"{BASE}/api/plan", data={
    "user_input": "成都3天游",
    "mock": True,
    "initial_revision_count": 1,
})

tid2 = None
for e in events4:
    if e.get("type") == "start":
        tid2 = e.get("thread_id")
    if e.get("type") == "hitl_start":
        print(f"  🔴 hitl_start!")
        break
assert tid2

# 3b. revise 恢复
print(f"\n[3b] POST /api/hitl/{tid2} → revise（人工让再改一版）")
events5 = sse_collect(f"{BASE}/api/hitl/{tid2}", data={"decision": "revise", "note": "时间太短，再优化下"})

revise_done = False
for e in events5:
    t = e.get("type")
    if t == "hitl_decision":
        print(f"  🟢 决策: revise")
    elif t == "revise":
        print(f"  🔄 revise: {e.get('to_node')}")
    elif t == "node":
        tag = ""
        if e.get("hitl_stage") or e.get("hitl_status"):
            tag = f" HITL=[{e.get('hitl_stage','')}]"
        print(f"  🚀 {e.get('node', '?'):12s} → {e.get('next_node', '')}{tag}")
    elif t == "done":
        revise_done = True
        print(f"  📦 done, final_plan={'yes' if e.get('final_plan') else 'no'}")
    elif t == "error":
        print(f"  ❌ {e}")

assert revise_done, "revise 后应该有 done"
print("✅ revise 恢复后自动重跑完成")

# ============ 收尾 ============
print("\n" + "="*60)
print("🎉 HTTP HITL 全流程 3 个测试全部通过！")
print("="*60)
proc.kill()
proc.wait(timeout=3)
print("server stopped")
