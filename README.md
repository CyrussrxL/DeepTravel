# 🗺️ DeepTravel

基于 **LangGraph** 的多 Agent 旅行规划管家。

用户输入旅行需求后，Coordinator 结构化解析并 fan-out 并行调度三个专业 Agent（行程/预算/安全）调用真实数据工具，经代码级交叉校验与 LLM 四维评审双重把关，审核不通过自动回退修订，整合输出完整方案；支持增量调整（只重跑受影响环节）、HITL 人工审核与跨会话持久化。

---

## ✨ 核心特性

| 特性 | 说明 |
|------|------|
| 并行有向图 | `Coordinator → {Itinerary ∥ Budget ∥ Safety} → Review → Integrate`，fan-out 并行调度 |
| 结构化输出 | Coordinator/Review 走 Function Calling + Pydantic 约束解码，产出可校验的字段化结果 |
| 确定性校验 | 预算总和 / 天数 / 日期区间三条代码级交叉校验，确定性捕获 LLM 幻觉（越界日期、总额偏差）并自动修正 |
| 审核回退 | Review 评分门控（阈值判定在代码层），`revise` 自动回退 Itinerary，`MAX_REVISION=2` 上限保护 |
| 增量调整 | LLM 语义路由定位最早受影响 Agent，仅重跑受影响段并复用上游产物（Checkpointer） |
| HITL 人工审核 | 修订超限 / 低分门控 / 节点超时自动转入人工审核，approve 或 revise 后从断点续跑 |
| 真实数据 | Itinerary 接入**高德地图 POI**，Safety 接入**心知天气**，API 不通自动降级、流程不中断 |
| 服务化 | FastAPI + SSE 流式推送各 Agent 进度与四维评审评分，Vue3 单页前端 |
| Mock 模式 | `MOCK_LLM=true` 跳过所有 LLM + API 调用，秒级验证图路由，无需任何 Key |

---

## 🏗️ 架构

```
                ┌──────────────┐
                │ coordinator  │ ◀── START（结构化输出：归一化需求）
                └──────┬───────┘
                       │  parsed_info
          ┌────────────┼────────────┐
          ▼            ▼            ▼      ← fan-out 并行
    ┌──────────┐ ┌──────────┐ ┌──────────┐
    │itinerary │ │  budget  │ │  safety  │
    │(高德POI) │ │          │ │(心知天气)│
    └────┬─────┘ └────┬─────┘ └────┬─────┘
         └────────────┼────────────┘
                      ▼                 ← fan-in
               ┌────────────┐
               │   review   │  四维评分 + 三条交叉校验
               └──┬─────┬───┘
          approve │     │ revise（≤ MAX_REVISION）
                  ▼     └──▶ 回退 itinerary
          ┌────────────┐         │
          │  integrate │ ◀───────┘
          └─────┬──────┘
                ▼
               END

  review 门控未过 ──▶ HITL 人工审核（暂停等待，approve/revise 续跑）
  调整请求 ──▶ 增量子图（语义路由到最早受影响节点，段内并行重跑）
```

**路由原则**：节点只写 `state["next_node"]`，`route_decision(state)` 纯函数统一调度；LLM 只产数据，阈值判定与路由决策全部在代码层。

---

## 📁 目录结构

```
DeepTravel/
├── .env.example          # 配置模板（复制为 .env 填入你的 Key）
├── requirements.txt
├── static/index.html     # Vue3 单页前端（Agent 进度 + 评审评分 + Markdown 渲染）
└── src/
    ├── main.py           # CLI 入口
    ├── server.py         # FastAPI + SSE（/api/plan /api/adjust /api/hitl）
    ├── graph.py          # LangGraph 并行图编排 + checkpoint 清理
    ├── adjust.py         # 增量调整子图（语义路由 + 并行重跑）
    ├── hitl.py           # HITL 门控与人工决策恢复
    ├── state.py          # TravelState TypedDict + reducers
    └── agents/
        ├── common.py     # LLM 初始化（DashScope 兼容端点，环境变量驱动）
        ├── tools.py      # 高德地图 / 心知天气工具封装
        └── nodes.py      # 七个 Agent 节点函数（含结构化输出与交叉校验）
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env      # Windows: copy .env.example .env
# 编辑 .env，填入 DASHSCOPE_API_KEY（必需）与 AMAP/SENIVERSE Key（可选）
```

所有配置均从环境变量读取，代码中零硬编码密钥。

### 3. 运行

```bash
# 方式一：Web 界面（推荐）
uvicorn src.server:app --host 0.0.0.0 --port 8000
# 浏览器打开 http://localhost:8000

# 方式二：CLI
python -m src.main "我想去成都旅游3天2晚，两个人，预算5000元，喜欢大熊猫和美食"

# 无 Key 冒烟（全 mock，秒级跑通流程）
MOCK_LLM=true python -m src.main
```

---

## 🔌 环境变量说明

| 变量 | 必填 | 提供者 | 用途 |
|------|------|--------|------|
| `DASHSCOPE_API_KEY` | ✅ | 阿里云百炼 | 主 LLM |
| `DASHSCOPE_MODEL` | - | 阿里云百炼 | 默认 `qwen3.8-27b`，支持 qwen3.8-max / qwen-plus / qwen3.7-flash |
| `DASHSCOPE_BASE_URL` | - | - | 默认 DashScope OpenAI 兼容端点 |
| `DASHSCOPE_ENABLE_THINKING` | - | - | 默认 `false`（qwen3 思考模式占用 output token） |
| `DASHSCOPE_MAX_TOKENS` | - | - | 默认 `8000`（中文长报告需要） |
| `AMAP_API_KEY` | - | 高德开放平台 | Itinerary Agent：POI 搜索 |
| `SENIVERSE_API_KEY` | - | 心知天气 | Safety Agent：天气查询 |
| `MOCK_LLM` | - | - | `true` 跳过所有外部调用 |

> **注意**：qwen3.8 系列必须走 DashScope OpenAI 兼容端点（`compatible-mode/v1`），旧端点对新模型返回 `url error`；思考模式强制开启的模型（如 qwen3.8-2.4t 系列）与本项目 `enable_thinking=false` 配置不兼容。

---

## 🧩 Agent 节点职责

| 节点 | 输入 | 真实工具 | 输出 |
|------|------|---------|------|
| **Coordinator** | 用户原始需求 | — | 结构化解析（目的地/日期/人数/预算/偏好，日期代码级钉正） |
| **Itinerary** | parsed_info | 高德地图 POI | 每日行程（上下午晚三段，附真实 POI 地址） |
| **Budget** | parsed_info | — | 预算估算表（算术校验：合计=人均×人数，错误自动覆盖修正） |
| **Safety** | parsed_info | 心知天气 | 安全评估（结合实时天气 + 预报） |
| **Review** | 三份子报告 | — | 四维评分（完整性/可行性/一致性/预算匹配）+ 三条交叉校验 |
| **Integrate** | 全部子报告 | — | 最终方案（Markdown + 摘要卡片） |
| **HITL** | 触发原因 | — | 暂停等待人工决策（approve / revise） |

---

## 📝 更新日志

### v0.2.0（2026-10）
- ✅ fan-out 并行化：三路 Agent 并行调度，全量生成端到端 ~84s（串行版 161s）
- ✅ 增量调整子图：语义路由 + 段内并行，仅重跑 4 节点、~48s
- ✅ Coordinator/Review 结构化输出（Function Calling + Pydantic）
- ✅ 预算总和/天数/日期区间三条代码级交叉校验 + 四维评审评分
- ✅ HITL 人工审核：修订超限/低分门控/节点超时自动转人工
- ✅ FastAPI + SSE + Vue3 前端（Agent 进度、评分条、历史会话）
- ✅ SQLite Checkpointer 跨会话持久化 + 启动时自动清理旧 checkpoint

### v0.1.0（2026-09）
- ✅ 六阶段 LangGraph MVP（串行版）
- ✅ 真实工具接入：高德地图 POI、心知天气
- ✅ 审核回退修订 + MAX_REVISION 上限保护
- ✅ TypedDict + Annotated reducer 状态管理

---

## 📄 License

MIT
