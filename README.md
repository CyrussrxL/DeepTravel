# 🗺️ DeepTravel

基于 **LangGraph** 的多 Agent 旅行规划管家。

用户输入旅行需求后，系统自动路由到 **Coordinator → Itinerary → Budget → Safety → Review → Integrate** 六阶段有向图，
经审核整合输出完整方案；支持审核回退修订、真实 POI/天气数据查询，以及后续扩展的交互式调整子图。

---

## ✨ 核心特性

| 特性 | 说明 |
|------|------|
| 六阶段有向图 | `Coordinator → Itinerary → Budget → Safety → Review → Integrate` |
| 条件路由 | 节点只写 `state.next_node`，统一路由函数决定下一跳 |
| 审核回退 | `Review.revise` → 自动回退到 `Itinerary` 重跑，带最大修订次数保护 |
| 真实数据 | Itinerary 接入 **高德地图 POI**，Safety 接入 **心知天气** |
| 可降级 | 第三方 API 不通时自动退化为文本规划，流程不中断 |
| Mock 模式 | `MOCK_LLM=true` 跳过真实 LLM + API，秒级验证图路由 |
| TypedDict 状态 | 避免 LangGraph 状态合并异常，reducer 显式声明 |

---

## 🏗️ 架构

```
                     ┌──────────────┐
                     │  coordinator │ ◀── START
                     └──────┬───────┘
                            │
                            ▼
                     ┌──────────────┐
                ┌───▶│  itinerary   │◀─────┐
                │    └──────┬───────┘      │
                │           │              │ review=revise
                │           ▼              │
                │    ┌──────────────┐      │
                │    │    budget    │      │
                │    └──────┬───────┘      │
                │           │              │
                │           ▼              │
                │    ┌──────────────┐      │
                │    │    safety    │      │
                │    └──────┬───────┘      │
                │           │              │
                │           ▼              │
                │    ┌──────────────┐      │
                │    │    review    │──────┘
                │    └──┬───────┬───┘
                │       │       │
                │  approve   revise
                │       │       │
                │       ▼       │
                │ ┌────────────┐│
                │ │  integrate ││
                │ └─────┬──────┘│
                │       │       │
                │       ▼       │
                │     END       │
                │               │
                └───────────────┘
```

**路由原则**：每个节点只写 `state["next_node"]`，`route_decision(state)` 纯函数统一调度，**绝不**在同一节点混用 `add_edge` 和 `add_conditional_edges`。

---

## 📁 目录结构

```
DeepTravelV/
├── .env                  # 🔐 不要提交（含真实 API Key）
├── .env.example          # 配置模板
├── .gitignore
├── README.md
├── requirements.txt
└── src/
    ├── main.py           # CLI 入口（python -m src.main）
    ├── graph.py          # LangGraph 六阶段图编排
    ├── state.py          # TravelState TypedDict + reducers
    └── agents/
        ├── common.py     # LLM 初始化（ChatOpenAI + DashScope 兼容端点）
        ├── tools.py      # 高德地图 / 心知天气 / 飞猪 工具封装
        └── nodes.py      # 六个 Agent 节点函数
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
cp .env.example .env      # Windows: copy .env.example .env
# 然后编辑 .env 填入你的 Key
```

当前使用的 LLM：`qwen3.8-27b`（阿里云 DashScope OpenAI 兼容端点）

### 3. 运行

```bash
# Mock 模式（无 API，秒级验证图路由）
python -m src.main

# 真实 LLM + 真实 API
python -m src.main "我想去成都旅游3天2晚，两个人，预算5000元，喜欢大熊猫和美食"
```

---

## 🔌 API 配置说明

| 变量 | 提供者 | 用途 |
|------|--------|------|
| `DASHSCOPE_API_KEY` | 阿里云百炼 | 主 LLM（qwen3.8-27b） |
| `DASHSCOPE_MODEL` | 阿里云百炼 | 默认 `qwen3.8-27b` |
| `DASHSCOPE_BASE_URL` | 阿里云百炼 | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `AMAP_API_KEY` | 高德开放平台 | Itinerary Agent：国内城市 POI 搜索 |
| `SENIVERSE_API_KEY` | 心知天气 | Safety Agent：国内城市天气查询 |
| `FLIGGY_API_KEY` | 飞猪（预留） | 无公开 HTTP API，暂不可直连 |
| `MOCK_LLM` | - | `true` 跳过所有 LLM + API 调用 |

> **注意**：qwen3.8 系列新模型必须走 DashScope OpenAI 兼容端点（`compatible-mode/v1`），
> 旧 `text-generation/generation` 端点对新模型返回 `url error`。

---

## 🧩 Agent 节点职责

| 节点 | 输入 | 真实工具 | 输出 |
|------|------|---------|------|
| **Coordinator** | 用户原始需求 | — | 归一化需求摘要（目的地/日期/人数/预算/偏好） |
| **Itinerary** | 归一化需求 | 高德地图 POI 搜索 | 每日行程（上下午晚三段，附真实 POI 地址） |
| **Budget** | 行程 | — | 预算估算表（按交通/住宿/餐饮/门票/其他拆分） |
| **Safety** | 行程 | 心知天气实时 + 3日预报 | 安全评估（结合真实天气数据） |
| **Review** | 三份子报告 | — | `[APPROVED]` / `[REVISE]` + 审核意见 |
| **Integrate** | 全部子报告 + 审核 | — | 最终方案（Markdown + 摘要卡片） |

---

## 🔁 修订 & 终止保护

- `Review` 返回 `[REVISE]` 时自动回退到 `Itinerary`，重跑后续 `Budget → Safety → Review`
- `revision_count >= MAX_REVISION(2)` 时强制通过，防止无限循环
- 所有 LLM 调用和第三方 API 调用均有 try/catch 降级为 mock 输出，保证流程不中断

---

## 📋 Phase 2 规划

- [ ] **FastAPI 服务层**：将图暴露为 HTTP 接口 + SSE 流式推送执行进度
- [ ] **可视化前端**：Vue3 CDN 单页，聊天式输入 + 六 Agent 进度面板 + 最终方案 Markdown 渲染
- [ ] **调整子图**：独立子图实现规划后交互式修改，关键词分析 → 受影响 Agent 子集 → 局部重跑
- [ ] **持久化存储**：LangGraph Checkpointer + 历史方案检索
- [ ] **HITL 节点**：思考超限或低置信度 → 人工审核
- [ ] **MCP 协议工具集**：为专业 Agent 配置差异化 MCP 工具

---

## 📝 更新日志

### v0.1.0（2026-09-18）
- ✅ 六阶段 LangGraph MVP 框架搭建
- ✅ 真实 LLM：qwen3.8-27b（DashScope 兼容端点）
- ✅ 真实工具：高德地图 POI、心知天气
- ✅ 审核回退修订 + MAX_REVISION 上限保护
- ✅ Mock 模式端到端验证
- ✅ TypedDict + Annotated reducer 状态管理
