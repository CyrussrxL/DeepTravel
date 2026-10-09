# 🗺️ DeepTravel

基于 **LangGraph** 的多 Agent 旅行规划管家。

用户输入旅行需求后，Coordinator 结构化解析，行程先生成作为下游上下文，再段内并行调度预算与安全两个专业 Agent 调用真实数据工具，经代码级交叉校验与 LLM 四维评审双重把关，审核不通过自动回退修订，整合输出完整方案；支持增量调整（只重跑受影响环节）、HITL 人工审核与跨会话持久化。

---

## ✨ 核心特性

| 特性 | 说明 |
|------|------|
| 两阶段依赖调度 | `Coordinator → Itinerary → {Budget ∥ Safety} → Review → Integrate`，行程先生成作为下游上下文，预算/安全段内并行 |
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
                       │ 单跳（行程先生成，作为下游上下文）
                       ▼
                ┌──────────────┐
                │  itinerary   │ ◀── 高德 POI 每日行程
                └──────┬───────┘
                       │ 两路 fan-out（budget/safety 互不依赖，段内并行）
          ┌────────────┴───────────┐
          ▼                        ▼
    ┌──────────┐             ┌──────────┐
    │  budget  │             │  safety  │ ◀── 心知天气
    └────┬─────┘             └────┬─────┘
         └──────────┬───────────┘
                    ▼                ← fan-in
               ┌────────────┐
               │   review   │  四维评分 + 三条交叉校验
               └──┬─────┬───┘
          approve │     │ revise（≤ MAX_REVISION）
                  ▼     └──▶ 回退 itinerary（行程重生成后段内并行重跑）
          ┌────────────┐         │
          │  integrate │ ◀───────┘
          └─────┬──────┘
                ▼
               END

  review 门控未过 ──▶ HITL 人工审核（暂停等待，approve/revise 续跑）
  调整请求 ──▶ 增量子图（语义路由到最早受影响节点，段内并行重跑）
```

**依赖依据**：budget（按行程消费项拆分预算）与 safety（针对行程活动的安全提示）都依赖 itinerary 的输出作为上下文；budget 与 safety 之间互不依赖 → 段内并行。两阶段依赖消除了「预算/安全看不到行程」导致的脱节 revise 双轮。

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
| `DASHSCOPE_MODEL` | ✅ | 阿里云百炼 | 填入你开通的模型名（必填，无默认值） |
| `DASHSCOPE_BASE_URL` | - | - | 默认 DashScope OpenAI 兼容端点 |
| `DASHSCOPE_ENABLE_THINKING` | - | - | 默认 `false`（思考模式会占用 output token 预算） |
| `DASHSCOPE_MAX_TOKENS` | - | - | 默认 `8000`（中文长报告需要） |
| `AMAP_API_KEY` | - | 高德开放平台 | Itinerary Agent：POI 搜索 |
| `SENIVERSE_API_KEY` | - | 心知天气 | Safety Agent：天气查询 |
| `MOCK_LLM` | - | - | `true` 跳过所有外部调用 |

> **注意**：新版模型必须走 DashScope OpenAI 兼容端点（`compatible-mode/v1`），旧端点对新模型返回 `url error`；思考模式强制开启的模型与本项目 `enable_thinking=false` 配置不兼容，选型时请先确认该约束。

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

## 📊 基准实测数据

同输入重复基准（「去成都玩 2 天 2 人预算 5000」，每模型 10 次真实生成，qwen3.8-max 完整 10 次 / qwen3.7-flash 有效 9 次）：

| 指标 | 旗舰模型 | 轻量模型 |
|------|---------|---------|
| 单元级缺陷率（违规单元 / 可校验单元） | 3.1%（6/192） | 10.8%（8/74） |
| 会话级拦截率（触发交叉校验的会话占比） | 50% | 67% |
| 全量生成平均耗时 | 137s | 78s |
| 主要缺陷类型 | 预算总额算术偏差 | 跨报告日期漂移 / 越界日期 |

> 「单元级缺陷」指可被三条交叉校验确定性判定的事实错误（预算偏差 / 日期越界 / 天数不符）。
> 两类模型的缺陷恰好分别命中预算校验与日期校验防线，均被拦截于交付前，未流出至最终方案。

性能基准：

| 场景 | 耗时 |
|------|------|
| 串行版全量生成 | 161s |
| 三路并行全量生成（旧拓扑） | ~84s（-48%，已废弃） |
| 两阶段依赖 + 段内并行全量生成（当前） | 待额度恢复重测（估算 ~100s） |
| 增量调整（语义路由 + 段内并行，4 节点） | ~48s |

> 两阶段依赖改造的动机：三路并行时 budget/safety 看不到行程输出，导致预算与行程消费推荐脱节、安全提示无法针对具体活动，实测中触发 revise 双轮反而比两阶段更慢。两阶段拓扑让预算/安全基于行程输出生成，单轮通过率显著提高。

---

## 📝 更新日志

### v0.2.0（2026-10）
- ✅ 两阶段依赖 + 段内并行：`coordinator → itinerary → {budget ∥ safety} → review`，行程先生成作为下游上下文，消除预算/安全与行程脱节导致的 revise 双轮
- ✅ 增量调整子图：语义路由 + 段内并行，仅重跑 4 节点、~48s
- ✅ 预算联动路由：预算类调整请求路由至行程起点重跑，并同步覆盖 parsed_info 基准，避免「行程未变而预算已变」的评审误报
- ✅ 结构化日期注入：coordinator 解析出的行程日期作为唯一权威来源注入三个执行 Agent 的 prompt，消除跨报告日期漂移
- ✅ Coordinator/Review 结构化输出（Function Calling + Pydantic）
- ✅ 预算总和/天数/日期区间三条代码级交叉校验 + 四维评审评分；预算算术校验（合计=人均×人数）错误自动覆盖修正，LLM 输出缺失类别时的估算兜底数据显式标注来源
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
