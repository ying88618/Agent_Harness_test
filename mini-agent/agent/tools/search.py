"""搜索工具（mock）。

用本地语料 + 关键词打分模拟检索。虽然是 mock，但故意返回结构化结果
（title / snippet / url / score），因为模型对"带出处的条目"引用得更准，
这也和真实搜索 API 的返回形态一致，后面换真接口不用改 prompt。
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..registry import tool
from ..types import ToolContext

_CORPUS: list[dict[str, Any]] = [
    {
        "title": "ReAct: Synergizing Reasoning and Acting in Language Models",
        "snippet": "提出让大模型交替产生推理轨迹（Reasoning）与动作（Acting），"
        "用工具返回的观察结果修正下一步推理，是当前 Agent 循环的主流范式。",
        "url": "https://arxiv.org/abs/2210.03629",
        "keywords": ["react", "agent", "推理", "工具调用", "循环", "loop"],
    },
    {
        "title": "Function Calling 与工具 Schema 设计",
        "snippet": "工具描述要写清楚「什么时候用」而不只是「是什么」。"
        "参数 Schema 越窄，模型幻觉调用越少；必填项应尽量少。",
        "url": "https://platform.openai.com/docs/guides/function-calling",
        "keywords": ["function calling", "工具", "schema", "tool", "注册"],
    },
    {
        "title": "Context Window 管理与对话压缩",
        "snippet": "长对话中应保留最近若干轮原文，更早的轮次折叠成摘要；"
        "裁剪时必须保持 assistant tool_calls 与 tool 消息成对出现，否则请求会被拒绝。",
        "url": "https://example.com/context-management",
        "keywords": ["context", "上下文", "压缩", "摘要", "token", "窗口"],
    },
    {
        "title": "多轮会话与 Session 隔离",
        "snippet": "同一个用户的不同窗口应各自维护消息列表与会话级状态，"
        "按 session_id 分片存储；跨进程续聊需要把状态落盘。",
        "url": "https://example.com/session-design",
        "keywords": ["session", "会话", "隔离", "窗口", "多用户"],
    },
    {
        "title": "Agent 可观测性：Trace 与执行日志",
        "snippet": "以 JSONL 逐事件记录 LLM 调用、工具调用与耗时，"
        "trace 与 context 分离——日志是给人看的，不能污染模型上下文。",
        "url": "https://example.com/agent-trace",
        "keywords": ["trace", "日志", "可观测", "调试", "jsonl"],
    },
    {
        "title": "2026 年华南地区气候概况",
        "snippet": "广州、厦门年平均气温 22℃ 左右，5-9 月为雨季，"
        "台风季集中在 7-9 月，湿度常年偏高。",
        "url": "https://example.com/climate",
        "keywords": ["天气", "气候", "广州", "厦门", "气温", "雨季"],
    },
    {
        "title": "如何写一份有效的周报",
        "snippet": "结构建议：本周完成（带量化结果）→ 遇到的问题与决策 → 下周计划。"
        "拒绝流水账，每条结论都要能对应到具体产出。",
        "url": "https://example.com/weekly-report",
        "keywords": ["周报", "汇报", "总结", "工作"],
    },
]

_SPLIT_RE = re.compile(r"[\s,，、;；/]+")


@tool(
    name="search",
    description="检索资料（本地知识库，模拟搜索引擎）。"
    "需要查事实、找定义、找参考链接时使用。返回带标题、摘要和链接的条目。",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "检索关键词，尽量简短具体"},
            "top_k": {
                "type": "integer",
                "description": "最多返回几条，默认 3，范围 1-5",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
)
def search(args: dict[str, Any], ctx: ToolContext) -> str:
    query = str(args.get("query", "")).strip()
    if not query:
        return "查询词为空，请提供要检索的关键词。"

    top_k = args.get("top_k", 3)
    try:
        top_k = max(1, min(int(top_k), 5))
    except (TypeError, ValueError):
        top_k = 3

    tokens = [t.lower() for t in _SPLIT_RE.split(query) if t]
    scored: list[tuple[float, dict[str, Any]]] = []

    for doc in _CORPUS:
        score = 0.0
        title = doc["title"].lower()
        snippet = doc["snippet"].lower()
        keywords = [k.lower() for k in doc["keywords"]]

        if query.lower() in title:
            score += 5
        for token in tokens:
            if token in title:
                score += 3
            if token in keywords:
                score += 2
            elif any(token in k for k in keywords):
                score += 1
            if token in snippet:
                score += 1
        if score > 0:
            scored.append((score, doc))

    if not scored:
        return (
            f"没有找到与「{query}」相关的资料。"
            "可以换更通用的关键词，或者直接基于已有信息回答用户。"
        )

    scored.sort(key=lambda x: (-x[0], x[1]["title"]))
    results = [
        {
            "title": doc["title"],
            "snippet": doc["snippet"],
            "url": doc["url"],
            "score": round(score, 1),
        }
        for score, doc in scored[:top_k]
    ]
    ctx.log("search_hit", query=query, hits=len(results))
    return json.dumps(results, ensure_ascii=False, indent=2)
