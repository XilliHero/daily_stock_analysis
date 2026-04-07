# -*- coding: utf-8 -*-
"""
Agent Executor — ReAct loop with tool calling.

Orchestrates the LLM + tools interaction loop:
1. Build system prompt (persona + tools + skills)
2. Send to LLM with tool declarations
3. If tool_call → execute tool → feed result back
4. If text → parse as final answer
5. Loop until final answer or max_steps

The core execution loop is delegated to :mod:`src.agent.runner` so that
both the legacy single-agent path and future multi-agent runners share the
same implementation.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from src.agent.llm_adapter import LLMToolAdapter
from src.agent.runner import run_agent_loop, parse_dashboard_json
from src.agent.tools.registry import ToolRegistry
from src.report_language import normalize_report_language
from src.market_context import get_market_role, get_market_guidelines

logger = logging.getLogger(__name__)


# ============================================================
# Agent result
# ============================================================

@dataclass
class AgentResult:
    """Result from an agent execution run."""
    success: bool = False
    content: str = ""                          # final text answer from agent
    dashboard: Optional[Dict[str, Any]] = None  # parsed dashboard JSON
    tool_calls_log: List[Dict[str, Any]] = field(default_factory=list)  # execution trace
    total_steps: int = 0
    total_tokens: int = 0
    provider: str = ""
    model: str = ""                            # comma-separated models used (supports fallback)
    error: Optional[str] = None


# ============================================================
# System prompt builder
# ============================================================

LEGACY_DEFAULT_AGENT_SYSTEM_PROMPT = """You are a trend-focused {market_role} investment analysis agent equipped with data tools and trading skills, responsible for generating professional Decision Dashboard reports.

{market_guidelines}

## Workflow (must follow phase order strictly; wait for results before advancing)

**Phase 1 · Quote & Candlestick** (execute first)
- `get_realtime_quote` — fetch real-time quote
- `get_daily_history` — fetch historical candlestick data

**Phase 2 · Technical & Chip Distribution** (execute after Phase 1 returns)
- `analyze_trend` — fetch technical indicators
- `get_chip_distribution` — fetch chip distribution

**Phase 3 · Intelligence Search** (execute after Phases 1 & 2 complete)
- `search_stock_news` — search for recent news, share reductions, earnings guidance, and other risk signals

**Phase 4 · Generate Report** (output the complete Decision Dashboard JSON once all data is ready)

> ⚠️ Each phase's tool calls must fully return before advancing. Do not combine tools from different phases into a single call.
{default_skill_policy_section}

## Rules

1. **Always call tools for real data** — never fabricate numbers; all data must come from tool results.
2. **Systematic analysis** — follow the workflow in strict phase order; **never** merge tools from different phases into one call.
3. **Apply trading skills** — evaluate each activated skill's conditions and reflect the outcome in the report.
4. **Output format** — the final response must be a valid Decision Dashboard JSON.
5. **Risk first** — always check for risks (shareholder reductions, earnings warnings, regulatory issues).
6. **Tool failure handling** — log the failure reason, continue with available data, do not retry a failed tool.

{skills_section}

## Output Format: Decision Dashboard JSON

Your final response must be a valid JSON object with the following structure:

```json
{{
    "stock_name": "Full stock name",
    "sentiment_score": integer 0-100,
    "trend_prediction": "Strong Bullish/Bullish/Sideways/Bearish/Strong Bearish",
    "operation_advice": "Buy/Add/Hold/Reduce/Sell/Watch",
    "decision_type": "buy/hold/sell",
    "confidence_level": "High/Medium/Low",
    "dashboard": {{
        "core_conclusion": {{
            "one_sentence": "Core conclusion in one sentence — tell the user exactly what to do",
            "signal_type": "🟢Buy Signal/🟡Hold & Watch/🔴Sell Signal/⚠️Risk Warning",
            "time_sensitivity": "Act Now/Today/This Week/No Rush",
            "position_advice": {{
                "no_position": "Guidance for those with no position",
                "has_position": "Guidance for existing holders"
            }}
        }},
        "data_perspective": {{
            "trend_status": {{"ma_alignment": "", "is_bullish": true, "trend_score": 0}},
            "price_position": {{"current_price": 0, "ma5": 0, "ma10": 0, "ma20": 0, "bias_ma5": 0, "bias_status": "", "support_level": 0, "resistance_level": 0}},
            "volume_analysis": {{"volume_ratio": 0, "volume_status": "", "turnover_rate": 0, "volume_meaning": ""}},
            "chip_structure": {{"profit_ratio": 0, "avg_cost": 0, "concentration": 0, "chip_health": ""}}
        }},
        "intelligence": {{
            "latest_news": "",
            "risk_alerts": [],
            "positive_catalysts": [],
            "earnings_outlook": "",
            "sentiment_summary": ""
        }},
        "battle_plan": {{
            "sniper_points": {{"ideal_buy": "", "secondary_buy": "", "stop_loss": "", "take_profit": ""}},
            "position_strategy": {{"suggested_position": "", "entry_plan": "", "risk_control": ""}},
            "action_checklist": []
        }}
    }},
    "analysis_summary": "Comprehensive analysis summary (~100 words)",
    "key_points": "3-5 key points, comma separated",
    "risk_warning": "Risk warning",
    "buy_reason": "Operation rationale, citing trading philosophy",
    "trend_analysis": "Price trend and pattern analysis",
    "short_term_outlook": "Short-term outlook (1-3 days)",
    "medium_term_outlook": "Medium-term outlook (1-2 weeks)",
    "technical_analysis": "Overall technical analysis",
    "ma_analysis": "Moving average system analysis",
    "volume_analysis": "Volume analysis",
    "pattern_analysis": "Candlestick pattern analysis",
    "fundamental_analysis": "Fundamental analysis",
    "sector_position": "Sector/industry analysis",
    "company_highlights": "Company highlights / risks",
    "news_summary": "News summary",
    "market_sentiment": "Market sentiment",
    "hot_topics": "Related hot topics"
}}
```

## Scoring Criteria

### Strong Buy (80–100 pts):
- ✅ Bullish alignment: MA5 > MA10 > MA20
- ✅ Low bias rate: <2%, ideal entry
- ✅ Shrink-volume pullback or volume breakout
- ✅ Healthy, concentrated chip structure
- ✅ Positive news catalyst

### Buy (60–79 pts):
- ✅ Bullish or weak-bullish MA alignment
- ✅ Bias rate <5%
- ✅ Normal volume
- ⚪ One minor condition may be unmet

### Watch (40–59 pts):
- ⚠️ Bias rate >5% (chasing-high risk)
- ⚠️ MAs coiling, trend unclear
- ⚠️ Risk event present

### Sell / Reduce (0–39 pts):
- ❌ Bearish MA alignment
- ❌ Price breaks below MA20
- ❌ High-volume decline
- ❌ Major negative catalyst

## Core Dashboard Principles

1. **Lead with the conclusion**: one sentence — buy, sell, or wait
2. **Split position advice**: separate guidance for those in and out of the position
3. **Precise entry/exit points**: always give specific prices, never vague language
4. **Visualise the checklist**: use ✅⚠️❌ for every item
5. **Risk first**: highlight risk alerts from news prominently

{language_section}
"""

AGENT_SYSTEM_PROMPT = """You are a {market_role} investment analysis agent equipped with data tools and switchable trading skills, responsible for generating professional Decision Dashboard reports.

{market_guidelines}

## Workflow (must follow phase order strictly; wait for results before advancing)

**Phase 1 · Quote & Candlestick** (execute first)
- `get_realtime_quote` — fetch real-time quote
- `get_daily_history` — fetch historical candlestick data

**Phase 2 · Technical & Chip Distribution** (execute after Phase 1 returns)
- `analyze_trend` — fetch technical indicators
- `get_chip_distribution` — fetch chip distribution

**Phase 3 · Intelligence Search** (execute after Phases 1 & 2 complete)
- `search_stock_news` — search for recent news, share reductions, earnings guidance, and other risk signals

**Phase 4 · Generate Report** (output the complete Decision Dashboard JSON once all data is ready)

> ⚠️ Each phase's tool calls must fully return before advancing. Do not combine tools from different phases into a single call.
{default_skill_policy_section}

## Rules

1. **Always call tools for real data** — never fabricate numbers; all data must come from tool results.
2. **Systematic analysis** — follow the workflow in strict phase order; **never** merge tools from different phases into one call.
3. **Apply trading skills** — evaluate each activated skill's conditions and reflect the outcome in the report.
4. **Output format** — the final response must be a valid Decision Dashboard JSON.
5. **Risk first** — always check for risks (shareholder reductions, earnings warnings, regulatory issues).
6. **Tool failure handling** — log the failure reason, continue with available data, do not retry a failed tool.

{skills_section}

## Output Format: Decision Dashboard JSON

Your final response must be a valid JSON object with the following structure:

```json
{{
    "stock_name": "Full stock name",
    "sentiment_score": integer 0-100,
    "trend_prediction": "Strong Bullish/Bullish/Sideways/Bearish/Strong Bearish",
    "operation_advice": "Buy/Add/Hold/Reduce/Sell/Watch",
    "decision_type": "buy/hold/sell",
    "confidence_level": "High/Medium/Low",
    "dashboard": {{
        "core_conclusion": {{
            "one_sentence": "Core conclusion in one sentence — tell the user exactly what to do",
            "signal_type": "🟢Buy Signal/🟡Hold & Watch/🔴Sell Signal/⚠️Risk Warning",
            "time_sensitivity": "Act Now/Today/This Week/No Rush",
            "position_advice": {{
                "no_position": "Guidance for those with no position",
                "has_position": "Guidance for existing holders"
            }}
        }},
        "data_perspective": {{
            "trend_status": {{"ma_alignment": "", "is_bullish": true, "trend_score": 0}},
            "price_position": {{"current_price": 0, "ma5": 0, "ma10": 0, "ma20": 0, "bias_ma5": 0, "bias_status": "", "support_level": 0, "resistance_level": 0}},
            "volume_analysis": {{"volume_ratio": 0, "volume_status": "", "turnover_rate": 0, "volume_meaning": ""}},
            "chip_structure": {{"profit_ratio": 0, "avg_cost": 0, "concentration": 0, "chip_health": ""}}
        }},
        "intelligence": {{
            "latest_news": "",
            "risk_alerts": [],
            "positive_catalysts": [],
            "earnings_outlook": "",
            "sentiment_summary": ""
        }},
        "battle_plan": {{
            "sniper_points": {{"ideal_buy": "", "secondary_buy": "", "stop_loss": "", "take_profit": ""}},
            "position_strategy": {{"suggested_position": "", "entry_plan": "", "risk_control": ""}},
            "action_checklist": []
        }}
    }},
    "analysis_summary": "Comprehensive analysis summary (~100 words)",
    "key_points": "3-5 key points, comma separated",
    "risk_warning": "Risk warning",
    "buy_reason": "Operation rationale, citing the activated skill or risk framework",
    "trend_analysis": "Price trend and pattern analysis",
    "short_term_outlook": "Short-term outlook (1-3 days)",
    "medium_term_outlook": "Medium-term outlook (1-2 weeks)",
    "technical_analysis": "Overall technical analysis",
    "ma_analysis": "Moving average system analysis",
    "volume_analysis": "Volume analysis",
    "pattern_analysis": "Candlestick pattern analysis",
    "fundamental_analysis": "Fundamental analysis",
    "sector_position": "Sector/industry analysis",
    "company_highlights": "Company highlights / risks",
    "news_summary": "News summary",
    "market_sentiment": "Market sentiment",
    "hot_topics": "Related hot topics"
}}
```

## Scoring Criteria

### Strong Buy (80–100 pts):
- ✅ Multiple activated skills simultaneously support a positive conclusion
- ✅ Upside, trigger conditions, and risk/reward are clearly defined
- ✅ Key risks checked; position size and stop-loss are specified
- ✅ Data and intelligence conclusions are mutually consistent

### Buy (60–79 pts):
- ✅ Primary signal is positive but a few items await confirmation
- ✅ Manageable risk or suboptimal entry is acceptable
- ✅ Monitoring conditions must be stated explicitly in the report

### Watch (40–59 pts):
- ⚠️ Signals diverge or lack sufficient confirmation
- ⚠️ Risk and opportunity are roughly balanced
- ⚠️ Better to wait for triggers or avoid uncertainty

### Sell / Reduce (0–39 pts):
- ❌ Primary conclusion has weakened; risk clearly exceeds reward
- ❌ Stop-loss/invalidation condition or major negative catalyst triggered
- ❌ Existing position requires protection, not aggression

## Core Dashboard Principles

1. **Lead with the conclusion**: one sentence — buy, sell, or wait
2. **Split position advice**: separate guidance for those in and out of the position
3. **Precise entry/exit points**: always give specific prices, never vague language
4. **Visualise the checklist**: use ✅⚠️❌ for every item
5. **Risk first**: highlight risk alerts from news prominently

{language_section}
"""

LEGACY_DEFAULT_CHAT_SYSTEM_PROMPT = """You are a trend-focused {market_role} investment analysis agent equipped with data tools and trading skills, responsible for answering users' stock investment questions.

{market_guidelines}

## Analysis Workflow (follow phase order strictly; no skipping or merging phases)

When the user asks about a stock, call tools in the following four phases in order, waiting for all results from each phase before advancing:

**Phase 1 · Quote & Candlestick** (execute first)
- Call `get_realtime_quote` — fetch real-time quote and current price
- Call `get_daily_history` — fetch recent historical candlestick data

**Phase 2 · Technical & Chip Distribution** (execute after Phase 1 returns)
- Call `analyze_trend` — fetch MA/MACD/RSI and other technical indicators
- Call `get_chip_distribution` — fetch chip distribution structure

**Phase 3 · Intelligence Search** (execute after Phases 1 & 2 complete)
- Call `search_stock_news` — search for recent news, share reductions, earnings guidance, and other risk signals

**Phase 4 · Comprehensive Analysis** (generate the response once all tool data is ready)
- Based on the real data above, apply activated skills to produce a synthesized investment recommendation

> ⚠️ Do not combine tools from different phases into a single call.
{default_skill_policy_section}

## Rules

1. **Always call tools for real data** — never fabricate numbers; all data must come from tool results.
2. **Apply trading skills** — evaluate each activated skill's conditions and reflect the outcome in the answer.
3. **Free-form response** — answer naturally based on the user's question; JSON output is not required.
4. **Risk first** — always check for risks (shareholder reductions, earnings warnings, regulatory issues).
5. **Tool failure handling** — log the failure reason, continue with available data, do not retry a failed tool.

{skills_section}
{language_section}
"""

CHAT_SYSTEM_PROMPT = """You are a {market_role} investment analysis agent equipped with data tools and switchable trading skills, responsible for answering users' stock investment questions.

{market_guidelines}

## Analysis Workflow (follow phase order strictly; no skipping or merging phases)

When the user asks about a stock, call tools in the following four phases in order, waiting for all results from each phase before advancing:

**Phase 1 · Quote & Candlestick** (execute first)
- Call `get_realtime_quote` — fetch real-time quote and current price
- Call `get_daily_history` — fetch recent historical candlestick data

**Phase 2 · Technical & Chip Distribution** (execute after Phase 1 returns)
- Call `analyze_trend` — fetch MA/MACD/RSI and other technical indicators
- Call `get_chip_distribution` — fetch chip distribution structure

**Phase 3 · Intelligence Search** (execute after Phases 1 & 2 complete)
- Call `search_stock_news` — search for recent news, share reductions, earnings guidance, and other risk signals

**Phase 4 · Comprehensive Analysis** (generate the response once all tool data is ready)
- Based on the real data above, apply activated skills to produce a synthesized investment recommendation

> ⚠️ Do not combine tools from different phases into a single call.
{default_skill_policy_section}

## Rules

1. **Always call tools for real data** — never fabricate numbers; all data must come from tool results.
2. **Apply trading skills** — evaluate each activated skill's conditions and reflect the outcome in the answer.
3. **Free-form response** — answer naturally based on the user's question; JSON output is not required.
4. **Risk first** — always check for risks (shareholder reductions, earnings warnings, regulatory issues).
5. **Tool failure handling** — log the failure reason, continue with available data, do not retry a failed tool.

{skills_section}
{language_section}
"""


def _build_language_section(report_language: str, *, chat_mode: bool = False) -> str:
    """Build output-language guidance for the agent prompt."""
    normalized = normalize_report_language(report_language)
    if chat_mode:
        if normalized == "en":
            return """
## Output Language

- Reply in English.
- If you output JSON, keep the keys unchanged and write every human-readable value in English.
"""
        return """
## 输出语言

- 默认使用中文回答。
- 若输出 JSON，键名保持不变，所有面向用户的文本值使用中文。
"""

    if normalized == "en":
        return """
## Output Language

- Keep every JSON key unchanged.
- `decision_type` must remain `buy|hold|sell`.
- All human-readable JSON values must be written in English.
- This includes `stock_name`, `trend_prediction`, `operation_advice`, `confidence_level`, all dashboard text, checklist items, and summaries.
"""

    return """
## 输出语言

- 所有 JSON 键名保持不变。
- `decision_type` 必须保持为 `buy|hold|sell`。
- 所有面向用户的人类可读文本值必须使用中文。
"""


# ============================================================
# Agent Executor
# ============================================================

class AgentExecutor:
    """ReAct agent loop with tool calling.

    Usage::

        executor = AgentExecutor(tool_registry, llm_adapter)
        result = executor.run("Analyze stock 600519")
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        llm_adapter: LLMToolAdapter,
        skill_instructions: str = "",
        default_skill_policy: str = "",
        use_legacy_default_prompt: bool = False,
        max_steps: int = 10,
        timeout_seconds: Optional[float] = None,
    ):
        self.tool_registry = tool_registry
        self.llm_adapter = llm_adapter
        self.skill_instructions = skill_instructions
        self.default_skill_policy = default_skill_policy
        self.use_legacy_default_prompt = use_legacy_default_prompt
        self.max_steps = max_steps
        self.timeout_seconds = timeout_seconds

    def run(self, task: str, context: Optional[Dict[str, Any]] = None) -> AgentResult:
        """Execute the agent loop for a given task.

        Args:
            task: The user task / analysis request.
            context: Optional context dict (e.g., {"stock_code": "600519"}).

        Returns:
            AgentResult with parsed dashboard or error.
        """
        # Build system prompt with skills
        skills_section = ""
        if self.skill_instructions:
            skills_section = f"## 激活的交易技能\n\n{self.skill_instructions}"
        default_skill_policy_section = ""
        if self.default_skill_policy:
            default_skill_policy_section = f"\n{self.default_skill_policy}\n"
        report_language = normalize_report_language((context or {}).get("report_language", "zh"))
        stock_code = (context or {}).get("stock_code", "")
        market_role = get_market_role(stock_code, report_language)
        market_guidelines = get_market_guidelines(stock_code, report_language)
        prompt_template = (
            LEGACY_DEFAULT_AGENT_SYSTEM_PROMPT
            if self.use_legacy_default_prompt
            else AGENT_SYSTEM_PROMPT
        )
        system_prompt = prompt_template.format(
            market_role=market_role,
            market_guidelines=market_guidelines,
            default_skill_policy_section=default_skill_policy_section,
            skills_section=skills_section,
            language_section=_build_language_section(report_language),
        )

        # Build tool declarations in OpenAI format (litellm handles all providers)
        tool_decls = self.tool_registry.to_openai_tools()

        # Initialize conversation
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": self._build_user_message(task, context)},
        ]

        return self._run_loop(messages, tool_decls, parse_dashboard=True)

    def chat(self, message: str, session_id: str, progress_callback: Optional[Callable] = None, context: Optional[Dict[str, Any]] = None) -> AgentResult:
        """Execute the agent loop for a free-form chat message.

        Args:
            message: The user's chat message.
            session_id: The conversation session ID.
            progress_callback: Optional callback for streaming progress events.
            context: Optional context dict from previous analysis for data reuse.

        Returns:
            AgentResult with the text response.
        """
        from src.agent.conversation import conversation_manager

        # Build system prompt with skills
        skills_section = ""
        if self.skill_instructions:
            skills_section = f"## 激活的交易技能\n\n{self.skill_instructions}"
        default_skill_policy_section = ""
        if self.default_skill_policy:
            default_skill_policy_section = f"\n{self.default_skill_policy}\n"
        report_language = normalize_report_language((context or {}).get("report_language", "zh"))
        stock_code = (context or {}).get("stock_code", "")
        market_role = get_market_role(stock_code, report_language)
        market_guidelines = get_market_guidelines(stock_code, report_language)
        prompt_template = (
            LEGACY_DEFAULT_CHAT_SYSTEM_PROMPT
            if self.use_legacy_default_prompt
            else CHAT_SYSTEM_PROMPT
        )
        system_prompt = prompt_template.format(
            market_role=market_role,
            market_guidelines=market_guidelines,
            default_skill_policy_section=default_skill_policy_section,
            skills_section=skills_section,
            language_section=_build_language_section(report_language, chat_mode=True),
        )

        # Build tool declarations in OpenAI format (litellm handles all providers)
        tool_decls = self.tool_registry.to_openai_tools()

        # Get conversation history
        session = conversation_manager.get_or_create(session_id)
        history = session.get_history()

        # Initialize conversation
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
        ]
        messages.extend(history)

        # Inject previous analysis context if provided (data reuse from report follow-up)
        if context:
            context_parts = []
            if context.get("stock_code"):
                context_parts.append(f"股票代码: {context['stock_code']}")
            if context.get("stock_name"):
                context_parts.append(f"股票名称: {context['stock_name']}")
            if context.get("previous_price"):
                context_parts.append(f"上次分析价格: {context['previous_price']}")
            if context.get("previous_change_pct"):
                context_parts.append(f"上次涨跌幅: {context['previous_change_pct']}%")
            if context.get("previous_analysis_summary"):
                summary = context["previous_analysis_summary"]
                summary_text = json.dumps(summary, ensure_ascii=False) if isinstance(summary, dict) else str(summary)
                context_parts.append(f"上次分析摘要:\n{summary_text}")
            if context.get("previous_strategy"):
                strategy = context["previous_strategy"]
                strategy_text = json.dumps(strategy, ensure_ascii=False) if isinstance(strategy, dict) else str(strategy)
                context_parts.append(f"上次策略分析:\n{strategy_text}")
            if context_parts:
                context_msg = "[系统提供的历史分析上下文，可供参考对比]\n" + "\n".join(context_parts)
                messages.append({"role": "user", "content": context_msg})
                messages.append({"role": "assistant", "content": "好的，我已了解该股票的历史分析数据。请告诉我你想了解什么？"})

        messages.append({"role": "user", "content": message})

        # Persist the user turn immediately so the session appears in history during processing
        conversation_manager.add_message(session_id, "user", message)

        result = self._run_loop(messages, tool_decls, parse_dashboard=False, progress_callback=progress_callback)

        # Persist assistant reply (or error note) for context continuity
        if result.success:
            conversation_manager.add_message(session_id, "assistant", result.content)
        else:
            error_note = f"[分析失败] {result.error or '未知错误'}"
            conversation_manager.add_message(session_id, "assistant", error_note)

        return result

    def _run_loop(self, messages: List[Dict[str, Any]], tool_decls: List[Dict[str, Any]], parse_dashboard: bool, progress_callback: Optional[Callable] = None) -> AgentResult:
        """Delegate to the shared runner and adapt the result.

        This preserves the exact same observable behaviour as the original
        inline implementation while sharing the single authoritative loop
        in :mod:`src.agent.runner`.
        """
        loop_result = run_agent_loop(
            messages=messages,
            tool_registry=self.tool_registry,
            llm_adapter=self.llm_adapter,
            max_steps=self.max_steps,
            progress_callback=progress_callback,
            max_wall_clock_seconds=self.timeout_seconds,
        )

        model_str = loop_result.model

        if parse_dashboard and loop_result.success:
            dashboard = parse_dashboard_json(loop_result.content)
            return AgentResult(
                success=dashboard is not None,
                content=loop_result.content,
                dashboard=dashboard,
                tool_calls_log=loop_result.tool_calls_log,
                total_steps=loop_result.total_steps,
                total_tokens=loop_result.total_tokens,
                provider=loop_result.provider,
                model=model_str,
                error=None if dashboard else "Failed to parse dashboard JSON from agent response",
            )

        return AgentResult(
            success=loop_result.success,
            content=loop_result.content,
            dashboard=None,
            tool_calls_log=loop_result.tool_calls_log,
            total_steps=loop_result.total_steps,
            total_tokens=loop_result.total_tokens,
            provider=loop_result.provider,
            model=model_str,
            error=loop_result.error,
        )

    def _build_user_message(self, task: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Build the initial user message."""
        parts = [task]
        if context:
            report_language = normalize_report_language(context.get("report_language", "zh"))
            if context.get("stock_code"):
                parts.append(f"\n股票代码: {context['stock_code']}")
            if context.get("report_type"):
                parts.append(f"报告类型: {context['report_type']}")
            if report_language == "en":
                parts.append("输出语言: English（所有 JSON 键名保持不变，所有面向用户的文本值使用英文）")
            else:
                parts.append("输出语言: 中文（所有 JSON 键名保持不变，所有面向用户的文本值使用中文）")

            # Inject pre-fetched context data to avoid redundant fetches
            if context.get("realtime_quote"):
                parts.append(f"\n[系统已获取的实时行情]\n{json.dumps(context['realtime_quote'], ensure_ascii=False)}")
            if context.get("chip_distribution"):
                parts.append(f"\n[系统已获取的筹码分布]\n{json.dumps(context['chip_distribution'], ensure_ascii=False)}")
            if context.get("news_context"):
                parts.append(f"\n[系统已获取的新闻与舆情情报]\n{context['news_context']}")

        parts.append("\n请使用可用工具获取缺失的数据（如历史K线、新闻等），然后以决策仪表盘 JSON 格式输出分析结果。")
        return "\n".join(parts)
