import json
from collections.abc import Generator
from typing import Any, cast

from exo.shared.types.common import ModelId
from exo.shared.types.worker.runner_response import (
    GenerationResponse,
    ToolCallResponse,
)
from exo.worker.engines.mlx.vendor.dsml_encoding import (
    ASSISTANT_TOKEN,
    BOS_TOKEN,
    DSML_TOKEN,
    EOS_TOKEN,
    THINKING_END,
    THINKING_START,
    TOOL_CALLS_END,
    TOOL_CALLS_START,
    USER_TOKEN,
    encode_messages,
    parse_dsml_output,
)
from exo.worker.runner.llm_inference.model_output_parsers import (
    parse_deepseek_v4,
    parse_deepseek_v32,
    parse_thinking_models,
)


def _parse_deepseek_with_thinking(
    source: Generator[GenerationResponse | None],
    starts_in_thinking: bool = False,
) -> Generator[GenerationResponse | ToolCallResponse | None]:
    return parse_deepseek_v32(
        parse_thinking_models(
            source,
            think_start=THINKING_START,
            think_end=THINKING_END,
            starts_in_thinking=starts_in_thinking,
        )
    )


# ── Shared fixtures ──────────────────────────────────────────────

_WEATHER_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a given city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "The city name"},
                    "units": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature units",
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current time in a timezone",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {"type": "string"},
                },
                "required": ["timezone"],
            },
        },
    },
]


def _simulate_tokens(
    texts: list[str],
    finish_on_last: bool = True,
) -> Generator[GenerationResponse]:
    """Simulate a model producing tokens from a list of text strings."""
    for i, text in enumerate(texts):
        is_last = i == len(texts) - 1
        yield GenerationResponse(
            text=text,
            token=i,
            finish_reason="stop" if (is_last and finish_on_last) else None,
            usage=None,
        )


# ── Test: Standard text response (no tool calls) ────────────────


class TestE2EStandardResponse:
    """Model generates a plain text response — no tool calling involved."""

    def test_plain_text_passthrough(self):
        """Simulate model producing: 'The weather in NYC is 72°F and sunny.'"""
        # Step 1: Encode the prompt (with tools available)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the weather in NYC?"},
        ]
        prompt = encode_messages(messages, thinking_mode="chat", tools=_WEATHER_TOOLS)

        # Verify prompt structure
        assert BOS_TOKEN in prompt
        assert "## Tools" in prompt
        assert "get_weather" in prompt
        assert f"{USER_TOKEN}What's the weather in NYC?{ASSISTANT_TOKEN}" in prompt

        # Step 2: Simulate model response — plain text tokens (no DSML)
        model_tokens = [
            "The weather",
            " in NYC",
            " is 72",
            "°F",
            " and sunny",
            ".",
        ]
        results = list(parse_deepseek_v32(_simulate_tokens(model_tokens)))

        # Step 3: Verify all tokens pass through as GenerationResponse
        gen_results = [r for r in results if isinstance(r, GenerationResponse)]
        tool_results = [r for r in results if isinstance(r, ToolCallResponse)]

        assert len(tool_results) == 0
        assert len(gen_results) == 6
        full_text = "".join(r.text for r in gen_results)
        assert full_text == "The weather in NYC is 72°F and sunny."
        assert gen_results[-1].finish_reason == "stop"


# ── Test: Tool call response ─────────────────────────────────────


class TestE2EToolCallResponse:
    """Model generates a DSML tool call — realistic token boundaries."""

    def test_realistic_tool_call_tokens(self):
        """Simulate model generating a get_weather tool call with realistic token splits.

        Real models split DSML markers across tokens unpredictably.
        This simulates how DeepSeek V3.2 actually tokenizes DSML output.
        """
        # Step 1: Encode prompt
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the weather in San Francisco?"},
        ]
        prompt = encode_messages(messages, thinking_mode="chat", tools=_WEATHER_TOOLS)
        assert "get_weather" in prompt

        # Step 2: Simulate realistic token-by-token model output
        # The model first produces some text, then a DSML tool call block
        model_tokens = [
            "I'll check the weather for you.",
            "\n\n",
            f"<{DSML_TOKEN}",  # marker split across tokens
            "function_calls>\n",
            f'<{DSML_TOKEN}invoke name="get_weather">\n',
            f'<{DSML_TOKEN}parameter name="city" string="true">',
            "San Francisco",
            f"</{DSML_TOKEN}parameter>\n",
            f'<{DSML_TOKEN}parameter name="units" string="false">',
            '"celsius"',
            f"</{DSML_TOKEN}parameter>\n",
            f"</{DSML_TOKEN}invoke>\n",
            f"</{DSML_TOKEN}function_calls>",
        ]

        results = list(parse_deepseek_v32(_simulate_tokens(model_tokens)))

        # Step 3: Verify
        gen_results = [r for r in results if isinstance(r, GenerationResponse)]
        tool_results = [r for r in results if isinstance(r, ToolCallResponse)]

        # Should have text tokens before tool call + one ToolCallResponse
        assert len(tool_results) == 1
        assert len(tool_results[0].tool_calls) == 1

        tc = tool_results[0].tool_calls[0]
        assert tc.name == "get_weather"
        args = json.loads(tc.arguments)  # pyright: ignore[reportAny]
        assert args["city"] == "San Francisco"
        assert args["units"] == "celsius"

        # The text before the tool call should still be yielded
        text_before = "".join(r.text for r in gen_results if not r.is_thinking)
        assert "check the weather" in text_before

    def test_multiple_tool_calls_in_one_block(self):
        """Model generates two tool calls in a single function_calls block."""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Weather in NYC and time in EST?"},
        ]
        prompt = encode_messages(messages, thinking_mode="chat", tools=_WEATHER_TOOLS)
        assert "get_weather" in prompt
        assert "get_time" in prompt

        # Simulate model output with two invocations
        model_tokens = [
            "Let me check both.\n\n",
            TOOL_CALLS_START,
            "\n",
            f'<{DSML_TOKEN}invoke name="get_weather">\n',
            f'<{DSML_TOKEN}parameter name="city" string="true">NYC</{DSML_TOKEN}parameter>\n',
            f"</{DSML_TOKEN}invoke>\n",
            f'<{DSML_TOKEN}invoke name="get_time">\n',
            f'<{DSML_TOKEN}parameter name="timezone" string="true">EST</{DSML_TOKEN}parameter>\n',
            f"</{DSML_TOKEN}invoke>\n",
            TOOL_CALLS_END,
        ]

        results = list(parse_deepseek_v32(_simulate_tokens(model_tokens)))
        tool_results = [r for r in results if isinstance(r, ToolCallResponse)]

        assert len(tool_results) == 1
        assert len(tool_results[0].tool_calls) == 2
        assert tool_results[0].tool_calls[0].name == "get_weather"
        assert tool_results[0].tool_calls[1].name == "get_time"

        args0 = json.loads(tool_results[0].tool_calls[0].arguments)  # pyright: ignore[reportAny]
        args1 = json.loads(tool_results[0].tool_calls[1].arguments)  # pyright: ignore[reportAny]
        assert args0 == {"city": "NYC"}
        assert args1 == {"timezone": "EST"}


# ── Test: Multi-turn tool use flow ───────────────────────────────


class TestE2EMultiTurnToolUse:
    """Full multi-turn: user asks → model calls tool → tool result → model answers."""

    def test_encode_multi_turn_with_tool_results(self):
        """Verify the prompt for turn 2 (after tool results) is correctly encoded."""
        # Turn 1: user asks, model calls tool
        # Turn 2: tool result provided, model answers
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "You are a weather assistant."},
            {"role": "user", "content": "What's the weather in NYC?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "NYC"}',
                        },
                    }
                ],
            },
            {"role": "tool", "content": '{"temperature": 72, "condition": "sunny"}'},
        ]

        prompt = encode_messages(messages, thinking_mode="chat", tools=_WEATHER_TOOLS)

        # Verify multi-turn structure
        assert BOS_TOKEN in prompt
        assert "You are a weather assistant." in prompt
        assert "## Tools" in prompt

        # The assistant's tool call should be encoded as DSML
        assert TOOL_CALLS_START in prompt
        assert f'<{DSML_TOKEN}invoke name="get_weather">' in prompt
        assert EOS_TOKEN in prompt

        # The tool result should be wrapped in function_results
        assert "<function_results>" in prompt
        assert "<result>" in prompt
        assert "72" in prompt
        assert "</function_results>" in prompt

        # Now simulate model answering after seeing the tool result
        model_tokens = [
            "The current",
            " weather in NYC",
            " is 72°F",
            " and sunny.",
        ]
        results = list(parse_deepseek_v32(_simulate_tokens(model_tokens)))

        gen_results = [r for r in results if isinstance(r, GenerationResponse)]
        tool_results = [r for r in results if isinstance(r, ToolCallResponse)]

        assert len(tool_results) == 0
        full_text = "".join(r.text for r in gen_results)
        assert full_text == "The current weather in NYC is 72°F and sunny."

    def test_multi_tool_results_encoding(self):
        """Verify encoding when model called two tools and both return results."""
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Weather and time?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "LA"}',
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "get_time",
                            "arguments": '{"timezone": "PST"}',
                        },
                    },
                ],
            },
            {"role": "tool", "content": "85F, clear skies"},
            {"role": "tool", "content": "3:42 PM PST"},
        ]

        prompt = encode_messages(messages, thinking_mode="chat", tools=_WEATHER_TOOLS)

        # Should have one function_results block with two results
        assert prompt.count("<function_results>") == 1
        assert prompt.count("</function_results>") == 1
        assert "<result>85F, clear skies</result>" in prompt
        assert "<result>3:42 PM PST</result>" in prompt


# ── Test: Thinking + tool call ───────────────────────────────────


class TestE2EThinkingAndToolCall:
    """Model uses thinking mode, reasons, then makes a tool call."""

    def test_thinking_then_tool_call(self):
        """Model thinks first, then produces a DSML tool call block."""
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "What's the weather?"},
        ]
        prompt = encode_messages(
            messages, tools=_WEATHER_TOOLS, thinking_mode="thinking"
        )
        # Thinking mode: prompt should end with <think>
        assert prompt.endswith(THINKING_START)

        # Simulate: model outputs <think>, thinks, closes thinking, then tool call.
        # Use the full production chain (parse_thinking_models → parse_deepseek_v32).
        model_tokens = [
            THINKING_START,
            "The user wants weather",
            " information. I should use",
            " the get_weather tool.",
            THINKING_END,
            "\n\n",
            TOOL_CALLS_START,
            "\n",
            f'<{DSML_TOKEN}invoke name="get_weather">\n',
            f'<{DSML_TOKEN}parameter name="city" string="true">',
            "San Francisco",
            f"</{DSML_TOKEN}parameter>\n",
            f"</{DSML_TOKEN}invoke>\n",
            TOOL_CALLS_END,
        ]

        results = list(_parse_deepseek_with_thinking(_simulate_tokens(model_tokens)))

        gen_results = [r for r in results if isinstance(r, GenerationResponse)]
        tool_results = [r for r in results if isinstance(r, ToolCallResponse)]

        # Should have thinking tokens + tool call
        thinking_results = [r for r in gen_results if r.is_thinking]

        assert len(thinking_results) >= 1
        thinking_text = "".join(r.text for r in thinking_results)
        assert "get_weather tool" in thinking_text

        assert len(tool_results) == 1
        assert tool_results[0].tool_calls[0].name == "get_weather"
        args = json.loads(tool_results[0].tool_calls[0].arguments)  # pyright: ignore[reportAny]
        assert args["city"] == "San Francisco"

    def test_thinking_prompt_encoding(self):
        """Verify thinking mode affects prompt encoding correctly."""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "Be thorough."},
            {"role": "user", "content": "What's the weather?"},
        ]

        # With thinking enabled
        prompt_think = encode_messages(
            messages, tools=_WEATHER_TOOLS, thinking_mode="thinking"
        )
        assert prompt_think.endswith(THINKING_START)

        # With thinking disabled
        prompt_no_think = encode_messages(
            messages, tools=_WEATHER_TOOLS, thinking_mode="chat"
        )
        assert not prompt_no_think.endswith(THINKING_START)

        # Both should have the same tool definitions
        assert "get_weather" in prompt_think
        assert "get_weather" in prompt_no_think


# ── Test: Round-trip encode → parse ──────────────────────────────


class TestE2ERoundTrip:
    """Verify that DSML we encode can be parsed back correctly."""

    def test_encoded_tool_call_is_parseable(self):
        """Encode an assistant tool call message, then parse the DSML output."""
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Weather?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Tokyo", "units": "celsius"}',
                        },
                    }
                ],
            },
        ]

        prompt = encode_messages(messages, thinking_mode="chat", tools=_WEATHER_TOOLS)

        # Extract the DSML function_calls block from the prompt
        start = prompt.index(TOOL_CALLS_START)
        end = prompt.index(TOOL_CALLS_END) + len(TOOL_CALLS_END)
        dsml_block = prompt[start:end]

        # Parse it back
        parsed = parse_dsml_output(dsml_block)
        assert parsed is not None
        assert len(parsed) == 1
        assert parsed[0].name == "get_weather"
        args = json.loads(parsed[0].arguments)  # pyright: ignore[reportAny]
        assert args["city"] == "Tokyo"
        assert args["units"] == "celsius"

    def test_encoded_multi_tool_call_round_trips(self):
        """Encode multiple tool calls, verify they parse back correctly."""
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Both please"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Paris"}',
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "get_time",
                            "arguments": '{"timezone": "CET"}',
                        },
                    },
                ],
            },
        ]

        prompt = encode_messages(messages, thinking_mode="chat", tools=_WEATHER_TOOLS)

        start = prompt.index(TOOL_CALLS_START)
        end = prompt.index(TOOL_CALLS_END) + len(TOOL_CALLS_END)
        dsml_block = prompt[start:end]

        parsed = parse_dsml_output(dsml_block)
        assert parsed is not None
        assert len(parsed) == 2
        assert parsed[0].name == "get_weather"
        assert parsed[1].name == "get_time"
        assert json.loads(parsed[0].arguments) == {"city": "Paris"}
        assert json.loads(parsed[1].arguments) == {"timezone": "CET"}


# ── Test: Edge cases with realistic token boundaries ─────────────


class TestE2EEdgeCases:
    """Edge cases that occur in real model inference."""

    def test_dsml_marker_split_at_fullwidth_pipe(self):
        """The fullwidth pipe character ｜ might be its own token."""
        # This is a realistic tokenization: the DSML marker is split at the ｜ chars
        model_tokens = [
            "Let me help.\n\n",
            "<\uff5c",  # start of ｜DSML｜
            "DSML\uff5c",  # rest of DSML token
            "function_calls>\n",
            f'<{DSML_TOKEN}invoke name="get_weather">\n',
            f'<{DSML_TOKEN}parameter name="city" string="true">NYC</{DSML_TOKEN}parameter>\n',
            f"</{DSML_TOKEN}invoke>\n",
            TOOL_CALLS_END,
        ]

        results = list(parse_deepseek_v32(_simulate_tokens(model_tokens)))
        tool_results = [r for r in results if isinstance(r, ToolCallResponse)]

        assert len(tool_results) == 1
        assert tool_results[0].tool_calls[0].name == "get_weather"

    def test_tool_call_with_nested_json_object(self):
        """Model passes a complex JSON object as a non-string parameter."""
        dsml_block = (
            f"{TOOL_CALLS_START}\n"
            f'<{DSML_TOKEN}invoke name="create_event">\n'
            f'<{DSML_TOKEN}parameter name="title" string="true">Team Standup</{DSML_TOKEN}parameter>\n'
            f'<{DSML_TOKEN}parameter name="config" string="false">'
            f'{{"recurring": true, "days": ["mon", "wed", "fri"], "time": "09:00"}}'
            f"</{DSML_TOKEN}parameter>\n"
            f"</{DSML_TOKEN}invoke>\n"
            f"{TOOL_CALLS_END}"
        )

        # Feed as single token (model might produce it all at once after prefill)
        results = list(parse_deepseek_v32(_simulate_tokens([dsml_block])))
        tool_results = [r for r in results if isinstance(r, ToolCallResponse)]

        assert len(tool_results) == 1
        tc = tool_results[0].tool_calls[0]
        assert tc.name == "create_event"
        args = json.loads(tc.arguments)  # pyright: ignore[reportAny]
        assert args["title"] == "Team Standup"
        assert args["config"]["recurring"] is True
        assert args["config"]["days"] == ["mon", "wed", "fri"]

    def test_text_with_angle_brackets_not_mistaken_for_dsml(self):
        """Angle brackets in normal text should not trigger DSML buffering."""
        model_tokens = [
            "The formula is ",
            "<x, y>",
            " where x > 0",
            " and y < 100.",
        ]

        results = list(parse_deepseek_v32(_simulate_tokens(model_tokens)))
        gen_results = [r for r in results if isinstance(r, GenerationResponse)]
        tool_results = [r for r in results if isinstance(r, ToolCallResponse)]

        assert len(tool_results) == 0
        full_text = "".join(r.text for r in gen_results)
        assert "formula" in full_text
        assert "<x, y>" in full_text

    def test_empty_model_response(self):
        """Model produces only EOS (empty response)."""
        model_tokens = [""]
        results = list(parse_deepseek_v32(_simulate_tokens(model_tokens)))
        gen_results = [r for r in results if isinstance(r, GenerationResponse)]
        assert len(gen_results) == 1
        assert gen_results[0].text == ""
        assert gen_results[0].finish_reason == "stop"


# ── Test: Full EPDP spec round-trip ──────────────────────────────


class TestE2EFullRoundTrip:
    """Full round-trip matching the vLLM EPDP spec.

    Simulates the complete multi-turn flow:
      Turn 1: user asks → think → tool call → tool result → think → answer
      Turn 2: user asks again → old reasoning stripped → think → answer
    """

    def test_single_tool_full_flow_with_thinking(self):
        """Complete flow: user → think → tool call → tool result → think → answer.

        This is the core EPDP flow from the vLLM spec.
        """
        # ── Turn 1.1: User asks, encode prompt ──
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "You are a weather assistant."},
            {"role": "user", "content": "How's the weather in Hangzhou?"},
        ]
        prompt_1 = encode_messages(
            messages, tools=_WEATHER_TOOLS, thinking_mode="thinking"
        )
        assert prompt_1.endswith(THINKING_START)
        assert "## Tools" in prompt_1
        assert "get_weather" in prompt_1

        # ── Turn 1.1: Model thinks, then calls tool ──
        model_tokens_1 = [
            THINKING_START,
            "The user wants to know the weather in Hangzhou.",
            " I need to use the get_weather tool.",
            THINKING_END,
            "\n\n",
            TOOL_CALLS_START,
            "\n",
            f'<{DSML_TOKEN}invoke name="get_weather">\n',
            f'<{DSML_TOKEN}parameter name="city" string="true">Hangzhou</{DSML_TOKEN}parameter>\n',
            f"</{DSML_TOKEN}invoke>\n",
            TOOL_CALLS_END,
        ]
        results_1 = list(
            _parse_deepseek_with_thinking(_simulate_tokens(model_tokens_1))
        )

        # Verify: thinking tokens + tool call
        gen_1 = [r for r in results_1 if isinstance(r, GenerationResponse)]
        tool_1 = [r for r in results_1 if isinstance(r, ToolCallResponse)]
        thinking_1 = [r for r in gen_1 if r.is_thinking]

        assert len(thinking_1) >= 1
        assert "get_weather tool" in "".join(r.text for r in thinking_1)
        assert len(tool_1) == 1
        assert tool_1[0].tool_calls[0].name == "get_weather"
        tc_args = json.loads(tool_1[0].tool_calls[0].arguments)  # pyright: ignore[reportAny]
        assert tc_args == {"city": "Hangzhou"}

        # ── Turn 1.2: Add assistant response + tool result to messages ──
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "The user wants to know the weather in Hangzhou. I need to use the get_weather tool.",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Hangzhou"}',
                        },
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "content": '{"temperature": "7~13°C", "condition": "Cloudy"}',
            }
        )

        # Encode prompt for turn 1.2
        prompt_2 = encode_messages(
            messages, tools=_WEATHER_TOOLS, thinking_mode="thinking"
        )

        # Verify: prompt has the full conversation structure
        assert TOOL_CALLS_START in prompt_2  # assistant's encoded tool call
        assert EOS_TOKEN in prompt_2  # assistant turn ends with EOS
        assert "<function_results>" in prompt_2
        assert "<result>" in prompt_2
        assert "Cloudy" in prompt_2
        assert "</function_results>" in prompt_2
        # After tool results with thinking enabled → <think> appended
        assert prompt_2.endswith(THINKING_START)
        # The assistant's reasoning_content should appear (it's after last_user_idx)
        assert "get_weather tool" in prompt_2

        # ── Turn 1.2: Model thinks about results, then answers ──
        model_tokens_2 = [
            THINKING_START,
            "The weather in Hangzhou is Cloudy, 7~13°C.",
            " I'll tell the user.",
            THINKING_END,
            "The weather in Hangzhou is currently cloudy with temperatures between 7°C and 13°C.",
        ]
        results_2 = list(
            _parse_deepseek_with_thinking(_simulate_tokens(model_tokens_2))
        )

        gen_2 = [r for r in results_2 if isinstance(r, GenerationResponse)]
        tool_2 = [r for r in results_2 if isinstance(r, ToolCallResponse)]
        thinking_2 = [r for r in gen_2 if r.is_thinking]
        non_thinking_2 = [r for r in gen_2 if not r.is_thinking]

        assert len(tool_2) == 0  # No more tool calls
        assert len(thinking_2) >= 1
        assert "Cloudy" in "".join(r.text for r in thinking_2)
        assert len(non_thinking_2) >= 1
        final_text = "".join(r.text for r in non_thinking_2)
        assert "7°C" in final_text
        assert "13°C" in final_text

    def test_multi_tool_full_flow(self):
        """Flow with two tools: user → think → 2 tool calls → 2 results → think → answer."""
        # ── Initial prompt ──
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "You help with weather and time."},
            {"role": "user", "content": "Weather in NYC and time in EST?"},
        ]
        prompt_1 = encode_messages(
            messages, tools=_WEATHER_TOOLS, thinking_mode="thinking"
        )
        assert prompt_1.endswith(THINKING_START)

        # ── Model thinks, calls both tools ──
        model_tokens_1 = [
            THINKING_START,
            "Two requests: weather and time. I'll call both.",
            THINKING_END,
            "\n\n",
            TOOL_CALLS_START,
            "\n",
            f'<{DSML_TOKEN}invoke name="get_weather">\n',
            f'<{DSML_TOKEN}parameter name="city" string="true">NYC</{DSML_TOKEN}parameter>\n',
            f"</{DSML_TOKEN}invoke>\n",
            f'<{DSML_TOKEN}invoke name="get_time">\n',
            f'<{DSML_TOKEN}parameter name="timezone" string="true">EST</{DSML_TOKEN}parameter>\n',
            f"</{DSML_TOKEN}invoke>\n",
            TOOL_CALLS_END,
        ]
        results_1 = list(parse_deepseek_v32(_simulate_tokens(model_tokens_1)))
        tool_1 = [r for r in results_1 if isinstance(r, ToolCallResponse)]

        assert len(tool_1) == 1
        assert len(tool_1[0].tool_calls) == 2
        assert tool_1[0].tool_calls[0].name == "get_weather"
        assert tool_1[0].tool_calls[1].name == "get_time"

        # ── Add assistant + both tool results ──
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "Two requests: weather and time. I'll call both.",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "NYC"}',
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "get_time",
                            "arguments": '{"timezone": "EST"}',
                        },
                    },
                ],
            }
        )
        messages.append({"role": "tool", "content": "72°F, sunny"})
        messages.append({"role": "tool", "content": "2:30 PM EST"})

        prompt_2 = encode_messages(
            messages, tools=_WEATHER_TOOLS, thinking_mode="thinking"
        )

        # Verify multi-tool result encoding
        # Count is 2: 1 in _TOOLS_SYSTEM_TEMPLATE example + 1 in conversation
        assert prompt_2.count("<function_results>") == 2
        assert prompt_2.count("</function_results>") == 2
        assert "<result>72°F, sunny</result>" in prompt_2
        assert "<result>2:30 PM EST</result>" in prompt_2
        assert prompt_2.endswith(THINKING_START)

        # ── Model thinks about results, answers ──
        model_tokens_2 = [
            THINKING_START,
            "Got both results. Weather is 72°F sunny, time is 2:30 PM.",
            THINKING_END,
            "In NYC it's currently 72°F and sunny. The time in EST is 2:30 PM.",
        ]
        results_2 = list(parse_deepseek_v32(_simulate_tokens(model_tokens_2)))

        tool_2 = [r for r in results_2 if isinstance(r, ToolCallResponse)]
        gen_2 = [r for r in results_2 if isinstance(r, GenerationResponse)]
        non_thinking_2 = [r for r in gen_2 if not r.is_thinking]

        assert len(tool_2) == 0
        final_text = "".join(r.text for r in non_thinking_2)
        assert "72°F" in final_text
        assert "2:30 PM" in final_text

    def test_two_user_turns_reasoning_stripped(self):
        """Turn 2: old reasoning_content is stripped from history.

        Per the vLLM spec, clear_reasoning_content is called between user turns
        to save bandwidth. Our _drop_old_thinking handles this.
        """
        # Full turn 1 conversation (already completed)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Weather in Hangzhou?"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "I need to call get_weather for Hangzhou.",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Hangzhou"}',
                        },
                    }
                ],
            },
            {"role": "tool", "content": "Cloudy 7~13°C"},
            {
                "role": "assistant",
                "content": "The weather in Hangzhou is cloudy, 7-13°C.",
                "reasoning_content": "The tool returned cloudy weather. I'll summarize.",
            },
            # Turn 2: user asks again
            {"role": "user", "content": "What about Beijing?"},
        ]

        prompt = encode_messages(
            messages, tools=_WEATHER_TOOLS, thinking_mode="thinking"
        )

        # Old reasoning_content from turn 1 assistants should be STRIPPED
        # (they're before the last user message at index 5)
        assert "I need to call get_weather" not in prompt
        assert "tool returned cloudy" not in prompt

        # But the assistant's content and tool calls should still be there
        assert "cloudy, 7-13°C" in prompt
        assert TOOL_CALLS_START in prompt

        # Prompt ends with <think> for the new turn
        assert prompt.endswith(THINKING_START)

        # ── Turn 2: Model thinks, calls tool for Beijing ──
        model_tokens = [
            THINKING_START,
            "Now the user wants Beijing weather.",
            THINKING_END,
            "\n\n",
            TOOL_CALLS_START,
            "\n",
            f'<{DSML_TOKEN}invoke name="get_weather">\n',
            f'<{DSML_TOKEN}parameter name="city" string="true">Beijing</{DSML_TOKEN}parameter>\n',
            f"</{DSML_TOKEN}invoke>\n",
            TOOL_CALLS_END,
        ]
        results = list(parse_deepseek_v32(_simulate_tokens(model_tokens)))
        tool_results = [r for r in results if isinstance(r, ToolCallResponse)]

        assert len(tool_results) == 1
        assert tool_results[0].tool_calls[0].name == "get_weather"
        args = json.loads(tool_results[0].tool_calls[0].arguments)  # pyright: ignore[reportAny]
        assert args == {"city": "Beijing"}

    def test_chained_tool_calls_loop(self):
        """Model calls tool, gets result, calls another tool, gets result, answers.

        This simulates the inner while loop from the vLLM spec where the model
        may need multiple sub-turns of tool calling before it has enough info.
        """
        # ── Sub-turn 1: user asks, model calls get_time ──
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What's the weather in Hangzhou tomorrow?"},
        ]

        prompt_1 = encode_messages(
            messages, tools=_WEATHER_TOOLS, thinking_mode="thinking"
        )
        assert prompt_1.endswith(THINKING_START)

        # Model first calls get_time to figure out the date
        model_tokens_1 = [
            THINKING_START,
            "I need the current date first to calculate tomorrow.",
            THINKING_END,
            "\n\n",
            TOOL_CALLS_START,
            "\n",
            f'<{DSML_TOKEN}invoke name="get_time">\n',
            f'<{DSML_TOKEN}parameter name="timezone" string="true">Asia/Shanghai</{DSML_TOKEN}parameter>\n',
            f"</{DSML_TOKEN}invoke>\n",
            TOOL_CALLS_END,
        ]
        results_1 = list(parse_deepseek_v32(_simulate_tokens(model_tokens_1)))
        tool_1 = [r for r in results_1 if isinstance(r, ToolCallResponse)]
        assert len(tool_1) == 1
        assert tool_1[0].tool_calls[0].name == "get_time"

        # ── Sub-turn 2: add tool result, model calls get_weather ──
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "I need the current date first to calculate tomorrow.",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_time",
                            "arguments": '{"timezone": "Asia/Shanghai"}',
                        },
                    }
                ],
            }
        )
        messages.append({"role": "tool", "content": "2025-12-01 14:30 CST"})

        prompt_2 = encode_messages(
            messages, tools=_WEATHER_TOOLS, thinking_mode="thinking"
        )
        assert "<result>2025-12-01 14:30 CST</result>" in prompt_2
        assert prompt_2.endswith(THINKING_START)

        # Model now knows the date, calls get_weather
        model_tokens_2 = [
            THINKING_START,
            "Today is 2025-12-01, so tomorrow is 2025-12-02.",
            " Now I can check weather for Hangzhou.",
            THINKING_END,
            "\n\n",
            TOOL_CALLS_START,
            "\n",
            f'<{DSML_TOKEN}invoke name="get_weather">\n',
            f'<{DSML_TOKEN}parameter name="city" string="true">Hangzhou</{DSML_TOKEN}parameter>\n',
            f"</{DSML_TOKEN}invoke>\n",
            TOOL_CALLS_END,
        ]
        results_2 = list(parse_deepseek_v32(_simulate_tokens(model_tokens_2)))
        tool_2 = [r for r in results_2 if isinstance(r, ToolCallResponse)]
        assert len(tool_2) == 1
        assert tool_2[0].tool_calls[0].name == "get_weather"

        # ── Sub-turn 3: add weather result, model answers ──
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "Today is 2025-12-01, so tomorrow is 2025-12-02. Now I can check weather for Hangzhou.",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Hangzhou"}',
                        },
                    }
                ],
            }
        )
        messages.append({"role": "tool", "content": "Sunny, 5~12°C"})

        prompt_3 = encode_messages(
            messages, tools=_WEATHER_TOOLS, thinking_mode="thinking"
        )
        # Should have both function_results blocks (one per tool round)
        # Count is 3: 1 in _TOOLS_SYSTEM_TEMPLATE example + 2 in conversation
        assert prompt_3.count("<function_results>") == 3
        assert prompt_3.count("</function_results>") == 3
        assert "<result>2025-12-01 14:30 CST</result>" in prompt_3
        assert "<result>Sunny, 5~12°C</result>" in prompt_3
        assert prompt_3.endswith(THINKING_START)

        # Model finally answers
        model_tokens_3 = [
            THINKING_START,
            "I have the weather for tomorrow in Hangzhou.",
            THINKING_END,
            "Tomorrow in Hangzhou will be sunny with temperatures between 5°C and 12°C.",
        ]
        results_3 = list(parse_deepseek_v32(_simulate_tokens(model_tokens_3)))

        tool_3 = [r for r in results_3 if isinstance(r, ToolCallResponse)]
        gen_3 = [r for r in results_3 if isinstance(r, GenerationResponse)]
        non_thinking_3 = [r for r in gen_3 if not r.is_thinking]

        assert len(tool_3) == 0  # No more tool calls — loop ends
        final_text = "".join(r.text for r in non_thinking_3)
        assert "sunny" in final_text.lower()
        assert "5°C" in final_text
        assert "12°C" in final_text


class TestMultiTurnThinkingPrompt:
    def test_no_orphan_think_end_in_multiturn(self):
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Hi!"},
            {"role": "assistant", "content": "Hello! How can I help you today?"},
            {"role": "user", "content": "Tell me about Paris."},
        ]
        prompt = encode_messages(messages, thinking_mode="thinking")
        assistant_token = "<\uff5cAssistant\uff5c>"
        parts = prompt.split(assistant_token)
        for part in parts[1:]:
            assert not part.startswith(THINKING_END), (
                f"Orphan </think> without <think> after <Assistant>: ...{assistant_token}{part[:50]}"
            )


class TestApplyChatTemplateWithToolCalls:
    def test_dsml_encoding_with_tool_calls_in_history(self):
        from exo.shared.types.text_generation import (
            InputMessage,
            InputMessageContent,
            TextGenerationTaskParams,
        )
        from exo.worker.engines.mlx.utils_mlx import apply_chat_template

        chat_template_messages: list[dict[str, Any]] = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Tokyo"}',
                        },
                    }
                ],
            },
            {"role": "tool", "content": "Sunny, 25°C"},
            {"role": "user", "content": "Thanks!"},
        ]

        from unittest.mock import MagicMock

        tokenizer = MagicMock()
        tokenizer.has_thinking = True
        tokenizer.think_start = "<think>"
        tokenizer.think_end = "</think>"

        params = TextGenerationTaskParams(
            model=ModelId("mlx-community/DeepSeek-V3.2-8bit"),
            input=[InputMessage(role="user", content=InputMessageContent("Thanks!"))],
            instructions=InputMessageContent("You are a helpful assistant."),
            enable_thinking=True,
            chat_template_messages=chat_template_messages,
            tools=_WEATHER_TOOLS,
        )

        prompt = apply_chat_template(tokenizer, params)
        assert "get_weather" in prompt
        assert "Tokyo" in prompt
        assert "Sunny" in prompt


class TestE2EDeepseekV4ToolCallParsing:
    """V4 emits `<｜DSML｜tool_calls>` (outer) wrapping `<｜DSML｜invoke …>` calls
    (the V4-Flash chat template promises this exact structure). Parser must
    extract the tool name + parameters back out."""

    def test_v4_tool_call_extracted_from_clean_output(self):
        """Clean V4 DSML output should yield a ToolCallResponse with the
        invoked tool name and parameter values — not bleed through as text."""
        # Realistic token splits matching the V4 tokenizer's known behavior:
        #   `<｜DSML｜tool_calls>` -> ['<', '｜DSML｜', 'tool', '_c', 'alls', '>']
        # The model emits tokens one-by-one in this multi-token pattern.
        model_tokens = [
            "<",
            DSML_TOKEN,
            "tool",
            "_c",
            "alls",
            ">",
            "\n<",
            DSML_TOKEN,
            "invoke",
            ' name="read"',
            ">\n<",
            DSML_TOKEN,
            "parameter",
            ' name="filePath" string="true"',
            ">",
            "/Users/l2/PycharmProjects/exo",
            "</",
            DSML_TOKEN,
            "parameter",
            ">\n</",
            DSML_TOKEN,
            "invoke",
            ">\n</",
            DSML_TOKEN,
            "tool",
            "_c",
            "alls",
            ">",
        ]

        results = list(parse_deepseek_v4(_simulate_tokens(model_tokens)))

        tool_results = [r for r in results if isinstance(r, ToolCallResponse)]
        text_results = [r for r in results if isinstance(r, GenerationResponse)]

        assert len(tool_results) == 1, (
            f"expected one ToolCallResponse, got {len(tool_results)} tool + "
            f"{len(text_results)} text results: text="
            f"{''.join(r.text for r in text_results)!r}"
        )
        tool_calls = tool_results[0].tool_calls
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "read"
        args = cast(dict[str, str], json.loads(tool_calls[0].arguments))
        assert args == {"filePath": "/Users/l2/PycharmProjects/exo"}

    def test_v4_tool_call_carries_usage_from_terminal_response(self):
        """Usage must ride out on the ToolCallResponse even though the DSML
        close marker arrives a token or two BEFORE finish_reason/usage.

        The mlx generator only builds usage on the is_done (finish_reason)
        response. For a tool call, the ``</｜DSML｜tool_calls>`` marker closes
        the block on an earlier token whose usage is None. If the parser emits
        the ToolCallResponse the instant the block closes, the token usage is
        lost (usage=None) and the client's context counter never advances on a
        tool-calling turn. The parser must hold the tool call until the terminal
        response and attach its usage.
        """
        from exo.api.types import Usage
        from exo.api.types.api import (
            CompletionTokensDetails,
            PromptTokensDetails,
        )

        usage = Usage(
            prompt_tokens=265,
            completion_tokens=12,
            total_tokens=277,
            prompt_tokens_details=PromptTokensDetails(cached_tokens=263),
            completion_tokens_details=CompletionTokensDetails(reasoning_tokens=4),
        )

        # Block closes on the ">" token (usage=None); a SEPARATE terminal
        # response carries finish_reason + usage, mirroring exo's real SSE
        # ordering for a tool call.
        text_tokens = [
            "<", DSML_TOKEN, "tool", "_c", "alls", ">",
            "\n<", DSML_TOKEN, "invoke", ' name="read"', ">\n<",
            DSML_TOKEN, "parameter", ' name="filePath" string="true"', ">",
            "/tmp/x", "</", DSML_TOKEN, "parameter", ">\n</",
            DSML_TOKEN, "invoke", ">\n</", DSML_TOKEN, "tool", "_c", "alls", ">",
        ]

        def _tokens() -> Generator[GenerationResponse]:
            for i, t in enumerate(text_tokens):
                yield GenerationResponse(
                    text=t, token=i, finish_reason=None, usage=None
                )
            # Terminal response: no text, carries finish_reason + usage.
            yield GenerationResponse(
                text="", token=len(text_tokens),
                finish_reason="tool_calls", usage=usage,
            )

        results = list(parse_deepseek_v4(_tokens()))
        tool_results = [r for r in results if isinstance(r, ToolCallResponse)]

        assert len(tool_results) == 1, f"expected one tool call, got {results!r}"
        tc = tool_results[0]
        assert tc.tool_calls[0].name == "read"
        # The whole point: usage survived onto the tool call.
        assert tc.usage is not None, "ToolCallResponse.usage was dropped"
        assert tc.usage.prompt_tokens == 265

    def test_v4_tool_call_usage_when_marker_and_finish_same_response(self):
        """When the full DSML block AND finish_reason land on the SAME response,
        usage is already present and must be attached without deferral."""
        from exo.api.types import Usage
        from exo.api.types.api import (
            CompletionTokensDetails,
            PromptTokensDetails,
        )

        usage = Usage(
            prompt_tokens=99,
            completion_tokens=5,
            total_tokens=104,
            prompt_tokens_details=PromptTokensDetails(cached_tokens=0),
            completion_tokens_details=CompletionTokensDetails(reasoning_tokens=0),
        )
        # NOTE: V4 wraps tool calls in ``<｜DSML｜tool_calls>`` — NOT the V3.2
        # ``function_calls`` markers (TOOL_CALLS_START/END). Build the V4
        # wrapper explicitly here.
        v4_start = f"<{DSML_TOKEN}tool_calls>"
        v4_end = f"</{DSML_TOKEN}tool_calls>"
        block = (
            f"{v4_start}"
            f'<{DSML_TOKEN}invoke name="read">'
            f'<{DSML_TOKEN}parameter name="filePath" string="true">/x</{DSML_TOKEN}parameter>'
            f"</{DSML_TOKEN}invoke>"
            f"{v4_end}"
        )

        def _one() -> Generator[GenerationResponse]:
            yield GenerationResponse(
                text=block, token=0, finish_reason="tool_calls", usage=usage
            )

        results = list(parse_deepseek_v4(_one()))
        tool_results = [r for r in results if isinstance(r, ToolCallResponse)]
        assert len(tool_results) == 1
        assert tool_results[0].usage is not None
        assert tool_results[0].usage.prompt_tokens == 99

    def test_v4_malformed_block_does_not_leak_dsml_tokens(self):
        """A malformed tool-call block (wrapper opened, but the body is not a
        valid invoke) must not leak raw ``<｜DSML｜...>`` control tokens into
        displayed content.

        Reproduces an observed generation hiccup: the model opens a
        ``tool_calls`` wrapper and then, instead of an ``invoke`` block,
        regurgitates a prior tool RESULT. ``parse_dsml_output`` returns None,
        and the stream parser falls back to yielding the residue as text — that
        residue must have the DSML special tokens stripped.
        """
        # Wrapper opens, body is a parroted git-diff result (no invoke), closes.
        model_tokens = [
            "<",
            DSML_TOKEN,
            "tool",
            "_c",
            "alls",
            ">",
            "\n<",
            DSML_TOKEN,
            "_cli.py | 6 ++++++",
            ' 6 files changed", "exit_code": 0, "error": null}',
            "</",
            DSML_TOKEN,
            "tool",
            "_c",
            "alls",
            ">",
        ]

        results = list(parse_deepseek_v4(_simulate_tokens(model_tokens)))

        tool_results = [r for r in results if isinstance(r, ToolCallResponse)]
        text_results = [r for r in results if isinstance(r, GenerationResponse)]
        full_text = "".join(r.text for r in text_results)

        # No spurious tool call should be emitted from the malformed block.
        assert len(tool_results) == 0
        # The DSML special token must never survive into displayed text.
        assert DSML_TOKEN not in full_text
        assert "<\uff5cDSML" not in full_text
        # But the human-readable residue is preserved (not silently dropped).
        assert "6 files changed" in full_text

    def test_v4_tool_call_after_thinking_block(self):
        """V4 reasoning models start in `<think>` and emit DSML tool calls
        after `</think>`. The thinking parser must hand a complete tool-call
        block off to `parse_deepseek_v4` without dropping markers."""
        # `</think>` token-splits into ['</think>'] in V4's tokenizer, so the
        # thinking parser sees it as a single token. Emit thinking, then the
        # DSML tool call.
        model_tokens = [
            "<think>",
            "The user wants me to explore the codebase.",
            "</think>",
            "<",
            DSML_TOKEN,
            "tool",
            "_c",
            "alls",
            ">",
            "\n<",
            DSML_TOKEN,
            "invoke",
            ' name="read"',
            ">\n<",
            DSML_TOKEN,
            "parameter",
            ' name="filePath" string="true"',
            ">",
            "/Users/l2/PycharmProjects/exo",
            "</",
            DSML_TOKEN,
            "parameter",
            ">\n</",
            DSML_TOKEN,
            "invoke",
            ">\n</",
            DSML_TOKEN,
            "tool",
            "_c",
            "alls",
            ">",
        ]

        results = list(
            parse_deepseek_v4(
                parse_thinking_models(
                    _simulate_tokens(model_tokens),
                    think_start="<think>",
                    think_end="</think>",
                    starts_in_thinking=True,
                )
            )
        )

        tool_results = [r for r in results if isinstance(r, ToolCallResponse)]
        text_results = [r for r in results if isinstance(r, GenerationResponse)]
        non_thinking_text = "".join(r.text for r in text_results if not r.is_thinking)

        assert len(tool_results) == 1, (
            f"expected ToolCallResponse, got {len(tool_results)} tool + "
            f"non-thinking text {non_thinking_text!r}"
        )
        tool_calls = tool_results[0].tool_calls
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "read"
        args = cast(dict[str, str], json.loads(tool_calls[0].arguments))
        assert args == {"filePath": "/Users/l2/PycharmProjects/exo"}


# Token-ID-based real-vs-quoted DSML detection.
#
# A REAL tool call emits the ``｜DSML｜`` sentinel as a dedicated special vocab
# token. When the model QUOTES the marker in reasoning/content prose (e.g.
# explaining ``<｜DSML｜tool_calls>`` in an answer), the same characters are
# produced as ordinary BPE text tokens. The parser must distinguish the two by
# the special-token *id*, not the decoded string — otherwise a quoted marker
# triggers a false tool-call parse that strips the markers and leaks the rest of
# the turn into ``content`` (observed 2026-06-15: a turn whose reasoning
# discussed the DSML syntax had its tail fused onto the answer).
_DSML_SPECIAL_ID = 999999  # stand-in for the ｜DSML｜ special vocab id


def _simulate_tokens_with_special(
    texts: list[str],
    special_text: str,
    special_id: int,
    finish_on_last: bool = True,
) -> Generator[GenerationResponse]:
    """Like ``_simulate_tokens`` but assigns ``special_id`` to any chunk whose
    text equals ``special_text`` (mirroring a real special vocab token), and a
    plain sequential id to everything else (ordinary BPE text)."""
    for i, text in enumerate(texts):
        is_last = i == len(texts) - 1
        yield GenerationResponse(
            text=text,
            token=special_id if text == special_text else i,
            finish_reason="stop" if (is_last and finish_on_last) else None,
            usage=None,
        )


class TestE2EDeepseekV4QuotedMarkerDetection:
    """The model quoting DSML markers in prose must not be parsed as a tool
    call (token-id-based real-vs-quoted detection)."""

    def test_quoted_marker_in_prose_does_not_trigger_tool_call(self):
        """Reproduces the 2026-06-15 leak: the model's answer DISCUSSES the
        DSML tool-call syntax, emitting the literal ``<｜DSML｜tool_calls>`` /
        ``</｜DSML｜tool_calls>`` text as ordinary tokens. With the special-token
        id supplied, the parser must treat this as plain content — no tool call,
        no marker stripping, full prose preserved verbatim."""
        prose_tokens = [
            "The parser opens a `<",
            DSML_TOKEN,
            "tool_calls>` wrapper but parrots a prior result instead of a valid `<",
            DSML_TOKEN,
            "invoke>` body, so we strip it. </",
            DSML_TOKEN,
            "tool_calls>` That is the whole bug.",
        ]

        # All chunks get plain sequential ids — the ｜DSML｜ never arrives as the
        # special token, so it's quoted prose, not a real call.
        def _tokens() -> Generator[GenerationResponse]:
            for i, t in enumerate(prose_tokens):
                yield GenerationResponse(
                    text=t,
                    token=i,  # never _DSML_SPECIAL_ID -> quoted
                    finish_reason="stop" if i == len(prose_tokens) - 1 else None,
                    usage=None,
                )

        results = list(
            parse_deepseek_v4(_tokens(), frozenset({_DSML_SPECIAL_ID}))
        )
        tool_results = [r for r in results if isinstance(r, ToolCallResponse)]
        text_results = [r for r in results if isinstance(r, GenerationResponse)]
        full_text = "".join(r.text for r in text_results)

        # No false tool call.
        assert len(tool_results) == 0, f"quoted marker wrongly parsed: {results!r}"
        # The prose is preserved verbatim — including the quoted marker text,
        # which is the model's own readable output, NOT a leaked control token.
        assert "That is the whole bug." in full_text
        assert "wrapper but parrots a prior result" in full_text
        assert full_text.count("tool_calls>") >= 1

    def test_real_tool_call_with_special_token_still_parses(self):
        """The genuine path: when the ｜DSML｜ sentinel arrives as its special
        vocab token, the block parses to a ToolCallResponse as before."""
        model_tokens = [
            "<", DSML_TOKEN, "tool", "_c", "alls", ">",
            "\n<", DSML_TOKEN, "invoke", ' name="read"', ">\n<",
            DSML_TOKEN, "parameter", ' name="filePath" string="true"', ">",
            "/tmp/x", "</", DSML_TOKEN, "parameter", ">\n</",
            DSML_TOKEN, "invoke", ">\n</", DSML_TOKEN, "tool", "_c", "alls", ">",
        ]

        results = list(
            parse_deepseek_v4(
                _simulate_tokens_with_special(
                    model_tokens, DSML_TOKEN, _DSML_SPECIAL_ID
                ),
                frozenset({_DSML_SPECIAL_ID}),
            )
        )
        tool_results = [r for r in results if isinstance(r, ToolCallResponse)]
        assert len(tool_results) == 1, f"real tool call not parsed: {results!r}"
        assert tool_results[0].tool_calls[0].name == "read"
        args = cast(
            dict[str, str], json.loads(tool_results[0].tool_calls[0].arguments)
        )
        assert args == {"filePath": "/tmp/x"}

    def test_legacy_fallback_without_ids_unchanged(self):
        """With no special-token ids supplied (e.g. a tokenizer that can't
        resolve them), the parser falls back to the legacy text-only behavior —
        a clean DSML block still parses to a tool call."""
        model_tokens = [
            "<", DSML_TOKEN, "tool", "_c", "alls", ">",
            "\n<", DSML_TOKEN, "invoke", ' name="read"', ">\n<",
            DSML_TOKEN, "parameter", ' name="filePath" string="true"', ">",
            "/tmp/x", "</", DSML_TOKEN, "parameter", ">\n</",
            DSML_TOKEN, "invoke", ">\n</", DSML_TOKEN, "tool", "_c", "alls", ">",
        ]
        # No ids -> legacy text-only path (today's behavior).
        results = list(parse_deepseek_v4(_simulate_tokens(model_tokens)))
        tool_results = [r for r in results if isinstance(r, ToolCallResponse)]
        assert len(tool_results) == 1
        assert tool_results[0].tool_calls[0].name == "read"
