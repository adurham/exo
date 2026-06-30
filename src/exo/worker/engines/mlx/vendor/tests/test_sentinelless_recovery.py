"""Tests for sentinel-less (wrong-dialect) tool-call recovery.

DSv4 occasionally emits the correct invoke/parameter structure WITHOUT the
``｜DSML｜`` sentinel (bare Claude/minimax dialect). ``parse_sentinelless_tool_call``
recovers the call so the tool runs instead of the tags leaking as content and
the call dropping. See dsml_encoding.py and model_output_parsers.py.
"""
import json

from exo.worker.engines.mlx.vendor.dsml_encoding import (
    DSML_TOKEN,
    parse_sentinelless_tool_call,
)


class TestParseSentinellessToolCall:
    def test_recovers_typed_param_dialect(self):
        """DSv4's own bare form: string="true|false" annotations, no sentinel.

        This is the exact leak from msg 95278 (2026-06-29)."""
        text = (
            "<tool_call>\n"
            '<invoke name="read_file">\n'
            '<parameter name="limit" string="false">15</parameter>\n'
            '<parameter name="path" string="true">~/.hermes/config.yaml</parameter>\n'
            "</invoke>"
        )
        calls = parse_sentinelless_tool_call(text)
        assert calls is not None
        assert len(calls) == 1
        assert calls[0].name == "read_file"
        args = json.loads(calls[0].arguments)
        # string="false" -> JSON-decoded to int
        assert args["limit"] == 15
        # string="true" -> kept verbatim (path is not valid JSON anyway)
        assert args["path"] == "~/.hermes/config.yaml"

    def test_recovers_plain_claude_minimax_dialect(self):
        """Pure Claude/minimax form: <parameter name="x">value</parameter>,
        no string= attribute at all (mlx-lm minimax_m2 corpus)."""
        text = (
            '<invoke name="multiply">\n'
            '<parameter name="a">12234585</parameter>\n'
            '<parameter name="b">48838483920</parameter>\n'
            "</invoke>"
        )
        calls = parse_sentinelless_tool_call(text)
        assert calls is not None
        assert len(calls) == 1
        assert calls[0].name == "multiply"
        args = json.loads(calls[0].arguments)
        assert args == {"a": 12234585, "b": 48838483920}

    def test_string_path_not_json_falls_back_to_raw(self):
        text = (
            '<invoke name="terminal">\n'
            '<parameter name="command" string="true">ls -la /tmp</parameter>\n'
            "</invoke>"
        )
        calls = parse_sentinelless_tool_call(text)
        assert calls is not None
        args = json.loads(calls[0].arguments)
        assert args["command"] == "ls -la /tmp"

    def test_multiple_invokes(self):
        text = (
            '<invoke name="a"><parameter name="x">1</parameter></invoke>'
            '<invoke name="b"><parameter name="y">2</parameter></invoke>'
        )
        calls = parse_sentinelless_tool_call(text)
        assert calls is not None
        assert [c.name for c in calls] == ["a", "b"]

    def test_returns_none_when_sentinel_present(self):
        """A real/quoted DSML block is owned by the sentinel parser, not us."""
        text = (
            f'<{DSML_TOKEN}invoke name="read_file">'
            f'<{DSML_TOKEN}parameter name="path" string="true">/x</{DSML_TOKEN}parameter>'
            f"</{DSML_TOKEN}invoke>"
        )
        assert parse_sentinelless_tool_call(text) is None

    def test_returns_none_on_prose(self):
        assert parse_sentinelless_tool_call("just some text, no tags") is None

    def test_returns_none_on_invoke_without_params(self):
        # An invoke with no parameters yields empty args; still recovered as a
        # zero-arg call (some tools take none). Verify it parses to {}.
        text = '<invoke name="ping"></invoke>'
        calls = parse_sentinelless_tool_call(text)
        assert calls is not None
        assert calls[0].name == "ping"
        assert json.loads(calls[0].arguments) == {}
