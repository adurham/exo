#!/usr/bin/env python3
import argparse
import json
import sys
from typing import Any, cast

import requests
from typing import Iterator


def stream_chat(host: str, model: str, query: str) -> None:
    url = f"http://{host}:52415/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    payload: dict[str, Any] = {
        "model": model,
        "stream": True,
        "messages": [{"role": "user", "content": query}],
    }

    try:
        with requests.post(url, headers=headers, json=payload, stream=True) as resp:
            resp.raise_for_status()
            for raw_line in cast(Iterator[str | bytes | None], resp.iter_lines(decode_unicode=True)):
                line: str | bytes | None = raw_line
                if not line or not isinstance(line, str):
                    continue

                # SSE lines look like: "data: {...}" or "data: [DONE]"
                line_str: str = line
                if not line_str.startswith("data:"):
                    continue

                data: str = line_str[len("data:") :].strip()
                if data == "[DONE]":
                    break

                try:
                    obj = cast(dict[str, Any], json.loads(data))
                except json.JSONDecodeError:
                    continue

                choices = cast(list[dict[str, Any]], obj.get("choices", []))
                for choice in choices:
                    delta = cast(dict[str, Any], choice.get("delta") or {})
                    content = cast(str | None, delta.get("content"))
                    if content:
                        print(content, end="", flush=True)

    except requests.RequestException as e:
        print(f"Request failed: {e}", file=sys.stderr)
        sys.exit(1)

    print()


def main() -> None:
    class Args(argparse.Namespace):
        host: str
        model: str
        file: str | None
        query: list[str]

    parser = argparse.ArgumentParser(
        description="Stream chat completions from a local server."
    )
    parser.add_argument("host", help="Hostname (without protocol), e.g. localhost")
    parser.add_argument(
        "--model",
        default="mlx-community/Qwen3-235B-Instruct-4bit",
        help="Model name to query (default: mlx-community/Qwen3-235B-Instruct-4bit)",
    )
    parser.add_argument(
        "-f",
        "--file",
        help="Path to a text file whose contents will be used as the query",
    )
    parser.add_argument(
        "query",
        nargs="*",
        help="Query text (if not using -f/--file). All remaining arguments are joined with spaces.",
    )

    args = cast(Args, parser.parse_args())
    host: str = args.host
    model: str = args.model
    file_path = args.file
    query: str = ""

    if file_path:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                query = f.read().strip()
        except OSError as e:
            print(f"Error reading file {file_path}: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.query:
        query_parts: list[str] = [str(p) for p in args.query]
        query = " ".join(query_parts)
    else:
        parser.error("You must provide either a query or a file (-f/--file).")
        return
    if query == "":
        parser.error("Query is empty after parsing.")
        return

    print(f"Querying model: {model}")
    stream_chat(host, model, query)


if __name__ == "__main__":
    main()
