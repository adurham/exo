# Offline LCP-coverage probe — prefill-round2 2026-09-02
#
# For the 54 recorded turn pairs (call_seq 33..87, main chat, Hermes session
# 20260901_120301_93ad7b, 2026-09-01): longest common PREFIX between call n's
# decode output (PROXY — true decode token ids are not persisted anywhere; see
# report) and the token ids the REAL production DSv4 chat-template path re-feeds
# as prompt at call n+1.
#
# Reproduces exactly (verified against source, read-only):
#   Hermes -> wire   agent/conversation_loop.py:2552-2660 (system first;
#                    assistant reasoning_content echoed; tool rows with
#                    tool_call_id; drop_thinking_only_and_merge_users)
#   exo adapter      src/exo/api/adapters/chat_completions.py:62
#                    chat_request_to_text_generation (skip empty; exclude_none)
#   exo template     src/exo/worker/engines/mlx/utils_mlx.py:1596/1476
#                    render_chat_template -> consolidate_system_messages ->
#                    _strip_v4_thinking_markers(content) on every assistant msg
#                    -> vendor/deepseek_v4_encoding.encode_messages
#   DSv4 encoding    merge_tool_messages -> sort_tool_results_by_call_order ->
#                    render_message; tools in play => drop_thinking DISABLED
#                    => every prior assistant turn re-rendered with its full
#                    reasoning_content + </think>; user/dev msg followed by
#                    non-assistant gets <｜Assistant｜>...(2 tokens: 128804,128821)
#   tokenization     cache.encode_prompt: tokenizer.encode(prompt, add_special_tokens=False)
#
import json
import re
import sqlite3
import bisect
import statistics
from collections import Counter
from tokenizers import Tokenizer

DB = 'file:/Users/adam.durham/.hermes/state.db?mode=ro'
SNAP = ('/Users/adam.durham/.cache/huggingface/hub/models--deepseek-ai--'
        'DeepSeek-V4-Flash-0731/snapshots/'
        '7872f01b1d1fe23eabc4c98b48bffcef5a386062')
OUT_JSON = ('/Users/adam.durham/repos/exo/tmp/prefill-round2-20260902/'
            'findings/lcp_probe.json')
SESSION = '20260901_120301_93ad7b'

BOS = '<｜begin▁of▁sentence｜>'
EOS = '<｜end▁of▁sentence｜>'
ASSISTANT = '<｜Assistant｜>'
USER = '<｜User｜>'
THINK_S = "<think>"
THINK_E = "</think>"
DSML = '｜DSML｜'

TOOLS_TEMPLATE = """## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<{dsml}tool_calls>" block like the following:

<{dsml}tool_calls>
<{dsml}invoke name="$TOOL_NAME">
<{dsml}parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</{dsml}parameter>
...
</{dsml}invoke>
<{dsml}invoke name="$TOOL_NAME2">
...
</{dsml}invoke>
</{dsml}tool_calls>

String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

If thinking_mode is enabled (triggered by {tstart}), you MUST output your complete reasoning inside {tstart}...{tend} BEFORE any tool calls or final response.

Otherwise, output directly after {tend} with tool calls or final response.

### Available Tool Schemas

{tool_schemas}

You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
"""

# ---------------------------------------------------------------- tokenizer
tok = Tokenizer.from_file(SNAP + '/tokenizer.json')
for tid, content in [(128803, USER), (128804, ASSISTANT), (128821, THINK_S),
                     (128822, THINK_E), (128825, DSML)]:
    enc = tok.encode(content, add_special_tokens=False).ids
    assert enc == [tid], (content, enc, tid)


def T(text):
    return tok.encode(text, add_special_tokens=False).ids


# ------------------------------------------------------------- db extraction
con = sqlite3.connect(DB, uri=True)
con.row_factory = sqlite3.Row
calls = con.execute(
    "SELECT call_seq, started_at, output_tokens, prompt_tokens_total FROM api_calls "
    "WHERE session_id=? AND call_seq BETWEEN 33 AND 87 "
    "AND datetime(started_at,'unixepoch','localtime') < '2026-09-01 17:00' "
    "ORDER BY started_at", (SESSION,)).fetchall()
starts = [c['started_at'] for c in calls]
seqs = [c['call_seq'] for c in calls]
out_tok = {c['call_seq']: c['output_tokens'] for c in calls}
prompt_tok_rec = {c['call_seq']: c['prompt_tokens_total'] for c in calls}

msgs = con.execute(
    "SELECT id, role, timestamp, content, tool_calls, tool_call_id, reasoning, "
    "reasoning_content, finish_reason FROM messages WHERE session_id=? "
    "ORDER BY id", (SESSION,)).fetchall()
con.close()

# slices[seq] = messages produced between call seq-1 and call seq
# (i.e. the NEW content re-fed in prompt(seq))
# message m is re-fed as NEW content in prompt(seq) for the first seq whose
# started_at is AFTER m's timestamp; equivalently it was produced by call seq-1.
slices = {q: [] for q in seqs}
for m in msgs:
    i = bisect.bisect_right(starts, m['timestamp'])
    if i == 0 or i >= len(starts):
        continue
    slices[seqs[i]].append(m)

# Hermes pre-call transforms whose effect we must mirror
def is_thinking_only(m):
    """AIAgent._is_thinking_only_assistant (approx): assistant, no tool_calls,
    empty/no content, has reasoning."""
    if m['role'] != 'assistant':
        return False
    tcs = json.loads(m['tool_calls'] or '[]')
    if tcs:
        return False
    if (m['content'] or '').strip():
        return False
    return bool(m['reasoning'] or m['reasoning_content'])


dropped_thinking_only = []
healed_empty = []
for q in seqs:
    lst = slices[q]
    kept = []
    for m in lst:
        if is_thinking_only(m):
            dropped_thinking_only.append(m['id'])
            continue
        # repair_empty_non_final_messages: empty non-final assistant stub
        if (m['role'] == 'assistant' and not (m['content'] or '').strip()
                and not (m['tool_calls'] or '[]').strip('[]')
                and m is not lst[-1]):
            healed_empty.append(m['id'])
        kept.append(m)
    slices[q] = kept

# ------------------------------------------------------ wire message builder
def wire_messages_for_call(seq):
    """Cumulative OpenAI-format message list Hermes sent for call `seq`."""
    wm = [{'role': 'system', 'content': SYSTEM_PROMPT}]
    for q in seqs:
        if q > seq:
            break
        for m in slices[q]:
            r = m['role']
            if r == 'user':
                wm.append({'role': 'user', 'content': m['content'] or ''})
            elif r == 'assistant':
                tcs = []
                for tc in json.loads(m['tool_calls'] or '[]'):
                    fn = tc.get('function') or {}
                    tcs.append({'id': tc.get('id'), 'type': 'function',
                                'function': {'name': fn.get('name'),
                                             'arguments': fn.get('arguments')}})
                a = {'role': 'assistant', 'content': m['content'] or ''}
                if m['reasoning_content'] is not None:
                    a['reasoning_content'] = m['reasoning_content']
                if tcs:
                    a['tool_calls'] = tcs
                wm.append(a)
            elif r == 'tool':
                wm.append({'role': 'tool', 'content': m['content'] or '',
                           'tool_call_id': m['tool_call_id']})
    return wm


# ------------------------------------------------------- exo production path
def strip_v4_thinking_markers(content):
    block = re.compile(THINK_S + ".*?" + THINK_E, re.DOTALL)
    if not content:
        return content
    cleaned = block.sub('', content)
    return cleaned.replace(THINK_S, '').replace(THINK_E, '')


def to_json(v):
    try:
        return json.dumps(v, ensure_ascii=False)
    except Exception:
        return json.dumps(v, ensure_ascii=True)


def encode_arguments_to_dsml(arguments_str):
    try:
        arguments = json.loads(arguments_str)
    except Exception:
        arguments = {'arguments': arguments_str}
    parts = []
    for k, v in arguments.items():
        parts.append('<{d}parameter name="{k}" string="{s}">{v}</{d}parameter>'
                     .format(d=DSML, k=k,
                             s='true' if isinstance(v, str) else 'false',
                             v=v if isinstance(v, str) else to_json(v)))
    return '\n'.join(parts)


def render_message(idx, messages):
    """vendor/deepseek_v4_encoding.render_message — thinking mode,
    drop_thinking=False (tools in play), reasoning_effort low."""
    p = ''
    msg = messages[idx]
    role = msg.get('role')
    content = msg.get('content')
    tool_calls = msg.get('tool_calls')
    reasoning_content = msg.get('reasoning_content')
    tools = msg.get('tools')

    if tools:
        tools = [t['function'] for t in tools]

    if role == 'system':
        p += content or ''
        if tools:
            p += '\n\n' + TOOLS_TEMPLATE.format(
                dsml=DSML, tstart=THINK_S, tend=THINK_E,
                tool_schemas='\n'.join(to_json(
                    {'name': t['function']['name'],
                     'description': t['function'].get('description', ''),
                     'parameters': t['function'].get('parameters', {})}
                    if 'function' in t else t) for t in tools))
    elif role == 'user':
        p += USER
        blocks = msg.get('content_blocks')
        if blocks:
            parts = []
            for b in blocks:
                if b['type'] == 'text':
                    parts.append(b.get('text', ''))
                elif b['type'] == 'tool_result':
                    parts.append('<tool_result>' + b.get('content', '')
                                 + '</tool_result>')
                else:
                    parts.append('[Unsupported ' + b['type'] + ']')
            p += '\n\n'.join(parts)
        else:
            p += content or ''
    elif role == 'assistant':
        tc_content = ''
        if tool_calls:
            tc_list = []
            for tc in tool_calls:
                tc_list.append('<' + DSML + 'invoke name="' + tc['function']['name']
                               + '">\n' + encode_arguments_to_dsml(tc['function']['arguments'])
                               + '\n</' + DSML + 'invoke>')
            tc_content += '\n\n<{d}tool_calls>\n{t}\n</{d}tool_calls>'.format(
                d=DSML, t='\n'.join(tc_list))
        thinking_part = (reasoning_content or '') + THINK_E
        p += thinking_part + (content or '') + tc_content + EOS
    else:
        raise NotImplementedError(role)

    # vendor transition logic (deepseek_v4_encoding.render_message): header is
    # appended when this is the LAST message, or when the NEXT message is an
    # assistant/latest_reminder (pre-filling the header for that assistant turn);
    # skipped when the next message is user/tool (they carry their own prefix).
    nxt = messages[idx + 1] if idx + 1 < len(messages) else None
    if nxt is not None and nxt.get('role') not in ('assistant', 'latest_reminder'):
        return p
    if role in ('user', 'developer'):
        p += ASSISTANT + THINK_S
    return p


def merge_and_sort(messages):
    merged = []
    for msg in messages:
        role = msg.get('role')
        if role == 'tool':
            blk = {'type': 'tool_result',
                   'tool_use_id': msg.get('tool_call_id', ''),
                   'content': msg.get('content', '')}
            if merged and merged[-1]['role'] == 'user' and 'content_blocks' in merged[-1]:
                merged[-1]['content_blocks'].append(blk)
            else:
                merged.append({'role': 'user', 'content_blocks': [blk]})
        elif role == 'user':
            tbs = [{'type': 'text', 'text': msg.get('content', '')}]
            if (merged and merged[-1]['role'] == 'user'
                    and 'content_blocks' in merged[-1]
                    and merged[-1].get('task') is None):
                merged[-1]['content_blocks'].extend(tbs)
            else:
                merged.append({'role': 'user', 'content': msg.get('content', ''),
                               'content_blocks': tbs})
        else:
            merged.append(msg)
    last_order = {}
    for msg in merged:
        if msg['role'] == 'assistant' and msg.get('tool_calls'):
            last_order = {}
            for i2, tc in enumerate(msg['tool_calls']):
                tid = tc.get('id') or ''
                if tid:
                    last_order[tid] = i2
        elif msg['role'] == 'user' and msg.get('content_blocks'):
            tbs = [b for b in msg['content_blocks'] if b['type'] == 'tool_result']
            if len(tbs) > 1 and last_order:
                srt = sorted(tbs, key=lambda b: last_order.get(b.get('tool_use_id', ''), 0))
                k = 0
                nb = []
                for b in msg['content_blocks']:
                    if b['type'] == 'tool_result':
                        nb.append(srt[k]); k += 1
                    else:
                        nb.append(b)
                msg['content_blocks'] = nb
    return merged


TASK_TOOLS = [{'type': 'function', 'function': {
    'name': '__probe_placeholder__', 'description': 'placeholder',
    'parameters': {'type': 'object', 'properties': {}}}}]

# The exact system-prompt text Hermes sent is not persisted (sessions.system_prompt
# is empty; no system rows in messages). It is byte-stable for the life of the
# conversation (Hermes cache-sacred invariant), sits entirely BEFORE the compared
# region, and therefore cancels in the common-prefix computation. Placeholder only
# keeps the render structure faithful.
SYSTEM_PROMPT = '[SYSTEM PROMPT NOT PERSISTED — placeholder; cancels in prefix]'


def exo_prompt(seq):
    """Full production render for the prompt of call `seq`."""
    wm = wire_messages_for_call(seq)
    ctm = []
    for msg in wm:
        if msg['role'] == 'system':
            ctm.append({'role': 'system', 'content': msg['content']})
            continue
        if (msg.get('content') in (None, '') and msg.get('reasoning_content') is None
                and msg.get('tool_calls') is None):
            continue
        d = {'role': msg['role'], 'content': msg.get('content')}
        if msg.get('reasoning_content') is not None:
            d['reasoning_content'] = msg['reasoning_content']
        if msg.get('tool_calls'):
            d['tool_calls'] = msg['tool_calls']
        if msg.get('tool_call_id') is not None:
            d['tool_call_id'] = msg['tool_call_id']
        ctm.append(d)
    sys_parts, non_sys = [], []
    for msg in ctm:
        if msg['role'] in ('system', 'developer'):
            if msg.get('content'):
                sys_parts.append(msg['content'])
        else:
            non_sys.append(msg)
    formatted = ([{'role': 'system', 'content': '\n'.join(sys_parts)}]
                 if sys_parts else []) + non_sys
    v4_messages = []
    for msg in formatted:
        m2 = dict(msg)
        if m2.get('role') == 'assistant' and isinstance(m2.get('content'), str):
            m2['content'] = strip_v4_thinking_markers(m2['content'])
        v4_messages.append(m2)
    for msg in v4_messages:
        if msg['role'] in ('system', 'developer'):
            msg['tools'] = TASK_TOOLS
            break
    full = merge_and_sort(v4_messages)
    out = BOS
    for idx in range(len(full)):
        out += render_message(idx, full)
    return out


# ----------------------------------------------------- decode-output proxy
def decode_proxy_text(seq_next):
    """Raw decode text of the completion produced by call seq_next-1 (PROXY).

    reasoning + content stored verbatim (exact). DSML tool-call block REBUILT
    from parse_dsml_output's re-serialized arguments -> identical to re-feed
    DSML BY CONSTRUCTION (raw emitted JSON spacing not persisted; optimistic).
    EOS appended (decode emits it; usage counts it)."""
    m = slices[seq_next][0]
    assert m['role'] == 'assistant'
    parts = [(m['reasoning'] or ''), THINK_E, (m['content'] or '')]
    tcs = json.loads(m['tool_calls'] or '[]')
    if tcs:
        # Variant A (primary): ONE tool_calls block containing all invokes
        # separated by '\n' - mirrors what the re-feed renders for a single
        # assistant.tool_calls array. The raw decode emission shape (one block
        # vs one block PER invoke) is not persisted; variant B (lower bound)
        # renders one block per invoke. See report.
        invs = ['<' + DSML + 'invoke name="' + tc['function']['name'] + '">\n'
                + encode_arguments_to_dsml(tc['function']['arguments'])
                + '\n</' + DSML + 'invoke>' for tc in tcs]
        parts.append('\n\n<' + DSML + 'tool_calls>\n' + '\n'.join(invs)
                     + '\n</' + DSML + 'tool_calls>')
    parts.append(EOS)
    return ''.join(parts)


def decode_proxy_text_multiblock(seq_next):
    """Variant B (lower bound): one <｜DSML｜tool_calls> block PER invoke."""
    m = slices[seq_next][0]
    parts = [(m['reasoning'] or ''), THINK_E, (m['content'] or '')]
    for tc in json.loads(m['tool_calls'] or '[]'):
        parts.append('\n\n<' + DSML + 'tool_calls>\n<' + DSML + 'invoke name="'
                     + tc['function']['name'] + '">\n'
                     + encode_arguments_to_dsml(tc['function']['arguments'])
                     + '\n</' + DSML + 'invoke>\n</' + DSML + 'tool_calls>')
    parts.append(EOS)
    return ''.join(parts)


# ------------------------------------------------------------------ main run
tok_cache = {}


def tcached(seq):
    if seq not in tok_cache:
        tok_cache[seq] = T(exo_prompt(seq))
    return tok_cache[seq]


pairs = []
for i in range(len(seqs) - 1):
    n, n1 = seqs[i], seqs[i + 1]
    if not slices[n1] or slices[n1][0]['role'] != 'assistant':
        continue
    pn = tcached(n)
    pn1 = tcached(n1)
    cp = 0
    for a, b in zip(pn, pn1):
        if a != b:
            break
        cp += 1
    tail_not_shared = len(pn) - cp          # expect 2 (the Assistant header)
    D = T(decode_proxy_text(n1))
    tgt = pn1[cp:]                           # header tokens matched by cp
    lcp = 0
    for a, b in zip(D, tgt):
        if a != b:
            break
        lcp += 1
    D_B = T(decode_proxy_text_multiblock(n1))
    lcp_b = 0
    for a, b in zip(D_B, tgt):
        if a != b:
            break
        lcp_b += 1
    m = slices[n1][0]
    if lcp >= len(D):
        cause = 'full-match'
        d_txt = r_txt = None
    else:
        d_txt = tok.decode(D[max(0, lcp - 8):lcp + 10])
        r_txt = tok.decode(tgt[max(0, lcp - 8):lcp + 10])
        if (m['reasoning_content'] or '') == ' ' and not (m['reasoning'] or ''):
            cause = 'reasoning-echo-space-pad'
        elif lcp == 0:
            cause = 'divergence-at-region-start'
        else:
            cause = 'text-divergence'
    pairs.append({
        'turn_pair_id': f'{n}->{n1}',
        'call_seq_prev': n, 'call_seq_next': n1,
        'decode_len_tokens': len(D),
        'decode_len_tokens_variant_b_multiblock': len(D_B),
        'lcp_tokens_variant_b_multiblock': lcp_b,
        'lcp_coverage_variant_b': round(lcp_b / len(D_B), 6) if D_B else None,
        'completion_tokens_recorded': out_tok[n],
        'prompt_len_prev_reconstructed': len(pn),
        'prompt_len_next_reconstructed': len(pn1),
        'prompt_tokens_prev_recorded': prompt_tok_rec[n],
        'prompt_tokens_next_recorded': prompt_tok_rec[n1],
        'common_prefix_prompt_n_to_n1': cp,
        'tail_not_shared_expected_2': tail_not_shared,
        'lcp_tokens': lcp,
        'lcp_coverage': round(lcp / len(D), 6) if D else None,
        'first_divergence_index': (cp + 2 + lcp) if lcp < len(D) else None,
        'divergence_cause': cause,
        'divergence_decode_text': d_txt,
        'divergence_refeed_text': r_txt,
        'decode_had_reasoning': bool(m['reasoning']),
        'reasoning_echo': ('verbatim' if m['reasoning'] == (m['reasoning_content'] or None)
                           else ('space-pad' if (m['reasoning_content'] or '') == ' '
                                 and not m['reasoning'] else 'other')),
        'finish_reason': m['finish_reason'],
    })

print(f'pairs computed: {len(pairs)}')
lens = [p['lcp_coverage'] for p in pairs]
lcps = [p['lcp_tokens'] for p in pairs]
q10 = statistics.quantiles(lens, n=10)
q4 = statistics.quantiles(lens, n=4)
print('coverage: min %.4f p10 %.4f p25 %.4f median %.4f mean %.4f max %.4f' % (
    min(lens), q10[0], q4[0], statistics.median(lens), statistics.mean(lens), max(lens)))
print('LCP tokens: min %d median %d max %d' % (min(lcps), statistics.median(lcps), max(lcps)))
print('causes:', Counter(p['divergence_cause'] for p in pairs))
inv = Counter(p['tail_not_shared_expected_2'] for p in pairs)
print('tail-not-shared (expect all 2):', inv.most_common(5))
bad = [p for p in pairs if abs(p['decode_len_tokens'] - p['completion_tokens_recorded'])
       > max(0.15 * p['completion_tokens_recorded'], 30)]
print('decode-len vs recorded: off pairs:', len(bad))
for p in bad[:15]:
    print('  ', p['turn_pair_id'], p['decode_len_tokens'], p['completion_tokens_recorded'],
          p['divergence_cause'])
# prompt length reconstruction check (system prompt unknown -> expect const offset)
offs = [p['prompt_tokens_prev_recorded'] - p['prompt_len_prev_reconstructed'] for p in pairs]
print('prompt offset (recorded - reconstructed): min %d max %d' % (min(offs), max(offs)))
print('dropped thinking-only rows:', dropped_thinking_only)
print('healed empty rows:', healed_empty)

json.dump({'n_pairs_reconstructed': len(pairs), 'per_pair': pairs,
           'note_dropped_thinking_only': dropped_thinking_only},
          open(OUT_JSON, 'w'), indent=1)
print('wrote', OUT_JSON)