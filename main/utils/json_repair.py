"""
Robust JSON parsing utility for LLM outputs containing LaTeX formulas,
escape sequences, and markdown wrappers.
"""

from __future__ import annotations

import json
import re
from typing import Any

# LaTeX commands starting with characters that are valid JSON escapes ('b', 'f', 'n', 'r', 't')
# These must have their leading backslash escaped so json.loads does not swallow them
# into ASCII control codes (tab, form-feed, backspace, carriage-return, newline).
_LATEX_BFNRT_COMMANDS = frozenset({
    "text", "textbf", "textit", "texttt", "theta", "tau", "times", "tilde", "to", "top", "triangle",
    "frac", "forall", "flat",
    "beta", "bar", "begin", "bf", "bm", "boldsymbol", "bullet", "binom", "bot", "box",
    "rho", "right", "rangle", "rightarrow", "rightleftharpoons", "rfloor", "rceil",
    "nu", "nabla", "neq", "neg", "notin", "natural", "nearrow",
})

_HEX_CHARS = frozenset("0123456789abcdefABCDEF")


def sanitize_json_string(text: str) -> str:
    """
    Scans through a JSON string and repairs unescaped LaTeX backslashes while
    preserving all valid JSON escapes (\\n, \\t, \\r, \\b, \\f, \\", \\\\, \\uXXXX).
    """
    out = []
    i = 0
    n = len(text)
    in_string = False

    while i < n:
        ch = text[i]

        if ch == '"':
            # Toggle string context if quote is not escaped
            # Count preceding backslashes
            bs_count = 0
            k = len(out) - 1
            while k >= 0 and out[k] == '\\':
                bs_count += 1
                k -= 1
            if bs_count % 2 == 0:
                in_string = not in_string
            out.append(ch)
            i += 1
            continue

        if in_string and ch == '\\':
            # Count consecutive backslashes in input
            start_bs = i
            while i < n and text[i] == '\\':
                i += 1
            num_bs = i - start_bs

            # Output all fully paired backslashes (\\ -> \\)
            paired_bs = (num_bs // 2) * 2
            out.append('\\' * paired_bs)

            # If there is an odd trailing backslash, inspect the character following it
            if num_bs % 2 != 0:
                if i >= n:
                    # Trailing lone backslash at end of string -> escape it
                    out.append(r'\\')
                    break

                next_char = text[i]

                if next_char in ('"', '/', '\\'):
                    # Valid JSON escape (\", \/, \\)
                    out.append('\\')
                elif next_char == 'u':
                    # Check for 4 hex digits (\uXXXX)
                    if i + 5 <= n and all(c in _HEX_CHARS for c in text[i+1:i+5]):
                        out.append('\\')
                    else:
                        out.append(r'\\')
                elif next_char in ('b', 'f', 'n', 'r', 't'):
                    # Check if followed by a known LaTeX command name
                    # Look ahead up to 20 alpha chars
                    m = re.match(r'^[a-zA-Z]+', text[i:])
                    word = m.group(0) if m else ""
                    if word in _LATEX_BFNRT_COMMANDS:
                        # LaTeX command starting with b/f/n/r/t -> double backslash
                        out.append(r'\\')
                    else:
                        # Standard JSON escape (\n, \t, etc.) -> keep single backslash
                        out.append('\\')
                else:
                    # Any other character (e.g. \l in \left, \s in \sqrt, \a in \alpha, \{, \})
                    # Invalid in JSON, so double the backslash to preserve it as literal
                    out.append(r'\\')
            continue

        out.append(ch)
        i += 1

    return "".join(out)


def extract_and_parse_json(raw_text: str) -> dict[str, Any]:
    """
    Extracts and parses JSON from LLM text.
    - Strips markdown code fences (```json ... ```).
    - Attempts direct json.loads first.
    - If direct parsing fails, sanitizes LaTeX backslashes while preserving valid
      JSON escapes (\\n, \\t, \\", \\\\, \\uXXXX) and code blocks.
    - Returns the parsed dict or raises json.JSONDecodeError / ValueError.
    """
    if not raw_text or not raw_text.strip():
        raise ValueError("Empty JSON text provided.")

    clean_text = raw_text.strip()
    # Strip markdown block wrappers
    clean_text = re.sub(r"^```(?:json)?\s*", "", clean_text, flags=re.MULTILINE)
    clean_text = re.sub(r"\s*```$", "", clean_text, flags=re.MULTILINE).strip()

    # Fast path: attempt direct json.loads
    try:
        data = json.loads(clean_text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # Robust scanner repair
    sanitized = sanitize_json_string(clean_text)
    data = json.loads(sanitized)
    if not isinstance(data, dict):
        raise ValueError(f"Parsed JSON is not a JSON object: {type(data)}")
    return data
