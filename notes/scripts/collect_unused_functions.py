#!/usr/bin/env python3
import json, os, re, subprocess, sys
from collections import defaultdict, Counter
from pathlib import Path
from typing import List, Optional, Dict, Any

WITH_CLIPPY = "--with-clippy" in sys.argv
OUT_TXT = Path("unused_symbols.txt")
OUT_JSONL = Path("unused_symbols.jsonl")  # machine-readable dump

# --- Matchers ---------------------------------------------------------------

# Any of these words must appear to consider the diag "unused-ish"
UNUSED_HINT = re.compile(r"\b(unused|never used|never read)\b", re.IGNORECASE)

# Extract names inside backticks: `name`
BACKTICKS = re.compile(r"`([^`]+)`")

# Role detection (so we can categorize)
ROLE_FUN = re.compile(r"\b(function|method|associated function|functions|methods|associated functions)\b", re.IGNORECASE)
ROLE_VAR = re.compile(r"\b(variable|variables|parameter|parameters)\b", re.IGNORECASE)
ROLE_FIELD = re.compile(r"\b(field|fields)\b", re.IGNORECASE)

# In practice, rustc uses:
#   - code "unused_variables" for variables/params
#   - code "dead_code" for unused/never-read private items including functions & fields
# But we rely primarily on the message text so it also works across toolchains.
def classify_message(msg_text: str) -> Optional[str]:
    if not UNUSED_HINT.search(msg_text):
        return None
    if ROLE_FUN.search(msg_text):
        return "function"
    if ROLE_VAR.search(msg_text):
        return "variable"
    if ROLE_FIELD.search(msg_text):
        return "field"
    # Sometimes the role is only in span labels or children; caller will try those too.
    return None

def extract_names(text: str) -> List[str]:
    # All backticked identifiers (handles multiples like: methods `a` and `b`)
    return list(dict.fromkeys(BACKTICKS.findall(text)))  # dedupe, preserve order

def pick_primary_span(spans: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not spans: return None
    for sp in spans:
        if sp.get("is_primary"): return sp
    return spans[0]

def relpath(p: Optional[str]) -> str:
    if not p: return "<unknown>"
    try:
        return str(Path(p).resolve().relative_to(Path.cwd()))
    except Exception:
        return p

# --- Cargo runners ----------------------------------------------------------

def run_cmd(cmd):
    try:
        return subprocess.run(cmd, check=False, capture_output=True, text=True, env=os.environ)
    except FileNotFoundError:
        print(f"Error: command not found: {cmd[0]}", file=sys.stderr)
        sys.exit(2)

def cargo_stream(kind: str):
    if kind == "check":
        cmd = ["cargo","check","--workspace","--all-targets","--all-features","--message-format=json"]
    else:
        cmd = ["cargo","clippy","--workspace","--all-targets","--all-features","--message-format=json"]
    proc = run_cmd(cmd)
    if proc.returncode not in (0,101):
        print(f"Warning: cargo {kind} exit code {proc.returncode} (parsing anyway)", file=sys.stderr)
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line:
            yield line

# --- Core collection --------------------------------------------------------

def try_classify_and_names(message_obj: Dict[str, Any]) -> (Optional[str], List[str]):
    # Check main message
    text = message_obj.get("message","") or ""
    role = classify_message(text)
    names = extract_names(text) if role else []

    # If we didn’t get a role or names, check span labels (often: “unused variable: `x`”)
    if not role or not names:
        for sp in message_obj.get("spans") or []:
            lab = sp.get("label","") or ""
            r = classify_message(lab)
            if r and not role:
                role = r
            if r:
                names.extend(extract_names(lab))

    # If still nothing, check children messages (notes/help)
    if not role or not names:
        for ch in message_obj.get("children") or []:
            m = ch.get("message","") or ""
            r = classify_message(m)
            if r and not role:
                role = r
            if r:
                names.extend(extract_names(m))

    # Dedupe names
    if names:
        names = list(dict.fromkeys(names))
    return role, names

def collect_unused():
    results = []  # for JSONL
    for kind in (["check"] + (["clippy"] if WITH_CLIPPY else [])):
        for line in cargo_stream(kind):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("reason") != "compiler-message":
                continue

            msg = obj.get("message") or {}
            if msg.get("level") != "warning":
                continue

            # Only messages that clearly indicate unused/never read
            role, names = try_classify_and_names(msg)
            if not role or not names:
                continue

            spans = msg.get("spans") or []
            p = pick_primary_span(spans)
            entry = {
                "role": role,                       # "function" | "variable" | "field"
                "names": names,                     # one or more
                "code": (msg.get("code") or {}).get("code") or "",
                "crate": (obj.get("target") or {}).get("name") or obj.get("package_id") or "<unknown-crate>",
                "file": relpath(p.get("file_name") if p else None),
                "line_start": p.get("line_start") if p else None,
                "col_start": p.get("column_start") if p else None,
                "message": msg.get("message",""),
            }
            results.append(entry)
    return results

# --- Reporting --------------------------------------------------------------

def write_reports(items: List[Dict[str, Any]]):
    # JSONL (full fidelity)
    with OUT_JSONL.open("w", encoding="utf-8") as jf:
        for it in items:
            jf.write(json.dumps(it, ensure_ascii=False) + "\n")

    # Aggregate by role → crate → file
    agg = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    counts = Counter()
    for it in items:
        role, crate, file = it["role"], it["crate"], it["file"]
        for name in it["names"]:
            agg[role][crate][file].append((name, it["line_start"], it["col_start"]))
            counts[(role)] += 1

    with OUT_TXT.open("w", encoding="utf-8") as f:
        total = sum(counts.values())
        f.write("# Unused Symbols Report (variables, fields, functions/methods)\n")
        f.write(f"# Workspace: {Path.cwd()}\n")
        f.write(f"# Total unused symbols: {total}\n")
        if total == 0:
            f.write("\nNo unused variables/fields/functions detected.\n")
            return

        f.write("\n## Totals by category\n")
        for role in ("variable","field","function"):
            f.write(f"- {role}: {counts.get(role, 0)}\n")

        # Print grouped listings
        for role in ("variable","field","function"):
            group = agg.get(role)
            if not group:
                continue
            f.write(f"\n## {role.capitalize()}s\n")
            for crate in sorted(group.keys()):
                f.write(f"== Crate: {crate} ==\n")
                for file in sorted(group[crate].keys()):
                    f.write(f"\n  File: {file}\n")
                    for (name, ls, cs) in sorted(group[crate][file], key=lambda x:(x[1] or 0, x[2] or 0, x[0])):
                        loc = f" (at {ls}:{cs})" if ls is not None else ""
                        f.write(f"    - {name}{loc}\n")
                f.write("\n")

def main():
    items = collect_unused()
    write_reports(items)
    print(f"Wrote: {OUT_TXT}")
    print(f"Also wrote: {OUT_JSONL}")

if __name__ == "__main__":
    main()