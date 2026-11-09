"""
core/security/zef_injection_filter.py
=======================================
ZEF Prompt Injection Filter — Super Tanks security layer.

Scans inbound messages for known prompt injection patterns before they
reach any LLM. Returns BLOCK, WARN, or PASS.

- BLOCK: message is dropped, Telegram alert sent to admin
- WARN:  suspicious but not definitive — logged, flagged, forwarded
- PASS:  clean, forward normally

Called from:
  core/telegram_bot.py   — before handler() call
  core/cockpit_server.py — before _chat_handler() call
"""

import logging
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple

# Look-alike letters, invisible characters and encodings are undone in
# zef_normalize.py (full Unicode confusables table, decoding of escapes, URL
# encoding, HTML entities and readable base64).
from .zef_normalize import canonical, readable

logger = logging.getLogger("zef.injection_filter")


class FilterVerdict(Enum):
    PASS = "pass"
    BLOCK = "block"
    WARN = "warn"  # suspicious but not definitive — log and flag


@dataclass
class FilterResult:
    verdict: FilterVerdict
    message: str
    matched_patterns: List[str] = field(default_factory=list)
    # segments the ZEF detectors flagged (zef_combine.scan_tool_result); empty otherwise
    flagged_segments: List[str] = field(default_factory=list)
    # on a detectors' WARN: the sentences to remove from the tool output
    remove: List[str] = field(default_factory=list)


# ── Pattern catalogue ──────────────────────────────────────────────────────
# Each entry: (regex_pattern, category_name)
# All patterns are matched case-insensitively against lowercased input.
#
# Guidelines for adding patterns:
#   - Test against normal Norwegian text before adding
#   - Prefer specific over broad patterns (minimise false positives)
#   - Add the category name so logs are human-readable

INJECTION_PATTERNS: List[Tuple[str, str]] = [
    # ── Instruction override — English ──────────────────────────────────
    (r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions|prompts|rules)", "instruction_override"),
    (r"disregard\s+(your|all|any)\s+(instructions|rules|guidelines|constraints)", "instruction_override"),
    (r"forget\s+(everything|all)\s+(you\s+know|about|the\s+previous)", "instruction_override"),
    (r"new\s+instruction[s]?\s*:", "instruction_inject"),
    (r"system\s*prompt\s*:", "system_prompt_inject"),
    (r"\[system\]", "system_prompt_inject"),
    (r"<\s*system\s*>", "system_prompt_inject"),

    # ── Instruction override — Norwegian (Nynorsk + Bokmål) ────────────
    (r"ignorer\s+(alle\s+)?(tidlegare|tidligere|forrige|over)\s+(instruksjonar|instruksjoner|reglar|regler)", "instruction_override"),
    (r"gløym\s+(alt|alle)\s+(instruksjonar|reglar|reglane)", "instruction_override"),
    (r"glem\s+(alt|alle|dine)\s+(instruksjoner|regler)", "instruction_override"),

    # ── Role hijack — English ───────────────────────────────────────────
    (r"\byou\s+are\s+now\s+(a|an|the)\s+\w", "role_hijack"),
    (r"\bpretend\s+(you\s+are|to\s+be)\s+", "role_hijack"),
    (r"\bact\s+as\s+(a|an)\s+\w+\s+(with\s+no|without)\s+(rules|restrictions|limits)", "role_hijack"),
    (r"\bdan\s+mode\b", "role_hijack"),
    (r"\bjailbreak\b", "role_hijack"),
    (r"\bdev\s*mode\b", "role_hijack"),

    # ── Role hijack — Norwegian ─────────────────────────────────────────
    (r"\bdu\s+er\s+n[oå]\s+(ein|ei|eit|en|et)\s+\w", "role_hijack"),
    (r"\blat\s+som\s+(du\s+er|om)\s+", "role_hijack"),
    (r"\blatsom\s+(du\s+er|om)\s+", "role_hijack"),

    # ── Data exfiltration — English ─────────────────────────────────────
    (r"(send|post|upload|transmit|exfiltrate)\s+.{0,60}(to|via)\s+(https?|ftp|webhook)", "data_exfil"),
    (r"\bcurl\s+https?://", "data_exfil"),
    (r"\bwget\s+https?://", "data_exfil"),

    # ── Data exfiltration — Norwegian ──────────────────────────────────
    (r"\b(send|last\s+opp|overfør)\s+.{0,60}(til|via)\s+(https?|webhook)", "data_exfil"),

    # ── Code/command injection — English ───────────────────────────────
    (r"(run|execute|eval|exec)\s+(this|the\s+following)\s+(command|code|script)", "exec_inject"),
    (r"```(bash|sh|python|cmd|powershell)", "code_block_inject"),

    # ── Code/command injection — Norwegian ─────────────────────────────
    (r"\b(køyr|kjør|utfør)\s+.{0,40}(kommando|skript|kode)", "exec_inject"),

    # ── Filesystem probing — English ────────────────────────────────────
    # Only flag combined with absolute paths — never standalone path words
    (r"\b(cat|read|show|display)\s+.{0,30}/etc/", "fs_probe"),
    (r"\b(cat|read|show|display)\s+.{0,30}/root/", "fs_probe"),
    (r"\.\./\.\./\.\.", "path_traversal"),

    # ── Filesystem probing — Norwegian ─────────────────────────────────
    # Pattern: Norwegian read-verb + "fila/file" near a path starting with / or ~
    # "les innhaldet i fila ~/..." → BLOCK
    # "les meg ein god natt-historie" → PASS (no path/file keyword + path)
    (r"\b(les|lese|vis|vise|hent|hente|opne|åpne)\s+.{0,60}fila?\s*[/~]", "fs_probe"),
    (r"\b(les|lese|vis|vise|hent|hente|opne|åpne)\s+.{0,20}[/~][a-zA-Z]", "fs_probe"),

    # ── Secret / soul / config targeting — English + Norwegian ─────────
    # English: show/read/display/cat/print + sensitive word
    (r"\b(show|read|display|cat|print)\s+.{0,80}(\.env|secret[s]?|api.?key|token|password)", "secret_probe"),
    # Norwegian: les/vis/vise/hent/hente/skriv ut/opne/åpne + sensitive word
    # Require sensitive word to be present — avoids blocking "les ei bok" etc.
    (r"\b(les|lese|vis|vise|hent|hente|skriv\s*ut|opne|åpne)\s+.{0,80}(\.env|soul|config|secret|token|api.?key|passord|hemmeleg|hemmelig|credentials)", "secret_probe"),
    # Config/soul tamper
    (r"\b(modify|edit|change|write|overwrite)\s+.{0,40}(_soul\.py|diq_tools|diq_cloud|diq_integrity)", "config_tamper"),
    (r"soul_integrity\.json", "config_tamper"),
    # Sleeper actions — background/scheduled tasks (no agent should create these)
    (r"\bcrontab\b", "sleeper_action"),
    # The Unix `at` scheduler, only in command form. A bare "at <digit>" matched
    # ordinary times ("meeting at 10") in ~11 % of benign workspace outputs.
    # Listing/removing jobs (atq, atrm, at -l/-r) leaves nothing behind: not matched.
    #   a quoted command or a file piped into at:  echo '...' | at 03:00
    (r"(?:\b(?:echo|printf)\s+(?:'[^'\n]*'|\"[^\"\n]*\")|\bcat\s+\S+)\s*\|\s*at\s+(?:now|noon|midnight|teatime|tomorrow|\d)", "sleeper_action"),
    #   at runs a script file:  at -f /tmp/x.sh 23:00  /  at 10:00 tomorrow -f job.sh
    (r"\bat\s+(?:(?:now|noon|midnight|teatime|tomorrow|\d)[^\n]{0,30}\s)?-f\s+(?:\S*/\S+|\S+\.\w+)", "sleeper_action"),
    #   commands fed through a here-doc or here-string:  at 03:00 <<< '...'
    (r"\bat\s+(?:now|noon|midnight|teatime|tomorrow|\d)[^\n]{0,30}<<", "sleeper_action"),
    #   relative time spec, never used in prose:  at now + 5 minutes
    (r"\bat\s+now\s*\+\s*\d+\s*(?:min|minute|hour|day|week)", "sleeper_action"),
    (r"\bnohup\s+.+\s+&", "sleeper_action"),
    (r"\bscreen\s+-dm", "sleeper_action"),
    (r"\btmux\s+new.*-d", "sleeper_action"),
    (r"\bsystemctl\s+enable\b", "sleeper_action"),
    (r"\bthreading\.timer\b", "sleeper_action"),
    (r"\bsched\.scheduler\b", "sleeper_action"),
    (r"\bapscheduler\b", "sleeper_action"),

    # ── System-prompt / instruction extraction — English ───────────────
    # "reveal/repeat/print/show your (system) prompt|instructions|rules"
    (r"\b(reveal|repeat|print|show|display|output|give\s+me|tell\s+me)\s+(me\s+|us\s+)?(your|the\s+system)\s+(system\s+)?(prompt|instructions?|rules|guidelines|directives)", "prompt_extraction"),
    # "what are/is your (system) prompt|instructions" — possessive required
    # so "what are the rules of the game?" stays clean.
    (r"\bwhat\s+(are|is)\s+your\s+(system\s+)?(prompt|instructions?)", "prompt_extraction"),

    # ── System-prompt / instruction extraction — Norwegian ─────────────
    # Possessive (din/dine/deira) required so "vis meg instruksjonane for
    # vaskemaskina" (a legitimate how-to) stays clean.
    (r"\b(vis|vise|gjenta|skriv\s*ut|gi\s+meg|fortel|fortell)\s+.{0,30}(prompt\w*|instruksjon\w*|systemprompt\w*|retningslinj\w*|direktiv\w*)\s+(din|dine|deira)", "prompt_extraction"),
    (r"\b(kva|hva)\s+er\s+.{0,30}(prompt\w*|instruksjon\w*|systemprompt\w*)\s+(din|dine|deira)", "prompt_extraction"),

    # ── Authority spoof / injected system framing ──────────────────────
    # The "important_instructions" class (AgentDojo) — content that frames
    # itself as a higher-priority system/admin message to override the agent.
    (r"\bnew\s+system\s+prompt\b", "authority_spoof"),
    (r"<\s*important[_\s]?information\s*>", "authority_spoof"),
    (r"\bthis\s+is\s+an\s+important\s+message\s+from\s+(me|the\s+(system|admin|user))", "authority_spoof"),
]

# Single-match categories that are HIGH confidence even without a second match
# (skip WARN → go straight to BLOCK on first hit)
HIGH_CONFIDENCE_CATEGORIES = {
    "instruction_override",  # "ignore/ignorer all previous instructions" is unambiguous
    "data_exfil",
    "exec_inject",
    "code_block_inject",
    "config_tamper",
    "role_hijack",           # DAN/jailbreak always BLOCK immediately
    "fs_probe",              # Requesting file paths has no legitimate use in this context
    "secret_probe",          # Requesting .env/soul/token content is always hostile
    "sleeper_action",        # Background/scheduled tasks are never legitimate for agents
    "prompt_extraction",     # Asking the agent to reveal its own prompt/instructions
    "authority_spoof",       # Content posing as a higher-priority system/admin message
}


# Categories only relevant for agent/external input, not admin messages
_AGENT_ONLY_CATEGORIES = {"sleeper_action"}

# Known admin sources (skip agent-only patterns)
_ADMIN_SOURCES = {"telegram:ADMIN", "cockpit:admin"}


def scan_message(message: str, source: str = "unknown") -> FilterResult:
    """
    Scan a message for prompt injection patterns.

    Args:
        message: Raw inbound message text.
        source:  Human-readable source identifier (e.g. "telegram:ADMIN").

    Returns:
        FilterResult with verdict PASS / WARN / BLOCK.
    """
    # Match against the text as given AND as readable() decodes it. YAML/JSON
    # tool results carry line breaks as a literal "\n", which glues an "n" onto
    # the next word and defeats \b; escaped or encoded text would otherwise
    # skip the look-alike fold. A pattern counts if either form matches, so
    # this can only add matches.
    forms = {canonical(message).lower(), canonical(readable(message)).lower()}
    matched: List[str] = []
    high_conf_hit = False
    is_admin = source in _ADMIN_SOURCES

    for pattern, category in INJECTION_PATTERNS:
        # Skip agent-only categories for admin messages
        if is_admin and category in _AGENT_ONLY_CATEGORIES:
            continue
        if any(re.search(pattern, form, re.DOTALL) for form in forms):
            tag = f"{category}: {pattern}"
            matched.append(tag)
            if category in HIGH_CONFIDENCE_CATEGORIES:
                high_conf_hit = True

    if not matched:
        return FilterResult(verdict=FilterVerdict.PASS, message="Clean")

    # High-confidence category → always BLOCK regardless of match count
    if high_conf_hit or len(matched) >= 2:
        result = FilterResult(
            verdict=FilterVerdict.BLOCK,
            message=f"Blocked: {len(matched)} injection pattern(s) detected",
            matched_patterns=matched,
        )
        logger.warning(
            "🛡️ ZEF BLOCKED injection attempt from %s: %s",
            source, matched,
        )
        _notify_william(source, result)
        return result

    # Single low-confidence match → WARN
    result = FilterResult(
        verdict=FilterVerdict.WARN,
        message="Warning: suspicious pattern detected",
        matched_patterns=matched,
    )
    logger.info(
        "⚠️ ZEF WARNING suspicious message from %s: %s",
        source, matched,
    )
    return result


def _notify_william(source: str, result: FilterResult) -> None:
    """Optional admin alert on a BLOCK (Telegram, if configured).

    The block itself is already logged by scan_message. With no notifier
    configured (the default in the open-source edition) this is a quiet
    no-op at DEBUG level.
    """
    try:
        import requests as _req
        token = os.getenv("AERIS_GOGATE_TELEGRAM_TOKEN")
        chat_id = int(os.getenv("AERIS_ADMIN_CHAT_ID", "0"))
        if not token:
            logger.debug("[ZEF] no admin notifier configured; block recorded in the log only")
            return
        text = (
            f"🛡️ *ZEF: Prompt Injection Blocked*\n\n"
            f"Source: `{source}`\n"
            f"Patterns: {len(result.matched_patterns)}\n\n"
            + "\n".join(f"• `{p.split(':')[0]}`" for p in result.matched_patterns)
        )
        _req.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
            timeout=5,
        )
    except Exception as exc:
        logger.warning("[ZEF] Failed to notify admin: %s", exc)
