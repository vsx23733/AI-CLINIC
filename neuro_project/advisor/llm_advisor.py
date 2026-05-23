"""Stage 11 — LLM advisor over Ollama.

Uses Ollama's REST API (default http://localhost:11434) through stdlib urllib —
no requests dependency. POST /api/chat with messages + format='json' so the
model is constrained to valid JSON output.

Public surface:
  - OllamaConfig         : host, model, temperature, format, timeout
  - OllamaClient.chat()  : low-level call
  - advise(...)          : end-to-end : build_prompt + chat + return raw JSON text
  - ping()               : check that an Ollama server is reachable

Graceful degradation:
  - ConnectionError on a clear message if the server is down
  - OllamaError wraps non-2xx responses
"""
from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..gap_analysis import GapReport
from .prompt_builder import AdMetadata, build_prompt


# ---------------------------------------------------------------- config
@dataclass
class OllamaConfig:
    host:        str   = "http://localhost:11434"
    model:       str   = "gemma4:latest"        # any locally-pulled tag works
    temperature: float = 0.4               # lower = more deterministic advice
    format_json: bool  = True              # constrain output to valid JSON
    timeout_s:   float = 600.0             # 10 min: covers cold-start model load
    keep_alive:  str   = "10m"             # keep model resident between calls
    # Reasoning models (gemma4, qwen3, deepseek-r1...) emit a private "thinking"
    # trace BEFORE the answer. For structured JSON output we don't want that —
    # `think=False` makes Ollama skip it so all tokens go to the answer.
    think:       Optional[bool] = False
    # Generous default : with format='json' the model may still emit some
    # whitespace, and a few hundred tokens are needed for the JSON itself.
    num_predict: int   = 4096
    options:     Dict[str, Any] = field(default_factory=dict)


class OllamaError(RuntimeError):
    """Raised when Ollama returns a non-2xx response."""


# ---------------------------------------------------------------- client
class OllamaClient:
    def __init__(self, cfg: Optional[OllamaConfig] = None) -> None:
        self.cfg = cfg or OllamaConfig()

    # ---- low-level HTTP
    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = self.cfg.host.rstrip("/") + path
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, method="POST",
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.cfg.timeout_s) as r:
                body = r.read().decode("utf-8")
                return json.loads(body)
        except urllib.error.HTTPError as e:
            # Server reachable but returned an error (4xx/5xx)
            try:
                body = e.read().decode("utf-8", errors="replace")
            except Exception:
                body = ""
            raise OllamaError(
                f"Ollama HTTP {e.code} on {path} : {body[:300]}") from e
        except urllib.error.URLError as e:
            raise ConnectionError(
                f"Cannot reach Ollama at {self.cfg.host} ({e.reason}). "
                "Is the server running ? Try : `ollama serve`") from e
        except (socket.timeout, TimeoutError) as e:
            raise ConnectionError(
                f"Ollama timeout after {self.cfg.timeout_s}s") from e

    # ---- public
    def ping(self) -> bool:
        """Lightweight check : list local models."""
        url = self.cfg.host.rstrip("/") + "/api/tags"
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                return r.status == 200
        except Exception:
            return False

    def preload(self, model: Optional[str] = None) -> None:
        """Load the model into RAM without generating tokens. Sending an empty
        prompt with num_predict=0 makes Ollama load weights then return ; the
        next chat() call is then fast. Useful before a JSON-constrained call
        whose cold-start would otherwise hit the timeout."""
        payload = {
            "model": model or self.cfg.model,
            "prompt": "",
            "stream": False,
            "keep_alive": self.cfg.keep_alive,
            "options": {"num_predict": 0},
        }
        self._post("/api/generate", payload)

    def chat(
        self, system: str, user: str,
        model: Optional[str] = None,
    ) -> str:
        """Send a (system, user) pair, return the assistant string content."""
        opts: Dict[str, Any] = {
            "temperature": self.cfg.temperature,
            "num_predict": self.cfg.num_predict,
        }
        opts.update(self.cfg.options)
        payload: Dict[str, Any] = {
            "model":   model or self.cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "stream":  False,
            "options": opts,
            "keep_alive": self.cfg.keep_alive,
        }
        if self.cfg.format_json:
            payload["format"] = "json"
        if self.cfg.think is not None:
            payload["think"] = self.cfg.think       # ignored by non-reasoning models
        resp = self._post("/api/chat", payload)

        # Ollama returns {"message": {"role":"assistant","content":"...",
        #                              "thinking": "..."  # reasoning models only
        #                            }, "done_reason": "stop"|"length"|...}
        msg = resp.get("message") or {}
        content  = (msg.get("content")  or "").strip()
        thinking = (msg.get("thinking") or "").strip()
        done_reason = resp.get("done_reason", "")

        if not content and thinking:
            # Reasoning model spent all tokens in the thinking trace before
            # producing the answer. Try to salvage JSON from the thinking
            # text ; the parser is tolerant.
            return thinking
        if not content:
            hint = ""
            if done_reason == "length":
                hint = (" (done_reason=length : increase OllamaConfig.num_predict, "
                        "currently {})".format(self.cfg.num_predict))
            raise OllamaError(f"Empty response from Ollama{hint}. Full resp: {resp!r}")
        return content


# ---------------------------------------------------------------- top-level helper
def advise(
    report: GapReport,
    ad_metadata: AdMetadata,
    cfg: Optional[OllamaConfig] = None,
    extra_instructions: Optional[str] = None,
) -> str:
    """End-to-end : build prompt, call Ollama, return raw JSON text.
    The text is *expected* to be valid JSON (we set format='json'), but
    parsing + validation happen in `recommendations.parse_advice`."""
    client = OllamaClient(cfg)
    system, user = build_prompt(report, ad_metadata,
                                extra_instructions=extra_instructions)
    return client.chat(system, user)


def list_local_models(cfg: Optional[OllamaConfig] = None) -> List[str]:
    """Inventory of models pulled locally on the Ollama server."""
    cfg = cfg or OllamaConfig()
    url = cfg.host.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            data = json.loads(r.read().decode("utf-8"))
    except Exception:
        return []
    return [m.get("name", "") for m in data.get("models", []) if m.get("name")]
