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
    timeout_s:   float = 120.0
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

    def chat(
        self, system: str, user: str,
        model: Optional[str] = None,
    ) -> str:
        """Send a (system, user) pair, return the assistant string content."""
        opts: Dict[str, Any] = {"temperature": self.cfg.temperature}
        opts.update(self.cfg.options)
        payload: Dict[str, Any] = {
            "model":   model or self.cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "stream":  False,
            "options": opts,
        }
        if self.cfg.format_json:
            payload["format"] = "json"
        resp = self._post("/api/chat", payload)
        # Ollama returns {"message": {"role":"assistant","content":"..."}}
        msg = resp.get("message") or {}
        content = msg.get("content")
        if not content:
            raise OllamaError(f"No content in Ollama response : {resp!r}")
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
