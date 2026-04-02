import requests
import re
import numpy as np
from typing import Optional
from src.cell2phys.config import Config


class PhysioBrain:
    """
    LLM interface for kinetic parameter inference.
    Connects to an OpenAI-compatible server (vLLM, Ollama, TGI, etc.).
    Uses Adaptive Semantic Caching (ASC) to minimize redundant LLM calls.
    """

    def __init__(self):
        self.api_base = Config.LLM_API_BASE
        self.api_key = Config.LLM_API_KEY
        self.model = Config.LLM_MODEL_NAME
        print(f"   [LLM] Connecting to {self.api_base} (model: {self.model})")

    def think_and_decide(
        self,
        system_prompt: str,
        user_prompt: str,
        low: float = 0.1,
        high: float = 5.0,
    ) -> float:
        """
        Queries the LLM for a kinetic parameter, bounded to [low, high].
        Checks ASC cache first; stores result on cache miss.
        """
        context_key = f"{system_prompt} ||| {user_prompt}"
        from src.cell2phys.utils.asc_engine import asc_engine

        # 1. Cache lookup
        cached = asc_engine.retrieve(context_key)
        if cached is not None:
            return float(np.clip(cached, low, high))

        # 2. LLM inference
        result = self._call_llm(system_prompt, user_prompt)

        # 3. Bound enforcement
        result = float(np.clip(result, low, high))

        # 4. Cache store
        asc_engine.store(context_key, result)
        return result

    def _call_llm(self, system_prompt: str, user_prompt: str) -> float:
        """Calls the OpenAI-compatible chat completions API."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 20,
        }
        try:
            resp = requests.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            match = re.search(r"[-+]?\d*\.?\d+", content)
            if match:
                return float(match.group())
            print(f"   [LLM] Could not parse response: {content!r}, defaulting to 1.0")
            return 1.0
        except Exception as e:
            raise RuntimeError(
                f"LLM server error: {e}. "
                f"Ensure your LLM server is running at {self.api_base}. "
                f"See README for setup instructions."
            )


# Lazy singleton
_brain: Optional[PhysioBrain] = None


def get_brain() -> PhysioBrain:
    global _brain
    if _brain is None:
        _brain = PhysioBrain()
    return _brain
