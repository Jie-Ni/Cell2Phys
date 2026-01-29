import requests
import json
import re
import time
from enum import Enum
from typing import Optional
from src.cell2phys.config import Config, ExecutionMode

class PhysioBrain:
    """
    Handles communication with the Large Language Model interface.
    Uses real LLM server for parameter inference.
    """
    
    def __init__(self):
        self.config = Config
        self.mode = Config.MODE
        self.api_base = Config.LLM_API_BASE
        self.api_key = Config.LLM_API_KEY
        self.model = Config.LLM_MODEL_NAME
        print(f"   LLM Client initialized in {self.mode.value} mode.")
        print(f"   LLM API: {self.api_base}")
        
    def think_and_decide(self, system_prompt: str, user_prompt: str, output_type: str = "float") -> float:
        """
        Sends a prompt to the LLM and retrieves a structured parameter.
        Uses Adaptive Semantic Caching (ASC) to minimize latency.
        """
        # 1. Check Cache (ASC)
        context_key = f"{system_prompt} ||| {user_prompt}"
        
        from src.cell2phys.utils.asc_engine import asc_engine
        
        cached_val = asc_engine.retrieve(context_key)
        if cached_val is not None:
            return cached_val

        # 2. Real LLM Inference
        result = self._real_inference(system_prompt, user_prompt)
            
        # 3. Store in Cache
        asc_engine.store(context_key, result)
        
        return result
            
    def _real_inference(self, system_prompt: str, user_prompt: str) -> float:
        """
        Calls the OpenAI-compatible API on the server.
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # Enforce JSON mode if supported, otherwise just prompt engineering
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.0,
            "max_tokens": 10
        }
        
        try:
            response = requests.post(f"{self.api_base}/chat/completions", headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            
            content = response.json()['choices'][0]['message']['content']
            
            # Parse the float from the string (Robustness)
            match = re.search(r"[-+]?\d*\.\d+|\d+", content)
            if match:
                return float(match.group())
            else:
                print(f"⚠️ LLM Output Parsing Failed: {content}. Fallback to 1.0")
                return 1.0
                
        except Exception as e:
            error_msg = f"❌ LLM Connection Error: {e}. Please ensure LLM server is running at {self.api_base}"
            print(error_msg)
            raise RuntimeError(error_msg)

# Singleton Instance
brain = PhysioBrain()
