from dataclasses import dataclass
from typing import Any

from backend.mana.callbacks.base import BaseCallbackHandler


@dataclass(kw_only=True)
class StdoutStreamingCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.streamed = False  # tracks if the response was streamed vs cached and printed at the end

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        self.streamed = True
        print(token, end="")

    def on_llm_end(self, response: str, **kwargs: Any) -> None:
        """Run when LLM ends running."""
        if not self.streamed:
            print(response, end="")
