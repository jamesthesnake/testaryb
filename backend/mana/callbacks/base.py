from dataclasses import dataclass
from typing import Any, Union

from pydantic import BaseModel


@dataclass(kw_only=True)
class BaseCallbackHandler:
    def on_llm_start(self, **kwargs: Any) -> None:
        """Run when LLM starts running."""

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""

    def on_llm_end(self, response: str, **kwargs: Any) -> None:
        """Run when LLM ends running."""

    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        """Run when LLM errors."""


@dataclass(kw_only=True)
class PydanticCallbackHandler(BaseCallbackHandler):
    def on_llm_end(self, response: BaseModel, **kwargs: Any) -> None:
        """Run when LLM ends running."""
