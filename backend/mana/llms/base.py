import abc
from dataclasses import dataclass
from typing import Sequence

from langchain.prompts import PromptTemplate

from backend.mana.callbacks.base import BaseCallbackHandler

MAX_RETRIES = 3
MAX_VALIDATION_ERRORS = 3


@dataclass(kw_only=True, frozen=True)
class ChatModel(abc.ABC):
    prompt_template: PromptTemplate | None = None
    """The prompt template to use for the completion"""

    @abc.abstractmethod
    def complete(
            self,
            user_prompt: str,
            system_prompt: str | None = None,
            callbacks: Sequence[BaseCallbackHandler] | None = None,
    ) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    async def acomplete(
            self,
            user_prompt: str,
            system_prompt: str | None = None,
            callbacks: Sequence[BaseCallbackHandler] | None = None,
    ) -> str:
        raise NotImplementedError

    def complete_with_prompt_template(self, callbacks: Sequence[BaseCallbackHandler] | None = None, **kwargs) -> str:
        """
        Args:
            kwargs: The kwargs to use to fill in the prompt template
        Returns:
            The chat completion
        """
        if self.prompt_template is None:
            raise ValueError("Prompt template is not set")
        else:
            prompt = self.prompt_template.format(**kwargs)
            return self.complete(prompt, callbacks=callbacks)

    async def acomplete_with_prompt_template(
            self, callbacks: Sequence[BaseCallbackHandler] | None = None, **kwargs
    ) -> str:
        """
        This is the async version of complete_with_prompt_template
        """
        if self.prompt_template is None:
            raise ValueError("Prompt template is not set")
        else:
            prompt = self.prompt_template.format(**kwargs)
            return await self.acomplete(prompt, callbacks=callbacks)
