from dataclasses import dataclass
from typing import Generic, Optional, Sequence, Type, TypeVar

from instructor import AsyncInstructor, Instructor
from instructor.exceptions import IncompleteOutputException
from pydantic import BaseModel, ValidationError
from tenacity import (
    RetryCallState,
)

from backend.mana.callbacks.base import PydanticCallbackHandler, BaseCallbackHandler
from backend.mana.llms.base import ChatModel

# Define a type variable that can be any subclass of BaseModel
T = TypeVar("T", bound=BaseModel)


@dataclass(kw_only=True, frozen=True)
class BaseInstructor(ChatModel, Generic[T]):
    model: str
    """The model to use for the completion"""
    response_model: Type[T]
    """The response model to use."""
    temperature: float = 1.0
    """
    The temperature to use for the completion
    Some evidence of 1.0 being a good default:
        https://x.com/corbtt/status/1801026166020833457
    """
    timeout: float = 60
    """The timeout to use for the completion"""
    seed: Optional[int] = None
    """The seed to use for the completion. Useful for reproducibility."""

    max_tokens: Optional[int] = None

    use_bedrock: bool = False

    def _get_client(self) -> Instructor:
        raise NotImplementedError

    def _get_async_client(self) -> AsyncInstructor:
        raise NotImplementedError

    @staticmethod
    def _is_validation_error(retry_state: RetryCallState) -> bool:
        """Check if the exception is a validation error."""
        return isinstance(retry_state.outcome.exception(), (ValidationError, IncompleteOutputException))

    def serialize_response(self, response: T | None) -> bytes | None:
        """
        Serialize the response model to a string via pydantic model_dump_json()
        """
        if response is None:
            return None
        if not isinstance(response, self.response_model):
            raise ValueError(f"Response is not an instance of {self.response_model.__name__}. got {type(response)}")
        return response.model_dump_json().encode("utf-8")

    def deserialize_response(self, response_raw: bytes | None) -> T | None:
        """
        Deserialize the response model from a string via pydantic model_validate_json()
        """
        if response_raw is None:
            return None
        return self.response_model.model_validate_json(response_raw)

    def complete(
            self,
            user_prompt: str,
            system_prompt: str | None = None,
            callbacks: Sequence[PydanticCallbackHandler] | None = None,
            validation_context: dict | None = None,
    ) -> T:
        """
        Args:
            user_prompt: The user's input to the model
            system_prompt: The system's input to the model (optional)
            callbacks: Sequence of callback handlers (optional)
        Returns:
            The parsed chat completion class of `response_model`
        """
        if callbacks is None:
            callbacks = []

        messages: []
        if system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        else:
            messages = [{"role": "user", "content": user_prompt}]

        try:
            client = self._get_client()
            res: T = client.chat.completions.create(
                messages=messages,
                model=self.model,
                timeout=self.timeout,
                stream=False,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_model=self.response_model,
                validation_context=validation_context,
            )
            for callback in callbacks:
                callback.on_llm_start()
            for callback in callbacks:
                callback.on_llm_end(res)
            return res

        except Exception as e:
            for callback in callbacks:
                callback.on_llm_error(e)
            raise e

    async def acomplete(
            self,
            user_prompt: str,
            system_prompt: str | None = None,
            callbacks: Sequence[BaseCallbackHandler] | None = None,
            validation_context: dict | None = None,
    ) -> T:
        """Async version of complete."""
        if callbacks is None:
            callbacks = []

        messages: []
        if system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        else:
            messages = [{"role": "user", "content": user_prompt}]

        try:
            client = self._get_async_client()
            res: T = await client.chat.completions.create(
                messages=messages,
                model=self.model,
                timeout=self.timeout,
                stream=False,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_model=self.response_model,
                validation_context=validation_context,
            )
            for callback in callbacks:
                callback.on_llm_start()
            for callback in callbacks:
                callback.on_llm_end(res)
            return res

        except Exception as e:
            for callback in callbacks:
                callback.on_llm_error(e)
            raise e

    def complete_with_prompt_template(self, callbacks: Sequence[BaseCallbackHandler] | None = None, **kwargs) -> T:
        return super().complete_with_prompt_template(callbacks=callbacks, **kwargs)  # type: ignore

    async def acomplete_with_prompt_template(
            self, callbacks: Sequence[BaseCallbackHandler] | None = None, **kwargs
    ) -> T:
        return await super().acomplete_with_prompt_template(callbacks=callbacks, **kwargs)  # type: ignore
