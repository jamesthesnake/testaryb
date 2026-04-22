import asyncio
import base64
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Type

import openai
import tiktoken
from httpcore import TimeoutException
from instructor import AsyncInstructor, Instructor, from_openai
from openai import APIError, APIStatusError, RateLimitError
from openai._streaming import AsyncStream, Stream
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
)
from openai.types.chat.completion_create_params import ResponseFormat
from tenacity import (
    RetryCallState,
)

from backend.mana.callbacks.base import BaseCallbackHandler
from backend.mana.config.openai import GPT4_TURBO
from backend.mana.llms.base import ChatModel
from backend.mana.llms.instructor_model import BaseInstructor, T
from backend.mana.config.env import OPENAI_API_KEY


def get_tokenizer(model: Optional[str] = None) -> tiktoken.Encoding:
    """Returns the tokenizer for the specified model."""
    if model is None:
        model = GPT4_TURBO
    return tiktoken.encoding_for_model(model)


@dataclass(kw_only=True, frozen=True)
class OpenAI(ChatModel):
    model: str
    """The model to use for the completion"""
    temperature: float = 1.0
    """
    The temperature to use for the completion
    Some evidence of 1.0 being a good default:
        https://x.com/corbtt/status/1801026166020833457
    """
    stream: bool = True
    """Whether to stream the response or not"""
    timeout: float = 10
    """The timeout to use for the completion"""
    seed: Optional[int] = None
    """The seed to use for the completion. Useful for reproducibility."""
    response_format: ResponseFormat = None
    """The response format to use for the completion. Useful for JSON mode. Default to standard text outputs."""

    @staticmethod
    def _is_rate_limit_error(retry_state: RetryCallState) -> bool:
        """Check if the error is due to rate limit."""
        exc = retry_state.outcome.exception()
        if isinstance(exc, RateLimitError):
            message = exc.response or ""
            return "exceeded your current quota" not in message
        return False

    def _is_retryable(self, retry_state: RetryCallState) -> bool:
        """Determine if an exception should trigger a retry."""
        exc = retry_state.outcome.exception()
        # tenacity always calls the @retry callback even when there's no exception / error
        if exc is None:
            return False

        if isinstance(exc, APIStatusError):
            status_code = exc.status_code
            if status_code >= 500:
                return True
            return self._is_rate_limit_error(retry_state)
        if isinstance(exc, (APIError, TimeoutException, TimeoutError)):
            return True
        return False

    def complete(
            self,
            user_prompt: str | List[dict],
            system_prompt: str | None = None,
            callbacks: Sequence[BaseCallbackHandler] | None = None,
    ) -> str:
        """
        Args:
            user_prompt: The user's input to the model
            system_prompt: The system's input to the model (optional)
            callbacks: Sequence of callback handlers (optional)
        Returns:
            The chat completion as a string
        """
        if callbacks is None:
            callbacks = []

        messages: list[ChatCompletionMessageParam]
        if system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        else:
            messages = [{"role": "user", "content": user_prompt}]

        def api_call(_messages: list[ChatCompletionMessageParam]) -> ChatCompletion | Stream[ChatCompletionChunk]:
            return client.chat.completions.create(
                messages=_messages,
                model=self.model,
                timeout=self.timeout,
                stream=self.stream,
                temperature=self.temperature,
                seed=self.seed,
                response_format=self.response_format,
            )

        try:
            client = openai.OpenAI(max_retries=0, api_key=OPENAI_API_KEY)
            completion = api_call(messages)
            for callback in callbacks:
                callback.on_llm_start()
            if not self.stream:
                res: str = completion.choices[0].message.content  # type: ignore
            else:
                res = ""
                for chunk in completion:
                    delta = chunk.choices[0].delta.content or ""  # type: ignore
                    for callback in callbacks:
                        callback.on_llm_new_token(delta)
                    res += delta
            for callback in callbacks:
                callback.on_llm_end(res)
            return res

        except Exception as e:
            for callback in callbacks:
                callback.on_llm_error(e)
            raise e

    async def acomplete(
            self,
            user_prompt: str | List[dict],
            system_prompt: str | None = None,
            callbacks: Sequence[BaseCallbackHandler] | None = None,
    ) -> str:
        """Async version of complete."""
        if callbacks is None:
            callbacks = []

        messages: list[ChatCompletionMessageParam]
        if system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        else:
            messages = [{"role": "user", "content": user_prompt}]

        async def api_call_async(
                _messages: list[ChatCompletionMessageParam],
        ) -> ChatCompletion | AsyncStream[ChatCompletionChunk]:
            return await async_client.chat.completions.create(
                messages=_messages,
                model=self.model,
                timeout=self.timeout,
                stream=self.stream,
                temperature=self.temperature,
                seed=self.seed,
                response_format=self.response_format,
            )

        try:
            async with openai.AsyncOpenAI(max_retries=0) as async_client:
                completion = await api_call_async(messages)
                for callback in callbacks:
                    callback.on_llm_start()
                if not self.stream:
                    res: str = completion.choices[0].message.content  # type: ignore
                else:
                    res = ""
                    async for chunk in completion:  # type: ignore
                        delta = chunk.choices[0].delta.content or ""
                        for callback in callbacks:
                            callback.on_llm_new_token(delta)
                        res += delta
                        await asyncio.sleep(0)

            for callback in callbacks:
                callback.on_llm_end(res)
            return res
        except Exception as e:
            for callback in callbacks:
                callback.on_llm_error(e)
            raise e

    @staticmethod
    def prompt_with_images(
            message: str,
            paths: [Path | str] = None,
            urls: [str] = None,
            detail: str = "auto",
            type: str = "base64",
            media_type="image/png",
    ) -> List[dict]:
        prompt = []
        if message:
            prompt.append({"type": "text", "text": message})

        paths = paths or []
        if not isinstance(paths, list):
            paths = [paths]

        def _image_to_data_uri(file_path):
            with open(file_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
                return f"data:{media_type};{type},{encoded_image}"

        for path in paths:
            path = str(path)
            if not Path(path).exists():
                raise FileNotFoundError(f"Image file not found: {path}")

            prompt.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": _image_to_data_uri(path),
                        "detail": detail,
                    },
                }
            )

        urls = urls or []
        if not isinstance(urls, list):
            urls = [urls]

        for url in urls:
            prompt.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": url,
                        "detail": detail,
                    },
                }
            )

        return prompt


@dataclass(kw_only=True, frozen=True)
class InstructorOpenAI(BaseInstructor):
    def __init__(self, model: str, response_model: Type[T], **kwargs):
        super().__init__(model=model, response_model=response_model, **kwargs)

    def _get_client(self) -> Instructor:
        client = openai.OpenAI(max_retries=0)
        return from_openai(client)

    def _get_async_client(self) -> AsyncInstructor:
        client = openai.AsyncOpenAI(max_retries=0)
        return from_openai(client)

    @staticmethod
    def _is_retryable(retry_state: RetryCallState) -> bool:
        """Determine if an exception should trigger a retry.

        This extends the base OpenAI retry logic by making sure we retry on validation errors.
        """
        # explicitly retry on validation error
        if BaseInstructor._is_validation_error(retry_state):
            return True
        return False
