from __future__ import annotations

import copy
import json
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, TypedDict, Union

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.pydantic_v1 import PrivateAttr

from langchain.callbacks.manager import CallbackManagerForLLMRun

if TYPE_CHECKING:
    import openllm


ServerType = Literal["http", "grpc"]


class IdentifyingParams(TypedDict):
    """Parameters for identifying a model as a typed dict."""

    model_name: str
    model_id: Optional[str]
    server_url: Optional[str]
    server_type: Optional[ServerType]
    embedded: bool
    llm_kwargs: Dict[str, Any]


logger = logging.getLogger(__name__)


class ChatOpenLLM(BaseChatModel):
    model_name: Optional[str] = None
    model_id: Optional[str] = None
    server_url: Optional[str] = None
    server_type: ServerType = "http"
    embedded: bool = True
    llm_kwargs: Dict[str, Any]

    _runner: Optional[openllm.LLMRunner] = PrivateAttr(default=None)
    _client: Union[
        openllm.client.HTTPClient, openllm.client.GrpcClient, None
    ] = PrivateAttr(default=None)

    class Config:
        extra = "forbid"

    def __init__(
        self,
        model_name: Optional[str] = None,
        *,
        model_id: Optional[str] = None,
        server_url: Optional[str] = None,
        server_type: Literal["grpc", "http"] = "http",
        embedded: bool = True,
        timeout: int = 30,
        **llm_kwargs: Any,
    ):
        try:
            import openllm
        except ImportError as e:
            raise ImportError(
                "Could not import openllm. Make sure to install it with "
                "'pip install openllm.'"
            ) from e

        llm_kwargs = llm_kwargs or {}

        if server_url is not None:
            logger.debug("'server_url' is provided, returning a openllm.Client")
            assert (
                model_id is None and model_name is None
            ), "'server_url' and {'model_id', 'model_name'} are mutually exclusive"
            client_cls = (
                openllm.client.HTTPClient
                if server_type == "http"
                else openllm.client.GrpcClient
            )
            client = client_cls(server_url, timeout=timeout)

            super().__init__(
                **{
                    "server_url": server_url,
                    "server_type": server_type,
                    "llm_kwargs": llm_kwargs,
                }
            )
            self._runner = None  # type: ignore
            self._client = client
        else:
            assert model_name is not None, "Must provide 'model_name' or 'server_url'"
            # since the LLM are relatively huge, we don't actually want to convert the
            # Runner with embedded when running the server. Instead, we will only set
            # the init_local here so that LangChain users can still use the LLM
            # in-process. Wrt to BentoML users, setting embedded=False is the expected
            # behaviour to invoke the runners remotely.
            # We need to also enable ensure_available to download and setup the model.
            runner = openllm.Runner(
                model_name=model_name,
                model_id=model_id,
                init_local=embedded,
                ensure_available=True,
                **llm_kwargs,
            )
            super().__init__(
                **{
                    "model_name": model_name,
                    "model_id": model_id,
                    "embedded": embedded,
                    "llm_kwargs": llm_kwargs,
                }
            )
            self._client = None  # type: ignore
            self._runner = runner

    @property
    def runner(self) -> openllm.LLMRunner:
        """
        Get the underlying openllm.LLMRunner instance for integration with BentoML.

        Example:
        .. code-block:: python

            llm = OpenLLM(
                model_name='flan-t5',
                model_id='google/flan-t5-large',
                embedded=False,
            )
            tools = load_tools(["serpapi", "llm-math"], llm=llm)
            agent = initialize_agent(
                tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION
            )
            svc = bentoml.Service("langchain-openllm", runners=[llm.runner])

            @svc.api(input=Text(), output=Text())
            def chat(input_text: str):
                return agent.run(input_text)
        """
        if self._runner is None:
            raise ValueError("OpenLLM must be initialized locally with 'model_name'")
        return self._runner

    @property
    def _identifying_params(self) -> IdentifyingParams:
        """Get the identifying parameters."""
        if self._client is not None:
            for k, v in self._client._config.items():
                if k not in self.llm_kwargs:
                    self.llm_kwargs[k] = v
            model_name = self._client._metadata.model_name
            model_id = self._client._metadata.model_id
        else:
            if self._runner is None:
                raise ValueError("Runner must be initialized.")
            model_name = self.model_name
            model_id = self.model_id
            try:
                self.llm_kwargs.update(
                    json.loads(self._runner.identifying_params["configuration"])
                )
            except (TypeError, json.JSONDecodeError):
                pass
        return IdentifyingParams(
            server_url=self.server_url,
            server_type=self.server_type,
            embedded=self.embedded,
            llm_kwargs=self.llm_kwargs,
            model_name=model_name,
            model_id=model_id,
        )

    @property
    def _llm_type(self) -> str:
        return "chatopenllm_client" if self._client else "chatopenllm"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: List[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        try:
            import openllm
        except ImportError as e:
            raise ImportError(
                "Could not import openllm. Make sure to install it with "
                "'pip install openllm'."
            ) from e

        copied = copy.deepcopy(self.llm_kwargs)
        copied.update(kwargs)
        config = openllm.AutoConfig.for_model(
            self._identifying_params["model_name"], **copied
        )

        if self._client:
            outputs = self._client.chat(
                message=[
                    {
                        "role": {"human": "User", "ai": "Chatbot"}[message.type],
                        "message": message.content,
                    }
                    for message in messages
                ],
                stop=stop,
                **config.model_dump(flatten=True),
            ).outputs
            generations = [ChatGeneration(message=output.text) for output in outputs]
        else:
            raise NotImplementedError

            assert self._runner is not None
            res = self._runner(messages, **config.model_dump(flatten=True))

            if isinstance(res, dict) and "text" in res:
                text = res["text"]
            elif isinstance(res, str):
                text = res
            else:
                raise ValueError(
                    "Expected result to be a dict with key 'text' or a string. "
                    f"Received {res}"
                )

            generations = [ChatGeneration(message=text) for output in outputs]

        return ChatResult(generations=generations)
