"""API request and response type definitions.

This module defines types for the REST API endpoints, including chat
completion, instance management, and model listing. Types follow OpenAI
compatible formats where applicable.
"""

import time
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator
from pydantic_core import PydanticUseDefault

from exo.shared.types.common import CommandId
from exo.shared.types.models import ModelId
from exo.shared.types.worker.instances import Instance, InstanceId, InstanceMeta
from exo.shared.types.worker.shards import Sharding

FinishReason = Literal[
    "stop", "length", "tool_calls", "content_filter", "function_call"
]
"""Reason why text generation finished.

Values:
    stop: Stop sequence was encountered.
    length: Maximum token limit was reached.
    tool_calls: Tool/function calls were generated.
    content_filter: Content was filtered.
    function_call: Function call was generated.
"""


class ModelListModel(BaseModel):
    """Model information in the model list response.

    Attributes:
        id: Model identifier.
        object: Always "model" for model objects.
        created: Unix timestamp when model metadata was created.
        owned_by: Owner/organization (always "exo" for EXO models).
        hugging_face_id: Hugging Face model ID if applicable.
        name: Human-readable model name.
        description: Model description.
        context_length: Maximum context length in tokens.
        tags: List of tags associated with the model.
        storage_size_megabytes: Size of model storage in megabytes.
        supports_tensor: Whether model supports tensor parallelism.
    """

    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "exo"
    hugging_face_id: str = Field(default="")
    name: str = Field(default="")
    description: str = Field(default="")
    context_length: int = Field(default=0)
    tags: list[str] = Field(default=[])
    storage_size_megabytes: int = Field(default=0)
    supports_tensor: bool = Field(default=False)


class ModelList(BaseModel):
    """Response containing a list of available models.

    Attributes:
        object: Always "list" for list responses.
        data: List of model information.
    """

    object: Literal["list"] = "list"
    data: list[ModelListModel]


class ChatCompletionMessageText(BaseModel):
    """Text content in a chat message.

    Attributes:
        type: Always "text" for text content.
        text: The text content.
    """

    type: Literal["text"] = "text"
    text: str


class ChatCompletionMessage(BaseModel):
    """A single message in a chat completion conversation.

    Attributes:
        role: Role of the message sender (system, user, assistant, etc.).
        content: Message content. Can be string, structured text, or list
            of text items. None for tool/function messages.
        thinking: Thinking/reasoning content (for GPT-OSS harmony format).
        name: Name of the message sender (for tool/function roles).
        tool_calls: List of tool calls made in this message.
        tool_call_id: ID of the tool call this message responds to.
        function_call: Function call information (legacy format).
    """

    role: Literal["system", "user", "assistant", "developer", "tool", "function"]
    content: (
        str | ChatCompletionMessageText | list[ChatCompletionMessageText] | None
    ) = None
    thinking: str | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    function_call: dict[str, Any] | None = None


class TopLogprobItem(BaseModel):
    """Top log probability candidate for a token position.

    Attributes:
        token: The token text.
        logprob: Log probability of this token.
        bytes: Byte representation of the token.
    """

    token: str
    logprob: float
    bytes: list[int] | None = None


class LogprobsContentItem(BaseModel):
    """Log probability information for a token in the response.

    Attributes:
        token: The token text.
        logprob: Log probability of this token.
        bytes: Byte representation of the token.
        top_logprobs: Top alternative tokens and their log probabilities.
    """

    token: str
    logprob: float
    bytes: list[int] | None = None
    top_logprobs: list[TopLogprobItem]


class Logprobs(BaseModel):
    """Log probability information for the completion.

    Attributes:
        content: List of log probability items for each token in the response.
    """

    content: list[LogprobsContentItem] | None = None


class PromptTokensDetails(BaseModel):
    """Detailed token count breakdown for prompt tokens.

    Attributes:
        cached_tokens: Number of tokens served from cache.
        audio_tokens: Number of audio tokens in the prompt.
    """

    cached_tokens: int = 0
    audio_tokens: int = 0


class CompletionTokensDetails(BaseModel):
    """Detailed token count breakdown for completion tokens.

    Attributes:
        reasoning_tokens: Number of reasoning/thinking tokens.
        audio_tokens: Number of audio tokens in the completion.
        accepted_prediction_tokens: Number of accepted prediction tokens.
        rejected_prediction_tokens: Number of rejected prediction tokens.
    """

    reasoning_tokens: int = 0
    audio_tokens: int = 0
    accepted_prediction_tokens: int = 0
    rejected_prediction_tokens: int = 0


class Usage(BaseModel):
    """Token usage statistics for a completion.

    Attributes:
        prompt_tokens: Number of tokens in the prompt.
        completion_tokens: Number of tokens in the completion.
        total_tokens: Total tokens (prompt + completion).
        prompt_tokens_details: Detailed breakdown of prompt tokens.
        completion_tokens_details: Detailed breakdown of completion tokens.
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: PromptTokensDetails | None = None
    completion_tokens_details: CompletionTokensDetails | None = None


class StreamingChoiceResponse(BaseModel):
    """A choice in a streaming chat completion response.

    Attributes:
        index: Index of this choice (for multi-choice responses).
        delta: Incremental message update (contains only new content).
        logprobs: Log probability information for this delta.
        finish_reason: Reason why generation finished (if this is the final chunk).
        usage: Token usage (typically only in final chunk).
    """

    index: int
    delta: ChatCompletionMessage
    logprobs: Logprobs | None = None
    finish_reason: FinishReason | None = None
    usage: Usage | None = None


class ChatCompletionChoice(BaseModel):
    """A choice in a non-streaming chat completion response.

    Attributes:
        index: Index of this choice (for multi-choice responses).
        message: Complete generated message.
        logprobs: Log probability information for the message.
        finish_reason: Reason why generation finished.
    """

    index: int
    message: ChatCompletionMessage
    logprobs: Logprobs | None = None
    finish_reason: FinishReason | None = None


class ChatCompletionResponse(BaseModel):
    """Response from a chat completion request.

    Attributes:
        id: Unique identifier for this completion.
        object: Always "chat.completion" for completion responses.
        created: Unix timestamp when the completion was created.
        model: Model identifier used for generation.
        choices: List of completion choices (supports both streaming and non-streaming).
        usage: Token usage statistics.
        service_tier: Service tier identifier (optional).
    """

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice | StreamingChoiceResponse]
    usage: Usage | None = None
    service_tier: str | None = None


class ChatCompletionTaskParams(BaseModel):
    """Parameters for a chat completion request.

    Attributes:
        model: Model identifier to use for generation.
        frequency_penalty: Penalty for token frequency (reduces repetition).
            Range typically -2.0 to 2.0.
        messages: List of messages in the conversation.
        logit_bias: Bias for specific tokens (token_id -> bias value).
        logprobs: Whether to include log probabilities in the response.
        top_logprobs: Number of top log probabilities to return per token.
        max_tokens: Maximum number of tokens to generate.
        n: Number of completion choices to generate.
        presence_penalty: Penalty for token presence (encourages new topics).
            Range typically -2.0 to 2.0.
        response_format: Format specification (e.g., JSON mode).
        seed: Random seed for deterministic generation.
        stop: Stop sequences that halt generation when encountered.
        stream: Whether to stream responses incrementally.
        temperature: Sampling temperature (0.0 = deterministic, higher = more random).
        top_p: Nucleus sampling parameter (cumulative probability threshold).
        tools: List of tools/functions available to the model.
        tool_choice: Which tool to use ("auto", "none", or specific tool).
        parallel_tool_calls: Whether to allow parallel tool calls.
        user: User identifier for tracking/abuse prevention.
    """

    model: str
    frequency_penalty: float | None = None
    messages: list[ChatCompletionMessage]
    logit_bias: dict[str, int] | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    max_tokens: int | None = None
    n: int | None = None
    presence_penalty: float | None = None
    response_format: dict[str, Any] | None = None
    seed: int | None = None
    stop: str | list[str] | None = None
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    user: str | None = None


class PlaceInstanceParams(BaseModel):
    """Parameters for placing a model instance.

    Attributes:
        model_id: Identifier for the model to place.
        sharding: Sharding strategy (defaults to Pipeline).
        instance_meta: Instance metadata/type (defaults to MlxRing).
        min_nodes: Minimum number of nodes required for placement.
    """

    model_id: str
    sharding: Sharding = Sharding.Pipeline
    instance_meta: InstanceMeta = InstanceMeta.MlxRing
    min_nodes: int = 1

    @field_validator("sharding", "instance_meta", mode="plain")
    @classmethod
    def use_default(cls, v: object):
        """Use default value if None or invalid type is provided."""
        if not v or not isinstance(v, (Sharding, InstanceMeta)):
            raise PydanticUseDefault()
        return v


class CreateInstanceParams(BaseModel):
    """Parameters for creating a model instance with explicit configuration.

    Attributes:
        instance: The instance configuration to create.
    """

    instance: Instance


class PlacementPreview(BaseModel):
    """Preview of how an instance would be placed.

    Attributes:
        model_id: Identifier for the model.
        sharding: Sharding strategy used.
        instance_meta: Instance metadata/type.
        instance: The instance configuration that would be created (if placement succeeds).
        memory_delta_by_node: Mapping of node IDs to additional memory bytes
            that would be used on each node (if placement succeeds).
        error: Error message if placement failed.
    """

    model_id: ModelId
    sharding: Sharding
    instance_meta: InstanceMeta
    instance: Instance | None = None
    memory_delta_by_node: dict[str, int] | None = None
    error: str | None = None


class PlacementPreviewResponse(BaseModel):
    """Response containing placement previews.

    Attributes:
        previews: List of placement previews (one per requested configuration).
    """

    previews: list[PlacementPreview]


class DeleteInstanceTaskParams(BaseModel):
    """Parameters for deleting an instance.

    Attributes:
        instance_id: Identifier for the instance to delete.
    """

    instance_id: str


class CreateInstanceResponse(BaseModel):
    """Response from creating an instance.

    Attributes:
        message: Status message.
        command_id: Command ID for tracking the instance creation.
    """

    message: str
    command_id: CommandId


class DeleteInstanceResponse(BaseModel):
    """Response from deleting an instance.

    Attributes:
        message: Status message.
        command_id: Command ID for tracking the instance deletion.
        instance_id: Identifier of the deleted instance.
    """

    message: str
    command_id: CommandId
    instance_id: InstanceId
