import json
from collections.abc import Callable
from typing import List, Optional, Union
from jinja2 import Template
from jinja2.sandbox import SandboxedEnvironment
from langchain_core.messages import ToolCall
from langchain_core.pydantic_v1 import BaseModel

def raise_exception(message):
    raise Exception(message)

template_env = SandboxedEnvironment()
template_env.globals['raise_exception'] = raise_exception
template_env.filters['to_json'] = lambda value: json.dumps(value)
template_env.filters['has_system_message'] = lambda messages: any(
    message.type == "system" for message in messages)
template_env.filters['last_human_message_idx'] = lambda messages: next(
    (len(messages) - i for i, message in enumerate(reversed(messages)) if message.type == "human"), None)


class ChatSchema(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    model_id: str
    """The Watsonx.ai model ID"""

    template: Template
    """
    Jinja2 prompt template used for formatting messages into a correctly
    formatted prompt for the model. At render time, a list of `BaseMessage`
    objects is passed to the template, along with available tools (if supported by the model).
    """

    tools: bool
    """Whether or not this model supports tools."""

    tools_parser: Optional[Callable[[str, bool], Union[str, List[ToolCall]]]] = None
    """
    Function to parse tool results from the model output. Must be provided if `tools` is True.
    This function will be invoked if the model is called with tools enabled (passing "tools" or
    binding them using `bind_tools`) and should return either a List of tool calls if the output
    of the model indicates that tools should be called (as per model's spec), or a string if the
    output is not a tool call.
    The first argument is the model output, and the second argument is a boolean indicating whether
    tools are forced (i.e. the model was called with `tool_choice="required" | True`).
    """

    tools_stop_sequences: Optional[List[str]] = None
    """Optional additional stop sequences to add to inference parameters if tool use is enabled."""
