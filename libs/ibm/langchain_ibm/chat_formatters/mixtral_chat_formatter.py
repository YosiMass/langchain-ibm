from typing import override

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage
)
from langchain_core.messages import BaseMessage

from .base import ChatFormatter


class MixtralChatFormatter(ChatFormatter):
    """Formats chat messages for the Mixtral8x7B instruct model."""

    @override
    def start(self):
        return "<s>"

    @override
    def end(self):
        return ""

    @override
    def system(self):
        return "[INST] "

    @override
    def human(self):
        return "[INST] "

    @override
    def ai(self):
        return ""

    @override
    def tool(self):
        raise NotImplementedError(
            "Tool messages are not supported in Mixtral.")

    @override
    def end_of(self, message: BaseMessage):
        if isinstance(message, AIMessage):
            return "</s> "
        elif isinstance(message, SystemMessage):
            return " [/INST]</s> "
        elif isinstance(message, HumanMessage):
            return " [/INST]"
        elif isinstance(message, FunctionMessage) or isinstance(message, ToolMessage):
            raise ValueError("Tool messages are not supported in Mixtral.")
        else:
            raise ValueError(f"Unknown message type: {type(message)}")
