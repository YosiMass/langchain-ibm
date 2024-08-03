from abc import ABC, abstractmethod
from typing import Iterable, List, Union, final

from langchain_core.messages.utils import convert_to_messages, _msg_to_chunk, _chunk_to_msg
from langchain_core.prompt_values import PromptValue

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    MessageLikeRepresentation,
    SystemMessage,
    ToolMessage
)


class ChatFormatter(ABC):
    """
    Formats a list of messages into a string for use as plain prompt in a `BaseLLM` model.
    """

    @abstractmethod
    def start(self) -> str:
        ...

    @abstractmethod
    def end(self) -> str:
        ...

    @abstractmethod
    def system(self) -> str:
        ...

    @abstractmethod
    def human(self) -> str:
        ...

    @abstractmethod
    def ai(self) -> str:
        ...

    @abstractmethod
    def tool(self) -> str:
        ...

    @abstractmethod
    def end_of(self, message: BaseMessage) -> str:
        ...

    @final
    def tag_of(self, message: BaseMessage) -> str:
        if isinstance(message, AIMessage):
            return self.ai()
        elif isinstance(message, SystemMessage):
            return self.system()
        elif isinstance(message, HumanMessage):
            return self.human()
        elif isinstance(message, FunctionMessage) or isinstance(message, ToolMessage):
            return self.tool()
        else:
            raise ValueError(f"Unknown message type: {type(message)}")

    @final
    def format(self, messages: list[BaseMessage]) -> str:
        if not messages:
            return ""

        messages = _merge_message_runs(messages)

        builder = [self.start()]
        for idx, message in enumerate(messages):
            assert isinstance(
                message.content, str), "Only messages with string content are supported by `ChatFormatter`."

            builder.append(f"{self.tag_of(message)}{message.content}")
            if idx == len(messages) - 1:
                if not isinstance(message, AIMessage):
                    builder.append(self.end_of(message))
                    builder.append(self.end())
            else:
                builder.append(self.end_of(message))

        return "".join(builder)


def _merge_message_runs(
    messages: Union[Iterable[MessageLikeRepresentation], PromptValue],
) -> List[BaseMessage]:
    if not messages:
        return []
    messages = convert_to_messages(messages)
    merged: List[BaseMessage] = []
    for msg in messages:
        curr = msg.copy(deep=True)
        last = merged.pop() if merged else None
        if not last:
            merged.append(curr)
        elif isinstance(curr, ToolMessage) or not isinstance(curr, last.__class__):
            merged.extend([last, curr])
        else:
            last_chunk = _msg_to_chunk(last)
            curr_chunk = _msg_to_chunk(curr)
            merged.append(_chunk_to_msg(last_chunk + curr_chunk))
    return merged
