import json
import logging
import random
import string
from typing import List, Union
from langchain_core.messages import ToolCall

from langchain_ibm._chat.chat_schema import ChatSchema, template_env

logger = logging.getLogger(__name__)

_alphanum = string.ascii_letters + string.digits

# Prompt template for granite chat
_TEMPLATE = template_env.from_string(
 "{%- if tools %}\n    {{- '<|start_of_role|>available_tools<|end_of_role|>\n' }}\n    {%- for tool in tools %}\n    {{- tool | tojson(indent=4) }}\n    {%- if not loop.last %}\n        {{- '\n\n' }}\n    {%- endif %}\n    {%- endfor %}\n    {{- '<|end_of_text|>\n' }}\n{%- endif %}\n{%- for message in messages %}\n    {%- if message.type == 'system' %}\n    {{- '<|start_of_role|>system<|end_of_role|>' + message.content + '<|end_of_text|>\n' }}\n    {%- elif message.type == 'human' %}\n    {{- '<|start_of_role|>user<|end_of_role|>' + message.content + '<|end_of_text|>\n' }}\n    {%- elif message.type == 'ai' %}\n    {{- '<|start_of_role|>assistant<|end_of_role|>'  + message.content + '<|end_of_text|>\n' }}\n    {%- elif message.type == 'assistant_tool_call' %}\n    {{- '<|start_of_role|>assistant<|end_of_role|><|tool_call|>' + message.content + '<|end_of_text|>\n' }}\n    {%- elif message.type == 'tool' %}\n    {{- '<|start_of_role|>tool_response<|end_of_role|>' + message.content + '<|end_of_text|>\n' }}\n    {%- endif %}\n    {%- if loop.last and add_generation_prompt %}\n    {{- '<|start_of_role|>assistant<|end_of_role|>' }}\n    {%- endif %}\n{%- endfor %}"
)

def parse_granite_3_tool_call(text: str, force_tool_call: bool) -> Union[str, List[ToolCall]]:
    tool_calls = []

    parse_tools = force_tool_call

    # granite adds sometimes 'assistant'
    if text.lstrip().startswith("assistant"):
        text = text.lstrip()[len("assistant"):]

    if parse_tools:
        try:
            json_calls = json.loads(text)
            if not isinstance(json_calls, list):
                json_calls = [json_calls]

            for call in json_calls:
                # Follow mistral's tool call id generation. See `mistral.py`
                id = "".join(random.choice(_alphanum) for _ in range(9))
                tool_call = ToolCall(
                    name=call["function"],
                    args=call["parameters"],
                    id=id,
                )
                tool_calls.append(tool_call)

            return tool_calls
        except:
            logger.error(
                "Failed to parse tool calls, falling back on returing text", exc_info=True)
            return text

GRANITE_3_8B_INSTRUCT = ChatSchema(
    model_id="ibm/granite-3-8b-instruct",
    template=_TEMPLATE,
    tools=True,
    tools_parser=parse_granite_3_tool_call,
    tools_stop_sequences=["<|eom_id|>"]
)

