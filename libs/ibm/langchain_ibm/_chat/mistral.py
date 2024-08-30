import json
import random
import string
from typing import List, Union
from langchain_core.messages import ToolCall

from langchain_ibm._chat.chat_schema import ChatSchema, template_env

_alphanum = string.ascii_letters + string.digits

# Prompt template shared by all Mistral models present in watsonx.ai.
#
# Supports tool calls for tool-enabled Mistral models.
# See: https://github.com/mistralai/mistral-common/blob/main/examples/tokenizer.ipynb
# See: https://huggingface.co/mistralai/Mistral-Large-Instruct-2407/blob/main/tokenizer_config.json#L6176
_TEMPLATE = template_env.from_string("""{%- if messages[0]["type"] == "system" %}
    {%- set system_message = messages[0]["content"] %}
    {%- set loop_messages = messages[1:] %}
{%- else %}
    {%- set loop_messages = messages %}
{%- endif %}
{%- if not tools is defined %}
    {%- set tools = none %}
{%- endif %}
{%- set user_messages = loop_messages | selectattr("type", "equalto", "human") | list %}
{%- set last_user_message = loop_messages | last_human_message_idx %}

{#- This block checks for alternating user/assistant messages, skipping tool calling messages #}
{%- set ns = namespace() %}
{%- set ns.index = 0 %}
{%- for message in loop_messages %}
    {%- if not (message.type == "tool" or message.type == "tool_results" or (message.tool_calls is defined and message.tool_calls)) %}
        {%- if (message["type"] == "human") != (ns.index % 2 == 0) %}
            {{- raise_exception("After the optional system message, conversation roles must alternate user/assistant/user/assistant/...") }}
        {%- endif %}
        {%- set ns.index = ns.index + 1 %}
    {%- endif %}
{%- endfor %}

{{- "<s>" }}
{%- for message in loop_messages %}
    {%- if message["type"] == "human" %}
        {%- if tools is not none and (message == user_messages[-1]) %}
            {{- "[AVAILABLE_TOOLS] [" }}
            {%- for tool in tools %}
                {%- set tool = tool.function %}
                {{- '{"type": "function", "function": {' }}
                {%- for key, val in tool.items() if key != "return" %}
                    {%- if val is string %}
                        {{- '"' + key + '": "' + val + '"' }}
                    {%- else %}
                        {{- '"' + key + '": ' + val|to_json }}
                    {%- endif %}
                    {%- if not loop.last %}
                        {{- ", " }}
                    {%- endif %}
                {%- endfor %}
                {{- "}}" }}
                {%- if not loop.last %}
                    {{- ", " }}
                {%- else %}
                    {{- "]" }}
                {%- endif %}
            {%- endfor %}
            {{- "[/AVAILABLE_TOOLS]" }}
        {%- endif %}
        {%- if loop.index == last_user_message and system_message is defined %}
            {{- "[INST] " + system_message + "\n\n" + message["content"] + "[/INST]" }}
        {%- else %}
            {{- "[INST] " + message["content"] + "[/INST]" }}
        {%- endif %}
    {%- elif message.tool_calls is defined and message.tool_calls %}
        {{- "[TOOL_CALLS] [" }}
        {%- for tool_call in message.tool_calls %}
            {%- set out = tool_call|to_json %}
            {{- "{" + '"name": "' + tool_call.name + '", "parameters": ' + tool_call.args|to_json + "" }}
            {%- if not tool_call.id is defined or tool_call.id|length != 9 %}
                {{- raise_exception("Tool call IDs should be alphanumeric strings with length 9!") }}
            {%- endif %}
            {{- ', "id": "' + tool_call.id + '"}' }}
            {%- if not loop.last %}
                {{- ", " }}
            {%- else %}
                {{- "]" + "</s>" }}
            {%- endif %}
        {%- endfor %}
    {%- elif message["type"] == "ai" %}
        {{- " " + message["content"]|trim }}
        {%- if not loop.last %}
            {{- "</s>" }}
        {%- endif %}
    {%- elif message["type"] == "tool_results" or message["type"] == "tool" %}
        {%- if message.content is defined and message.content.content is defined %}
            {%- set content = message.content.content %}
        {%- else %}
            {%- set content = message.content %}
        {%- endif %}
        {{- '[TOOL_RESULTS] {"content": ' + content|string + ", " }}
        {%- if not message.tool_call_id is defined or message.tool_call_id|length != 9 %}
            {{- raise_exception("Tool call IDs should be alphanumeric strings with length 9!") }}
        {%- endif %}
        {{- '"call_id": "' + message.tool_call_id + '"}[/TOOL_RESULTS]' }}
    {%- else %}
        {{- raise_exception("Only user and assistant roles are supported, with the exception of an initial optional system message!") }}
    {%- endif %}
{%- endfor %}""")


def parse_mistral_tool_call(text: str) -> Union[str, List[ToolCall]]:
    tool_calls = []
    if text.strip().startswith("[TOOL_CALLS]"):
        text = text.strip()[len("[TOOL_CALLS]"):]

        try:
            json_calls = json.loads(text)
            if not isinstance(json_calls, list):
                json_calls = [json_calls]

            for call in json_calls:
                # A Mistral tool call id is a random string of 9 characters in [a-zA-Z0-9].
                # https://github.com/mistralai/mistral-common/blob/00780973136e3dac4d541e0712174eb34016debf/src/mistral_common/protocol/instruct/validator.py#L307
                id = "".join(random.choice(_alphanum) for _ in range(9))
                tool_call = ToolCall(
                    name=call["name"],
                    args=call["arguments"],
                    id=id,
                )
                tool_calls.append(tool_call)

            return tool_calls
        except:
            return text
    else:
        return text


MISTRAL_LARGE = ChatSchema(
    model_id="mistralai/mistral-large",
    template=_TEMPLATE,
    tools=True,
    tools_parser=parse_mistral_tool_call,
)

MIXTRAL_8X7B_V01 = ChatSchema(
    model_id="mistralai/mixtral-8x7b-instruct-v01",
    template=_TEMPLATE,
    tools=False,
)
