import json
import logging
import random
import string
from typing import List, Union
from langchain_core.messages import ToolCall
from pydantic.v1 import parse

from langchain_ibm._chat.chat_schema import ChatSchema, template_env

logger = logging.getLogger(__name__)

_alphanum = string.ascii_letters + string.digits

# Prompt template shared by all llama3 and llama3.1 models present in watsonx.ai.
#
# Supports tool calls for llama3.1 models.
# See: https://llama.meta.com/docs/model-cards-and-prompt-formats/llama3_1/#json-based-tool-calling
# See: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct/blob/main/tokenizer_config.json#L2053
_TEMPLATE = template_env.from_string("""<|begin_of_text|>
{%- if not tools_in_user_message is defined %}
    {%- set tools_in_user_message = false %}
{%- endif %}
{%- if not date_string is defined %}
    {%- set date_string = "26 Jul 2024" %}
{%- endif %}
{%- if not tools is defined %}
    {%- set tools = none %}
{%- endif %}

{#- This block extracts the system message, so we can slot it into the right place. #}
{%- if messages[0]['type'] == 'system' %}
    {%- set system_message = messages[0]['content']|trim %}
    {%- set messages = messages[1:] %}
{%- else %}
    {%- set system_message = "" %}
{%- endif %}

{#- System message + builtin tools #}
{{- "<|start_header_id|>system<|end_header_id|>\n\n" }}
{%- if builtin_tools is defined or tools is not none %}
    {{- "Environment: ipython\n" }}
{%- endif %}
{%- if builtin_tools is defined %}
    {{- "Tools: " + builtin_tools | reject('equalto', 'code_interpreter') | join(", ") + "\n\n"}}
{%- endif %}
{{- "Cutting Knowledge Date: December 2023\n" }}
{{- "Today Date: " + date_string + "\n\n" }}
{%- if tools is not none and not tools_in_user_message %}
    {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}
    {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
    {{- "Do not use variables.\n\n" }}
    {%- for t in tools %}
        {{- t | tojson(indent=4) }}
        {{- "\n\n" }}
    {%- endfor %}
{%- endif %}
{{- system_message }}
{{- "<|eot_id|>" }}

{#- Custom tools are passed in a user message with some extra guidance #}
{%- if tools_in_user_message and not tools is none %}
    {#- Extract the first user message so we can plug it in here #}
    {%- if messages | length != 0 %}
        {%- set first_user_message = messages[0]['content']|trim %}
        {%- set messages = messages[1:] %}
    {%- else %}
        {{- raise_exception("Cannot put tools in the first user message when there's no first user message!") }}
    {%- endif %}
    {{- '<|start_header_id|>user<|end_header_id|>\n\n' -}}
    {{- "Given the following functions, please respond with a JSON for a function call " }}
    {{- "with its proper arguments that best answers the given prompt.\n\n" }}
    {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
    {{- "Do not use variables.\n\n" }}
    {%- for t in tools %}
        {{- t | tojson(indent=4) }}
        {{- "\n\n" }}
    {%- endfor %}
    {{- first_user_message + "<|eot_id|>"}}
    {%- if messages | length == 0 %}
        {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
    {%- endif %}
{%- endif %}

{%- for message in messages %}
    {%- if not (message.type == 'ipython' or message.type == 'tool' or message.tool_calls) %}
        {{- '<|start_header_id|>' }} 
        {%- if message['type'] == 'ai' %}
            {{- 'assistant' }}
        {%- elif message['type'] == 'human' %}
            {{- 'user' }}
        {%- else %}
            {{- message['type'] }}
        {%- endif %}
        {{- '<|end_header_id|>\n\n'+ message['content'] | trim }}
        {%- if not (message['type'] == 'ai' and loop.last) %}
            {{- '<|eot_id|>' }}
        {%- endif %}
    {%- elif message.tool_calls %}
        {%- if not message.tool_calls|length == 1 %}
            {{- raise_exception("This model only supports single tool-calls at once!") }}
        {%- endif %}
        {%- set tool_call = message.tool_calls[0] %}
        {%- if builtin_tools is defined and tool_call.name in builtin_tools %}
            {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' -}}
            {{- "<|python_tag|>" + tool_call.name + ".call(" }}
            {%- for arg_name, arg_val in tool_call.arguments | items %}
                {{- arg_name + '="' + arg_val + '"' }}
                {%- if not loop.last %}
                    {{- ", " }}
                {%- endif %}
                {%- endfor %}
            {{- ")" }}
        {%- else  %}
            {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' -}}
            {{- '<|python_tag|>{"name": "' + tool_call.name + '", ' }}
            {{- '"parameters": ' }}
            {{- tool_call.args | tojson }}
            {{- "}" }}
        {%- endif %}
        {%- if builtin_tools is defined %}
            {#- This means we're in ipython mode #}
            {{- "<|eom_id|>" }}
        {%- else %}
            {{- "<|eot_id|>" }}
        {%- endif %}
    {%- elif message.type == "tool" or message.type == "ipython" %}
        {{- "<|start_header_id|>ipython<|end_header_id|>\n\n" -}}
        {"output": "{{ message.content }}"}
        {{- "<|eot_id|>" }}
    {%- endif %}
    {%- if message['type'] != 'ai' and loop.last  %}
        {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
        {%- if force_tool_call %}
            {{- "<|python_tag|>" }}
        {%- endif %}
    {%- endif %}
{%- endfor %}""")

def parse_llama31_tool_call(text: str, force_tool_call: bool) -> Union[str, List[ToolCall]]:
    tool_calls = []

    parse_tools = force_tool_call
    if not parse_tools and text.lstrip().startswith("<|python_tag|>"):
        text = text.lstrip()[len("<|python_tag|>"):]
        parse_tools = True

    if parse_tools:
        try:
            json_calls = json.loads(text)
            if not isinstance(json_calls, list):
                json_calls = [json_calls]

            for call in json_calls:
                # Follow mistral's tool call id generation. See `mistral.py`
                id = "".join(random.choice(_alphanum) for _ in range(9))
                tool_call = ToolCall(
                    name=call["name"],
                    args=call["parameters"],
                    id=id,
                )
                tool_calls.append(tool_call)

            return tool_calls
        except:
            logger.error(
                "Failed to parse tool calls, falling back on returing text", exc_info=True)
            return text

    # Ugly hack if the model doesn't use the <|python_tag|> token
    try:
        parsed_call = json.loads(text)
        if not isinstance(parsed_call, list):
            parsed_call = [parsed_call]

        if all(map(lambda call: isinstance(call, dict) and "name" in call and "parameters" in call, parsed_call)):
            logger.warning(
                "Model did not generate tool call token, but response is a valid json tool call, parsing it anyway")
            for call in parsed_call:
                id = "".join(random.choice(_alphanum) for _ in range(9))
                tool_calls.append(ToolCall(name=call["name"], args=call["parameters"], id=id))
            return tool_calls
    except json.JSONDecodeError:
        pass

    return text


# ==== Llama 3.1 Models  with tool call support ====

LLAMA31_405B = ChatSchema(
    model_id="meta-llama/llama-3-405b-instruct",
    template=_TEMPLATE,
    tools=True,
    tools_parser=parse_llama31_tool_call,
    tools_stop_sequences=["<|eom_id|>"]
)

LLAMA31_70B = ChatSchema(
    model_id="meta-llama/llama-3-1-70b-instruct",
    template=_TEMPLATE,
    tools=True,
    tools_parser=parse_llama31_tool_call,
    tools_stop_sequences=["<|eom_id|>"]
)

LLAMA31_8B = ChatSchema(
    model_id="meta-llama/llama-3-1-8b-instruct",
    template=_TEMPLATE,
    tools=True,
    tools_parser=parse_llama31_tool_call,
    tools_stop_sequences=["<|eom_id|>"]
)

# ==== Llama 3 models ====

LLAMA3_70B = ChatSchema(
    model_id="meta-llama/llama-3-70b-instruct",
    template=_TEMPLATE,
    tools=False,
)

LLAMA3_8B = ChatSchema(
    model_id="meta-llama/llama-3-8b-instruct",
    template=_TEMPLATE,
    tools=False,
)
