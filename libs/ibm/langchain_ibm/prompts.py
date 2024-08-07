from langchain_core.prompts import PromptTemplate

MIXTRAL_PROMPT = PromptTemplate.from_template("""{%- for message in messages -%}
    {%- if message.type == "system" -%}
        <s>[INST] {{ message.content }} [/INST]</s>
    {%- elif message.type == "human" -%}
        {%- if (messages|length - loop.index == 0 or messages|length - loop.index == 1) and tools -%}
            [AVAILABLE_TOOLS] {{ tools }} [/AVAILABLE_TOOLS]
        {%- endif -%}
        [INST] {{ message.content }} [/INST]
    {%- elif message.type == "ai" -%}
        {%- if message.content -%}
            {{ message.content }}
            {%- if messages|length - index != 1 -%}
                </s>
            {%- endif -%}
        {%- elif message.tool_calls -%}
            [TOOL_CALLS] [
            {%- for call in message.tool_calls -%}
                {"name": "{{ call.function.name }}", "arguments": {{ call.function.arguments }}}
            {%- endfor -%}
            ]</s>
        {%- endif -%}
    {%- elif message.type == "tool" -%}
        [TOOL_RESULTS] {"content": {{ message.content }}} [/TOOL_RESULTS]
    {%- endif -%}
{%- endfor -%}""", template_format="jinja2")
