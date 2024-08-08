from langchain_core.prompts import PromptTemplate

from langchain_ibm._chat.chat_schema import ChatSchema

# Prompt template for granite chat
_PROMPT = PromptTemplate.from_template("""{%- for message in messages -%}
{%- if message.type == "system" -%}
<|system|>
{{ message.content }}
{% elif message.type == "human" -%}
<|user|>
{{ message.content }}
{% elif message.type == "ai" -%}
{%-     if loop.last -%}
<|assistant|>
{{ message.content }}
{%-     else -%}
<|assistant|>
{{ message.content }}
{%      endif -%}
{% endif -%}
{%- if loop.last and message.type != "ai" -%}
<|assistant|>
{%- endif -%}
{%- endfor -%}""", template_format="jinja2")

GRANITE_13B_CHAT_V2 = ChatSchema(
    model_id="ibm/granite-13b-chat-v2",
    prompt_template=_PROMPT,
    tools=False
)