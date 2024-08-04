"""Test for chat formatters"""

from langchain_core.messages import AIMessageChunk
from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain_ibm.chat_formatters.llama3_chat_formatter import Llama3ChatFormatter


CONVERSATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a respectful AI assistant."),
    MessagesPlaceholder("chat_history"),
    ("user", "{input}"),
])

HISTORY = [
    ("user", "Hello, how are you?"),
    ("assistant", "Hello! I'm doing well, thank you for asking. I'm a large language model assistant."),
    ("user", "What's the capital of Italy?"),
    ("assistant", "Rome"),
]

INSTRUCT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert at extracting entities from text. Always extract entities using the following json output format:
{{
    "type": "entity_type",
    "value": "entity_value"
}}"""),
    ("user",
     "Here is the input:\n\"{input}\". Remember to always respond in the json format specified above."),
    ("assistant", "Here is the json output:\n")
])

UNMERGED_MESSAGES = ChatPromptTemplate.from_messages([
    ("system", "You are a respectful AI assistant."),
    ("user", "What is the capital of Italy?"),
    AIMessageChunk(content="The"),
    AIMessageChunk(content=" capital"),
    AIMessageChunk(content=" of Italy"),
    AIMessageChunk(content=" is"),
    AIMessageChunk(content=" Rome"),
])


def test_llama3_conversation_no_history() -> None:
    expected = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a respectful AI assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

Hello, how are you?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

    formatter = Llama3ChatFormatter()
    assert (formatter.format(CONVERSATION_PROMPT.format_messages(
        chat_history=[], input="Hello, how are you?")) == expected)


def test_llama3_conversation_history() -> None:
    expected = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a respectful AI assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

Hello, how are you?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Hello! I'm doing well, thank you for asking. I'm a large language model assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

What's the capital of Italy?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Rome<|eot_id|><|start_header_id|>user<|end_header_id|>

Thanks!<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

    formatter = Llama3ChatFormatter()
    assert (formatter.format(CONVERSATION_PROMPT.format_messages(
        chat_history=HISTORY, input="Thanks!")) == expected)


def test_llama3_instruct_prompt() -> None:
    expected = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert at extracting entities from text. Always extract entities using the following json output format:
{
    "type": "entity_type",
    "value": "entity_value"
}<|eot_id|><|start_header_id|>user<|end_header_id|>

Here is the input:
"The capital of Italy is Rome.". Remember to always respond in the json format specified above.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Here is the json output:
"""

    formatter = Llama3ChatFormatter()
    assert (formatter.format(INSTRUCT_PROMPT.format_messages(
        input="The capital of Italy is Rome.")) == expected)


def test_llama3_merge_messages() -> None:
    expected = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a respectful AI assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

What is the capital of Italy?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

The capital of Italy is Rome"""

    formatter = Llama3ChatFormatter()
    assert (formatter.format(UNMERGED_MESSAGES.format_messages()) == expected)


# TODO: test for Mixral and Granite
