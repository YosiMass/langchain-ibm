from typing import override

from .base import ChatFormatter


class Llama3ChatFormatter(ChatFormatter):
    """Formats chat messages for the Llama3 instruct model."""

    @override
    def start(self):
        return "<|begin_of_text|>"

    @override
    def end(self):
        return self.ai()

    @override
    def system(self):
        return "<|start_header_id|>system<|end_header_id|>\n\n"

    @override
    def human(self):
        return "<|start_header_id|>user<|end_header_id|>\n\n"

    @override
    def ai(self):
        return "<|start_header_id|>assistant<|end_header_id|>\n\n"

    @override
    def tool(self):
        return "<|start_header_id|>ipython<|end_header_id|>\n\n"

    @override
    def end_of(self, _):
        return "<|eot_id|>"
