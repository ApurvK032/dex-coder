"""A stand-in for the Anthropic client that replays scripted tool calls (no network, no cost)."""

from dataclasses import dataclass, field


@dataclass
class Block:
    type: str
    text: str | None = None
    id: str | None = None
    name: str | None = None
    input: dict | None = None

    def model_dump(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class Usage:
    input_tokens: int = 1000
    output_tokens: int = 500
    cache_creation_input_tokens: int = 2000
    cache_read_input_tokens: int = 8000


@dataclass
class Message:
    content: list
    stop_reason: str = "tool_use"
    usage: Usage = field(default_factory=Usage)
    _request_id: str = "req_fake"


class _Stream:
    def __init__(self, message: Message) -> None:
        self.message = message

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self) -> Message:
        return self.message


class FakeClient:
    """``script`` is a list of (tool_name, input) pairs, one tool call per turn."""

    def __init__(self, script: list[tuple[str, dict]]) -> None:
        self.script = list(script)
        self.requests: list[dict] = []
        self.messages = self

    def stream(self, **params):
        self.requests.append(params)
        if not self.script:
            return _Stream(Message([Block("text", text="Done.")], stop_reason="end_turn"))
        name, args = self.script.pop(0)
        n = len(self.requests)
        return _Stream(Message([Block("text", text=f"Calling {name}."), Block("tool_use", id=f"toolu_{n}", name=name, input=args)]))
