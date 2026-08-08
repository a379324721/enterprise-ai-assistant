import json

from enterprise_ai_assistant.api.routes import _encode_sse


def test_sse_event_is_typed_utf8_json() -> None:
    encoded = _encode_sse("token", {"content": "差\n旅"})

    assert encoded.startswith("event: token\ndata: ")
    assert encoded.endswith("\n\n")
    payload = encoded.split("data: ", maxsplit=1)[1].strip()
    assert json.loads(payload) == {"content": "差\n旅"}
