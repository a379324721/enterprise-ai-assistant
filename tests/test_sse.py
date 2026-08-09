import json

from langchain_core.messages import AIMessage, HumanMessage

from enterprise_ai_assistant.api.routes import _conversation_messages, _encode_sse


def test_sse_event_is_typed_utf8_json() -> None:
    encoded = _encode_sse("token", {"content": "差\n旅"})

    assert encoded.startswith("event: token\ndata: ")
    assert encoded.endswith("\n\n")
    payload = encoded.split("data: ", maxsplit=1)[1].strip()
    assert json.loads(payload) == {"content": "差\n旅"}


def test_conversation_messages_keep_only_final_answer_per_turn() -> None:
    messages = [
        HumanMessage(content="我明天要出差"),
        AIMessage(content="正在查询差旅制度"),
        AIMessage(content="还需要补充出差地点。"),
        HumanMessage(content="去上海"),
        AIMessage(content="已记录出差地点。"),
    ]

    transcript = _conversation_messages(messages)

    assert [message.model_dump() for message in transcript] == [
        {"role": "user", "text": "我明天要出差"},
        {"role": "assistant", "text": "还需要补充出差地点。"},
        {"role": "user", "text": "去上海"},
        {"role": "assistant", "text": "已记录出差地点。"},
    ]
