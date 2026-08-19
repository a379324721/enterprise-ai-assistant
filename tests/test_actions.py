import pytest

from enterprise_ai_assistant.repositories.actions import _decode_json_object


def test_decode_json_object_accepts_asyncpg_json_string() -> None:
    result = _decode_json_object(
        '{"reference_id":"travel-1","status":"recorded","destination":"上海"}'
    )

    assert result == {
        "reference_id": "travel-1",
        "status": "recorded",
        "destination": "上海",
    }


def test_decode_json_object_accepts_mapping_from_custom_codec() -> None:
    value = {"reference_id": "travel-1", "status": "recorded"}

    assert _decode_json_object(value) == value


def test_decode_json_object_rejects_non_object_json() -> None:
    with pytest.raises(TypeError, match="must be an object"):
        _decode_json_object('["unexpected"]')
