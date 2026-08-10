import json
import os
import subprocess
import sys
from pathlib import Path


def test_exception_is_written_with_traceback(tmp_path: Path) -> None:
    log_file = tmp_path / "assistant.log"
    environment = {**os.environ, "TEST_LOG_FILE": str(log_file)}
    script = """
import os
import structlog

from enterprise_ai_assistant.core.logging import configure_logging

configure_logging(
    os.environ["TEST_LOG_FILE"],
    max_bytes=1024,
    backup_count=1,
    environment="development",
)
try:
    raise ValueError("测试异常")
except ValueError:
    structlog.get_logger().exception("test_failure")
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    file_record = json.loads(log_file.read_text(encoding="utf-8").strip())
    assert "test_failure" in completed.stdout
    assert "ValueError: 测试异常" in completed.stdout
    assert "Traceback (most recent call last)" in completed.stdout
    assert "\\n" not in completed.stdout
    assert file_record["event"] == "test_failure"
    assert "ValueError: 测试异常" in file_record["exception"]
    assert "Traceback (most recent call last)" in file_record["exception"]
    assert "exc_info" not in file_record
