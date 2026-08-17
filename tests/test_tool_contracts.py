from datetime import date

import pytest
from pydantic import ValidationError

from enterprise_ai_assistant.tools import LeaveRequestInput, TravelApplicationInput


def test_travel_application_rejects_reverse_date_range() -> None:
    with pytest.raises(ValidationError, match="end_date"):
        TravelApplicationInput(
            destination="上海",
            start_date=date(2026, 8, 20),
            end_date=date(2026, 8, 19),
            purpose="客户交流",
        )


def test_leave_request_rejects_reverse_date_range() -> None:
    with pytest.raises(ValidationError, match="end_date"):
        LeaveRequestInput(
            leave_type="annual",
            start_date=date(2026, 8, 20),
            end_date=date(2026, 8, 19),
        )
