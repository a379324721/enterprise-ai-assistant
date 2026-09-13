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


def test_one_way_trip_needs_no_end_date() -> None:
    trip = TravelApplicationInput(
        destination="上海",
        start_date=date(2026, 9, 16),
        trip_type="one_way",
        purpose="培训",
    )

    assert trip.end_date is None


def test_round_trip_still_requires_end_date() -> None:
    """放开 end_date 不能让漏问返程日期的往返申请混过校验。"""
    with pytest.raises(ValidationError, match="end_date is required"):
        TravelApplicationInput(destination="上海", start_date=date(2026, 9, 16), purpose="培训")


def test_one_way_trip_rejects_end_date() -> None:
    with pytest.raises(ValidationError, match="one_way"):
        TravelApplicationInput(
            destination="上海",
            start_date=date(2026, 9, 16),
            end_date=date(2026, 9, 18),
            trip_type="one_way",
            purpose="培训",
        )


def test_leave_request_rejects_reverse_date_range() -> None:
    with pytest.raises(ValidationError, match="end_date"):
        LeaveRequestInput(
            leave_type="annual",
            start_date=date(2026, 8, 20),
            end_date=date(2026, 8, 19),
        )
