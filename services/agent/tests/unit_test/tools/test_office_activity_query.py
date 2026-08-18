# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import aiohttp
import pytest

from vss_agents.tools.office_activity_query import OfficeActivityQueryConfig
from vss_agents.tools.office_activity_query import OfficeActivityQueryInput
from vss_agents.tools.office_activity_query import fetch_office_activity


@pytest.mark.asyncio
async def test_fetch_office_activity_filters_person_and_preserves_range() -> None:
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json = AsyncMock(
        return_value={
            "timezone": "Asia/Hong_Kong",
            "start": "2026-08-17T00:00:00+08:00",
            "end": "2026-08-18T00:00:00+08:00",
            "events": [
                {"person_id": 1, "person_name": "张三", "description": "阅读材料", "duration_seconds": 600},
                {"person_id": 2, "person_name": "李四", "description": "在电脑前工作", "duration_seconds": 900},
            ],
        },
    )
    response.__aenter__ = AsyncMock(return_value=response)
    response.__aexit__ = AsyncMock(return_value=False)
    session = MagicMock()
    session.get.return_value = response
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)

    config = OfficeActivityQueryConfig(office_api_url="http://office-api:8090")
    input_data = OfficeActivityQueryInput(date="2026-08-17", person="张")
    with patch("vss_agents.tools.office_activity_query.aiohttp.ClientSession", return_value=session):
        result = await fetch_office_activity(config, input_data)

    assert result.error is None
    assert result.event_count == 1
    assert result.total_seconds == 600
    assert result.people[0]["person_name"] == "张三"
    session.get.assert_called_once_with(
        "http://office-api:8090/api/activity/events",
        params={"date": "2026-08-17"},
    )


@pytest.mark.asyncio
async def test_fetch_office_activity_reports_connection_error() -> None:
    session = MagicMock()
    session.get.side_effect = aiohttp.ClientError("connection refused")
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)

    config = OfficeActivityQueryConfig(office_api_url="http://office-api:8090")
    with patch("vss_agents.tools.office_activity_query.aiohttp.ClientSession", return_value=session):
        result = await fetch_office_activity(config, OfficeActivityQueryInput(date="2026-08-17"))

    assert result.event_count == 0
    assert result.error is not None
    assert "Failed to reach office API" in result.error


@pytest.mark.asyncio
async def test_fetch_office_activity_detail_returns_motion_evidence() -> None:
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json = AsyncMock(return_value={
        "id": 7,
        "description": "从椅子上起身",
        "observations": [{
            "observed_actions": ["膝关节角度增大"],
            "motion_facts": {"posture_transitions": [{"type": "stood_up"}]},
            "storyboards": {"person": "/api/activity/evidence/4/person"},
        }],
    })
    response.__aenter__ = AsyncMock(return_value=response)
    response.__aexit__ = AsyncMock(return_value=False)
    session = MagicMock()
    session.get.return_value = response
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)

    config = OfficeActivityQueryConfig(office_api_url="http://office-api:8090")
    with patch("vss_agents.tools.office_activity_query.aiohttp.ClientSession", return_value=session):
        result = await fetch_office_activity(config, OfficeActivityQueryInput(event_id=7))

    assert result.event_count == 1
    assert result.event_detail is not None
    assert result.event_detail["id"] == 7
    session.get.assert_called_once_with(
        "http://office-api:8090/api/activity/events/7",
        params={},
    )
