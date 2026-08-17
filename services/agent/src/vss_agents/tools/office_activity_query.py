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

"""Read-only Agent tool for querying the office daily activity log."""

from collections.abc import AsyncGenerator
import logging
from typing import TypedDict

import aiohttp
from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from pydantic import BaseModel
from pydantic import Field

logger = logging.getLogger(__name__)


class PersonSummary(TypedDict):
    """Per-person totals returned to the Agent."""

    person_id: object
    person_name: str
    event_count: int
    total_seconds: int


class OfficeActivityQueryConfig(FunctionBaseConfig, name="office_activity_query"):
    """Configuration for the office activity log tool."""

    office_api_url: str = Field(..., description="Base URL of the office assistant API")
    timeout: int = Field(15, description="Request timeout in seconds")


class OfficeActivityQueryInput(BaseModel):
    """Filters for an activity-log query."""

    date: str | None = Field(None, description="Local date in YYYY-MM-DD format")
    start: str | None = Field(None, description="Range start as an ISO date or datetime")
    end: str | None = Field(None, description="Range end as an ISO date or datetime")
    person: str | None = Field(None, description="Person name or a case-insensitive part of the name")
    keyword: str | None = Field(None, description="Activity keyword, such as 阅读, 交谈, or writing")
    limit: int = Field(200, ge=1, le=500, description="Maximum number of detailed events to return")


class OfficeActivityQueryOutput(BaseModel):
    """Structured activity events and per-person totals."""

    timezone: str = ""
    start: str = ""
    end: str = ""
    event_count: int = 0
    total_seconds: int = 0
    people: list[dict[str, object]] = Field(default_factory=list)
    events: list[dict[str, object]] = Field(default_factory=list)
    truncated: bool = False
    error: str | None = None


async def fetch_office_activity(
    config: OfficeActivityQueryConfig,
    input_data: OfficeActivityQueryInput,
) -> OfficeActivityQueryOutput:
    """Fetch activity events and apply the optional human-readable person filter."""
    url = f"{config.office_api_url.rstrip('/')}/api/activity/events"
    params: dict[str, str] = {}
    if input_data.date:
        params["date"] = input_data.date
    else:
        if input_data.start:
            params["start"] = input_data.start
        if input_data.end:
            params["end"] = input_data.end
    if input_data.keyword:
        params["q"] = input_data.keyword

    timeout = aiohttp.ClientTimeout(total=config.timeout)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session, session.get(url, params=params) as response:
            response.raise_for_status()
            data = await response.json(content_type=None)
    except aiohttp.ClientError as error:
        logger.error("Office activity API connection error: %s", error)
        return OfficeActivityQueryOutput(error=f"Failed to reach office API at {url}: {error}")
    except (TypeError, ValueError) as error:
        logger.error("Office activity API returned invalid data: %s", error)
        return OfficeActivityQueryOutput(error=f"Invalid office activity response: {error}")

    raw_events = data.get("events", []) if isinstance(data, dict) else []
    events = [event for event in raw_events if isinstance(event, dict)]
    if input_data.person:
        person_needle = input_data.person.strip().casefold()
        events = [event for event in events if person_needle in str(event.get("person_name", "")).casefold()]

    summaries: dict[str, PersonSummary] = {}
    for event in events:
        person_name = str(event.get("person_name", "未知人员"))
        summary = summaries.setdefault(
            person_name,
            {
                "person_id": event.get("person_id"),
                "person_name": person_name,
                "event_count": 0,
                "total_seconds": 0,
            },
        )
        summary["event_count"] += 1
        summary["total_seconds"] += int(event.get("duration_seconds", 0))

    truncated = len(events) > input_data.limit
    visible_events = events[: input_data.limit]
    return OfficeActivityQueryOutput(
        timezone=str(data.get("timezone", "")),
        start=str(data.get("start", "")),
        end=str(data.get("end", "")),
        event_count=len(events),
        total_seconds=sum(int(event.get("duration_seconds", 0)) for event in events),
        people=[dict(summary) for summary in summaries.values()],
        events=visible_events,
        truncated=truncated,
    )


@register_function(config_type=OfficeActivityQueryConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def office_activity_query(config: OfficeActivityQueryConfig, _builder: Builder) -> AsyncGenerator[FunctionInfo]:
    """Register the read-only office activity query tool."""

    async def _office_activity_query(input_data: OfficeActivityQueryInput) -> OfficeActivityQueryOutput:
        """Query what recognized people did during a date or time range."""
        return await fetch_office_activity(config, input_data)

    yield FunctionInfo.create(
        single_fn=_office_activity_query,
        description=_office_activity_query.__doc__,
        input_schema=OfficeActivityQueryInput,
        single_output_schema=OfficeActivityQueryOutput,
    )
