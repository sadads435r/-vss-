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

"""Tool to query the current office occupancy (how many people are working now)."""

from collections.abc import AsyncGenerator
import json
import logging

import aiohttp
from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from pydantic import BaseModel
from pydantic import Field

logger = logging.getLogger(__name__)


class OfficePeopleCountConfig(FunctionBaseConfig, name="office_people_count"):
    """Configuration for the office people count tool."""

    office_api_url: str = Field(
        ...,
        description="Base URL of the office assistant API (e.g., http://127.0.0.1:8090)",
    )
    timeout: int = Field(
        15,
        description="Request timeout in seconds",
    )


class OfficePeopleCountInput(BaseModel):
    """Input for querying current office people count."""

    detail: bool = Field(
        default=False,
        description="If true, include per-person work durations; otherwise only the count.",
    )


class OfficePeopleCountOutput(BaseModel):
    """Output from the office people count query."""

    working_count: int = Field(..., description="Number of people currently in the office / at their workstations")
    total_people: int = Field(..., description="Total number of registered people seen today")
    people: list[dict] = Field(
        default_factory=list,
        description="Per-person details (name, present, work_seconds) when detail=true",
    )
    raw: dict = Field(default_factory=dict, description="Raw API response")
    error: str | None = Field(None, description="Error message if the query failed")


@register_function(config_type=OfficePeopleCountConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def office_people_count(config: OfficePeopleCountConfig, builder: Builder) -> AsyncGenerator[FunctionInfo]:
    """Query the office assistant API for the current number of people working."""

    async def _office_people_count(input_data: OfficePeopleCountInput) -> OfficePeopleCountOutput:
        url = f"{config.office_api_url.rstrip('/')}/api/person/activity/today"
        timeout = aiohttp.ClientTimeout(total=config.timeout)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session, session.get(url) as response:
                response.raise_for_status()
                data = json.loads(await response.text())
        except aiohttp.ClientError as e:
            logger.error(f"Office API connection error: {e}")
            return OfficePeopleCountOutput(
                working_count=0,
                total_people=0,
                error=f"Failed to reach office API at {url}: {e}",
            )
        except Exception as e:
            logger.error(f"Office people count query failed: {e}")
            return OfficePeopleCountOutput(
                working_count=0,
                total_people=0,
                error=str(e),
            )

        working_count = int(data.get("working_count", 0))
        people = data.get("people", [])
        detail_people = [
            {
                "name": str(p.get("label", "unknown")),
                "present": bool(p.get("present", False)),
                "work_seconds": int(p.get("work_seconds", 0)),
            }
            for p in people
            if p.get("person_id") is not None
        ]
        return OfficePeopleCountOutput(
            working_count=working_count,
            total_people=len(detail_people),
            people=detail_people if input_data.detail else [],
            raw=data,
        )

    yield FunctionInfo.create(
        single_fn=_office_people_count,
        description=_office_people_count.__doc__,
        input_schema=OfficePeopleCountInput,
        single_output_schema=OfficePeopleCountOutput,
    )
