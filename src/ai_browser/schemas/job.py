"""
Job posting schema for structured extraction.

This is the test-case schema used to validate the end-to-end pipeline:
    URL → find careers → extract jobs → save results

The schema is intentionally generic — it captures fields commonly found
across most job listing pages.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class JobPosting(BaseModel):
    """Structured representation of a single job posting."""

    title: str = Field(
        description="The job title (e.g., 'Senior Software Engineer')"
    )
    company: str = Field(
        default="",
        description="The company name, if listed on the page"
    )
    location: str = Field(
        default="",
        description="The job location (e.g., 'Remote', 'New York, NY')"
    )
    department: str = Field(
        default="",
        description="The department or team (e.g., 'Engineering', 'Marketing')"
    )
    employment_type: str = Field(
        default="",
        description="Employment type (e.g., 'Full-time', 'Part-time', 'Contract')"
    )
    description_summary: str = Field(
        default="",
        description="A brief summary of the job description"
    )
    url: str = Field(
        default="",
        description="Direct URL to the full job posting"
    )
    salary_range: str = Field(
        default="",
        description="Salary range if listed (e.g., '$120K-$180K')"
    )
    date_posted: str = Field(
        default="",
        description="Date the job was posted, if available"
    )


class JobPostingList(BaseModel):
    """Container for multiple job postings extracted from a page."""

    jobs: list[JobPosting] = Field(
        default_factory=list,
        description="List of job postings found on the page"
    )
    source_url: str = Field(
        default="",
        description="The URL from which these jobs were extracted"
    )
    total_count: int = Field(
        default=0,
        description="Total number of jobs found (may differ from len(jobs) if paginated)"
    )
