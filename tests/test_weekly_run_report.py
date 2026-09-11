from datetime import datetime, timezone

from scripts.weekly_run_report import assign_lanes, make_report, workflow_jobs

WORKFLOW = """
name: example
jobs:
  prepare:
    name: Prepare
    steps: []
  scan:
    name: Scan (${{ matrix.part }})
    strategy:
      max-parallel: ${{ github.event_name == 'workflow_dispatch' && 2 || 5 }}
"""


def test_workflow_jobs_uses_scheduled_capacity():
    jobs = workflow_jobs(WORKFLOW, "schedule")
    assert [(job.job_id, job.display_prefix, job.capacity) for job in jobs] == [
        ("prepare", "Prepare", 1),
        ("scan", "Scan (", 5),
    ]


def test_workflow_jobs_uses_dispatch_capacity():
    assert workflow_jobs(WORKFLOW, "workflow_dispatch")[1].capacity == 2


def test_assign_lanes_reuses_first_available_lane():
    jobs = [
        {
            "name": "a",
            "started_at": "2026-01-01T00:00:00Z",
            "completed_at": "2026-01-01T01:00:00Z",
        },
        {
            "name": "b",
            "started_at": "2026-01-01T00:30:00Z",
            "completed_at": "2026-01-01T02:00:00Z",
        },
        {
            "name": "c",
            "started_at": "2026-01-01T01:15:00Z",
            "completed_at": "2026-01-01T03:00:00Z",
        },
    ]
    lanes = assign_lanes(jobs, 2, datetime.now(timezone.utc))
    assert [[job["name"] for job in lane] for lane in lanes] == [["a", "c"], ["b"]]


def test_queued_jobs_do_not_force_overlapping_executions_onto_one_lane():
    jobs = [
        {
            "name": "queued",
            "status": "queued",
            "started_at": "2026-01-01T00:00:00Z",
            "completed_at": None,
        },
        {
            "name": "shard 3",
            "status": "completed",
            "started_at": "2026-01-01T01:13:00Z",
            "completed_at": "2026-01-01T04:10:00Z",
        },
        {
            "name": "shard 5",
            "status": "completed",
            "started_at": "2026-01-01T01:13:00Z",
            "completed_at": "2026-01-01T08:33:00Z",
        },
    ]
    now = datetime(2026, 1, 1, 9, tzinfo=timezone.utc)
    lanes = assign_lanes(jobs, 2, now)
    shard_lanes = {
        job["name"]: lane_index
        for lane_index, lane in enumerate(lanes)
        for job in lane
        if job["name"].startswith("shard")
    }
    assert shard_lanes["shard 3"] != shard_lanes["shard 5"]


def test_report_does_not_redeclare_browser_top_global():
    run = {
        "id": 1,
        "name": "example",
        "event": "schedule",
        "status": "completed",
        "created_at": "2026-01-01T00:00:00Z",
        "html_url": "https://example.test/run/1",
    }
    report = make_report(run, workflow_jobs(WORKFLOW, "schedule"), [])
    assert "const top=" not in report
    assert "chartTop=104" in report
    assert "rotate(-35" in report
    assert "'text-anchor':'start'" in report
    assert "example · run 1 · snapshot 2026-01-01" in report
    assert "toLocaleTimeString" in report
    assert "Duration:" in report
    assert "formatDuration" in report
    assert "font-size:15px" in report
    assert "mouseenter" in report
    assert "var(--queued)" in report
    assert "right=140" in report
    assert "href:j.html_url" in report
    assert "target:'_blank'" in report
    assert "E = embedding" in report
    assert "G = generative" in report
    assert "Type: ${kindName}" in report
    assert "Queued jobs" in report
    assert "if(row.section_start)" in report
    assert "contextmenu" in report
    assert "Link copied to clipboard" in report
    assert '<html lang="en" data-theme="light">' in report
    assert 'data-theme="light"' in report
    assert "--ok:#3fb950" in report
    assert "--live:#58a6ff" in report
    assert "w=Math.max(2" in report
    assert "hitW=Math.max(24,w)" in report
