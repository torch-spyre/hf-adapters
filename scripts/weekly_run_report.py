#!/usr/bin/env python3
"""Create a self-contained timeline report for push-to-clickhouse.

The only runtime dependency is an authenticated GitHub CLI (`gh`).
"""

from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_REPO = "torch-spyre/hf-adapters"
DEFAULT_WORKFLOW = "push-to-clickhouse.yaml"


def gh_json(*args: str) -> Any:
    command = ["gh", *args]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise SystemExit("GitHub CLI (`gh`) is required but was not found") from exc
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() or exc.stdout.strip() or "unknown error"
        raise SystemExit(f"gh failed: {message}") from exc
    return json.loads(result.stdout)


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


@dataclass
class JobDefinition:
    job_id: str
    display_prefix: str
    capacity: int


def workflow_jobs(source: str, event: str) -> list[JobDefinition]:
    """Extract job names and max-parallel without requiring a YAML package."""
    in_jobs = False
    blocks: list[tuple[str, list[str]]] = []
    current: tuple[str, list[str]] | None = None
    for line in source.splitlines():
        if line == "jobs:":
            in_jobs = True
            continue
        if not in_jobs:
            continue
        match = re.match(r"^  ([A-Za-z0-9_-]+):\s*$", line)
        if match:
            current = (match.group(1), [])
            blocks.append(current)
        elif current is not None:
            current[1].append(line)

    definitions = []
    for job_id, lines in blocks:
        body = "\n".join(lines)
        name_match = re.search(r"^    name:\s*(.+?)\s*$", body, re.MULTILINE)
        display = name_match.group(1).strip("'\"") if name_match else job_id
        prefix = display.split("${{", 1)[0].rstrip()
        maximum = re.search(r"^\s+max-parallel:\s*(.+?)\s*$", body, re.MULTILINE)
        capacity = 1
        if maximum:
            value = maximum.group(1)
            conditional = re.search(
                r"github\.event_name\s*==\s*['\"]workflow_dispatch['\"]\s*&&\s*(\d+)\s*\|\|\s*(\d+)",
                value,
            )
            literal = re.fullmatch(r"\s*(\d+)\s*", value)
            if conditional:
                capacity = int(
                    conditional.group(1 if event == "workflow_dispatch" else 2)
                )
            elif literal:
                capacity = int(literal.group(1))
            else:
                raise SystemExit(
                    f"Cannot evaluate max-parallel for job {job_id}: {value}"
                )
        definitions.append(JobDefinition(job_id, prefix, capacity))
    return definitions


def assign_lanes(
    jobs: list[dict[str, Any]], capacity: int, now: datetime
) -> list[list[dict[str, Any]]]:
    lanes: list[list[dict[str, Any]]] = [[] for _ in range(capacity)]
    available = [datetime.min.replace(tzinfo=timezone.utc)] * capacity
    # GitHub populates started_at for queued matrix jobs with their queue time.
    # It is not an execution interval, so place queued jobs after real work as
    # zero-width markers at the report endpoint.
    ordered = sorted(
        jobs,
        key=lambda item: (
            item.get("status") == "queued",
            parse_time(item.get("started_at")) or now,
        ),
    )
    for job in ordered:
        queued = job.get("status") == "queued"
        start = now if queued else parse_time(job.get("started_at")) or now
        candidates = [index for index, end in enumerate(available) if end <= start]
        lane = (
            candidates[0]
            if candidates
            else min(range(capacity), key=available.__getitem__)
        )
        lanes[lane].append(job)
        available[lane] = (
            start if queued else parse_time(job.get("completed_at")) or now
        )
    return lanes


def fetch_run(repo: str, workflow: str, run_id: int | None) -> dict[str, Any]:
    if run_id is not None:
        return gh_json("api", f"repos/{repo}/actions/runs/{run_id}")
    runs = gh_json(
        "run",
        "list",
        "--repo",
        repo,
        "--workflow",
        workflow,
        "--limit",
        "1",
        "--json",
        "databaseId,event,headSha,name,status,conclusion,createdAt,updatedAt,url",
    )
    if not runs:
        raise SystemExit(f"No runs found for {workflow} in {repo}")
    run = runs[0]
    return {
        "id": run["databaseId"],
        "event": run["event"],
        "head_sha": run["headSha"],
        "name": run["name"],
        "status": run["status"],
        "conclusion": run["conclusion"],
        "created_at": run["createdAt"],
        "updated_at": run["updatedAt"],
        "html_url": run["url"],
    }


def fetch_workflow(repo: str, workflow: str, sha: str) -> str:
    data = gh_json(
        "api", f"repos/{repo}/contents/.github/workflows/{workflow}?ref={sha}"
    )
    import base64

    return base64.b64decode(data["content"]).decode()


def fetch_jobs(repo: str, run_id: int) -> list[dict[str, Any]]:
    pages = gh_json(
        "api",
        "--paginate",
        "--slurp",
        f"repos/{repo}/actions/runs/{run_id}/jobs?per_page=100",
    )
    return [job for page in pages for job in page.get("jobs", [])]


def make_report(
    run: dict[str, Any], definitions: list[JobDefinition], jobs: list[dict[str, Any]]
) -> str:
    now = datetime.now(timezone.utc)
    rows = []
    queued_groups = []
    unmatched = jobs[:]
    for definition in definitions:
        matched = [
            job
            for job in unmatched
            if job.get("name", "").startswith(definition.display_prefix)
        ]
        unmatched = [job for job in unmatched if job not in matched]
        queued = [job for job in matched if job.get("status") == "queued"]
        active = [job for job in matched if job.get("status") != "queued"]
        queued_groups.append({"label": definition.job_id, "jobs": queued})
        for index, lane in enumerate(assign_lanes(active, definition.capacity, now), 1):
            rows.append(
                {
                    "label": f"#{index}",
                    "jobs": lane,
                    "section_start": index == 1,
                    "group": definition.job_id,
                }
            )

    times = [parse_time(run.get("created_at")) or now]
    for job in jobs:
        times.extend(
            filter(
                None,
                [
                    parse_time(job.get("started_at")),
                    parse_time(job.get("completed_at")),
                ],
            )
        )
    if any(not job.get("completed_at") for job in jobs):
        times.append(now)
    payload = {
        "run": run,
        "rows": rows,
        "queued": queued_groups,
        "unmatched": [job.get("name") for job in unmatched],
        "start": min(times).timestamp() * 1000,
        "end": max(times).timestamp() * 1000,
        "generated": now.isoformat(),
    }
    safe_payload = json.dumps(payload).replace("</", "<\\/")
    run_date = (parse_time(run.get("created_at")) or now).date().isoformat()
    title = html.escape(
        f"{run.get('name', DEFAULT_WORKFLOW)} · run {run['id']} · snapshot {run_date}"
    )
    return f"""<!doctype html>
<html lang="en" data-theme="light"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>
:root{{--bg:#0d1117;--surface:#161b22;--panel:#21262d;--ink:#f0f6fc;--muted:#8b949e;--line:#30363d;--section:#6e7681;--bar-text:#0d1117;--tooltip-bg:#f0f6fc;--tooltip-ink:#0d1117;--ok:#3fb950;--bad:#f85149;--live:#58a6ff;--queued:#8b949e}}
[data-theme="light"]{{--bg:#f7f8fa;--surface:#fff;--panel:#eef1f5;--ink:#172033;--muted:#667085;--line:#d0d5dd;--section:#667085;--bar-text:#101828;--tooltip-bg:#101828;--tooltip-ink:#fff}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:16px system-ui,sans-serif}}
header{{padding:24px 28px 16px}}.title-row{{display:flex;align-items:center;justify-content:space-between;gap:18px}}h1{{font-size:22px;margin:0 0 6px}}a{{color:#79adff}}.meta{{color:var(--muted)}}#theme-toggle{{padding:7px 11px;border:1px solid var(--line);border-radius:6px;background:var(--surface);color:var(--ink);font:inherit;cursor:pointer}}
.legend{{display:flex;flex-wrap:wrap;gap:18px;margin-top:12px}}.key::before{{content:"";display:inline-block;width:12px;height:12px;margin-right:6px;border-radius:2px;background:var(--c);vertical-align:-1px}}.kind-key{{font-weight:700}}
#queue{{margin:0 20px 14px;padding:14px 18px;background:var(--panel);border:1px solid var(--line);border-radius:8px}}#queue h2{{font-size:17px;margin:0 0 10px}}.queue-groups{{display:flex;flex-wrap:wrap;gap:18px}}.queue-group{{min-width:180px}}.queue-group h3{{font-size:14px;margin:0 0 5px}}.queue-jobs{{display:flex;flex-wrap:wrap;gap:6px}}.queue-job{{display:inline-block;padding:4px 8px;border-radius:5px;background:var(--queued);color:#101828;text-decoration:none;font-size:14px;font-weight:650}}.queue-job:hover{{outline:2px solid var(--ink)}}
#wrap{{overflow:auto;margin:0 20px 28px;background:var(--surface);border:1px solid var(--line);border-radius:8px}}svg{{display:block;min-width:1100px}}.axis,.grid{{stroke:var(--line)}}.grid{{stroke-dasharray:3 4}}.label{{fill:var(--ink);font-size:14px;font-weight:500}}.tick{{fill:var(--muted);font-size:14px}}.bar{{stroke:var(--surface);stroke-width:1;rx:3;pointer-events:none}}.hit{{fill:transparent;cursor:pointer}}.kind{{fill:var(--bar-text);font-size:13px;font-weight:800;text-anchor:middle;pointer-events:none}}.empty{{fill:var(--panel);stroke:var(--line)}}
#tooltip{{display:none;position:fixed;z-index:10;max-width:520px;padding:10px 12px;border-radius:6px;background:var(--tooltip-bg);color:var(--tooltip-ink);font-size:15px;line-height:1.5;white-space:pre-line;pointer-events:none;box-shadow:0 4px 14px #0008}}
</style></head><body><header><div class="title-row"><h1>{title}</h1><button id="theme-toggle" type="button">Dark theme</button></div><div class="meta"><a href="{html.escape(run.get('html_url', ''))}">Open GitHub run</a> · event: {html.escape(run.get('event', 'unknown'))} · status: {html.escape(run.get('status', 'unknown'))}</div>
<div class="legend"><span class="key" style="--c:var(--ok)">successful</span><span class="key" style="--c:var(--bad)">failed</span><span class="key" style="--c:var(--live)">running</span><span class="kind-key">E = embedding</span><span class="kind-key">G = generative</span></div></header><section id="queue"><h2>Queued jobs</h2><div class="queue-groups"></div></section><div id="wrap"><svg id="chart"></svg></div><div id="tooltip"></div>
<script>const DATA={safe_payload};
const svg=document.querySelector('#chart'),tooltip=document.querySelector('#tooltip'),NS='http://www.w3.org/2000/svg',left=250,right=140,chartTop=104,rowH=32,width=Math.max(1100,innerWidth-42),height=chartTop+DATA.rows.length*rowH+34;
const themeToggle=document.querySelector('#theme-toggle');themeToggle.addEventListener('click',()=>{{const light=document.documentElement.dataset.theme==='light';document.documentElement.dataset.theme=light?'dark':'light';themeToggle.textContent=light?'Light theme':'Dark theme';}});
svg.setAttribute('width',width);svg.setAttribute('height',height);svg.setAttribute('viewBox',`0 0 ${{width}} ${{height}}`);
const add=(tag,a,p=svg)=>{{const e=document.createElementNS(NS,tag);for(const[k,v]of Object.entries(a))e.setAttribute(k,v);p.append(e);return e}}, span=Math.max(DATA.end-DATA.start,60000),x=t=>left+(t-DATA.start)/(span)*(width-left-right);
const formatTime=t=>new Date(t).toLocaleString(),formatDuration=ms=>{{const total=Math.max(0,Math.round(ms/1000)),h=Math.floor(total/3600),m=Math.floor(total%3600/60),s=total%60;return [h&&`${{h}}h`,(h||m)&&`${{m}}m`,`${{s}}s`].filter(Boolean).join(' ');}};
const copyText=async value=>{{try{{await navigator.clipboard.writeText(value);}}catch{{const area=document.createElement('textarea');area.value=value;area.style.position='fixed';area.style.opacity='0';document.body.append(area);area.select();document.execCommand('copy');area.remove();}}}};
const queueRoot=document.querySelector('.queue-groups');let queuedCount=0;DATA.queued.forEach(group=>{{if(!group.jobs.length)return;queuedCount+=group.jobs.length;const box=document.createElement('div');box.className='queue-group';const heading=document.createElement('h3');heading.textContent=`${{group.label}} (${{group.jobs.length}})`;box.append(heading);const list=document.createElement('div');list.className='queue-jobs';group.jobs.forEach(job=>{{const kind=job.name.includes('(embedding ')?'E':job.name.includes('(generative ')?'G':'•',link=document.createElement('a');link.className='queue-job';link.href=job.html_url;link.target='_blank';link.rel='noopener';link.textContent=kind;link.title=job.name;list.append(link);}});box.append(list);queueRoot.append(box);}});if(!queuedCount)queueRoot.textContent='None';
for(let i=0;i<=8;i++){{const xx=left+(width-left-right)*i/8,t=DATA.start+span*i/8,labelY=chartTop-24;add('line',{{x1:xx,y1:chartTop-12,x2:xx,y2:height-25,class:'grid'}});const q=add('text',{{x:xx,y:labelY,'text-anchor':'start',transform:`rotate(-35 ${{xx}} ${{labelY}})`,class:'tick'}});q.textContent=new Date(t).toLocaleTimeString([],{{hour:'2-digit',minute:'2-digit'}});}}
DATA.rows.forEach((row,i)=>{{const y=chartTop+i*rowH;if(row.section_start){{add('line',{{x1:0,y1:y,x2:width,y2:y,stroke:'var(--section)','stroke-width':2}});const heading=add('text',{{x:18,y:y+21,class:'label','font-weight':'800'}});heading.textContent=row.group;}}let q=add('text',{{x:left-10,y:y+21,'text-anchor':'end',class:'label'}});q.textContent=row.label;add('line',{{x1:left,y1:y+rowH,x2:width-right,y2:y+rowH,class:'axis'}});if(!row.jobs.length)add('rect',{{x:left,y:y+7,width:3,height:18,class:'empty'}});row.jobs.forEach(j=>{{const s=j.started_at?Date.parse(j.started_at):DATA.end,e=j.completed_at?Date.parse(j.completed_at):DATA.end,startX=x(s),w=Math.max(2,x(e)-startX),hitW=Math.max(24,w),hitX=Math.max(left,startX-(hitW-w)/2),cls=!j.completed_at?'var(--live)':j.conclusion==='success'?'var(--ok)':'var(--bad)',kind=j.name.includes('(embedding ')?'E':j.name.includes('(generative ')?'G':'',kindName=kind==='E'?'Embedding':kind==='G'?'Generative':'Other',link=add('a',{{href:j.html_url,target:'_blank',rel:'noopener'}}),hit=add('rect',{{x:hitX,y:y+4,width:hitW,height:24,class:'hit'}},link),r=add('rect',{{x:startX,y:y+5,width:w,height:22,fill:cls,class:'bar'}},link),kindLabel=add('text',{{x:startX+w/2,y:y+21,class:'kind'}},link),start=j.started_at?formatTime(s):'Not started',end=j.completed_at?formatTime(e):'Still running',details=`${{j.name}}\nType: ${{kindName}}\nStatus: ${{j.conclusion||j.status}}\nStart: ${{start}}\nEnd: ${{end}}\nDuration: ${{formatDuration(e-s)}}`;kindLabel.textContent=w>=18?kind:'';hit.addEventListener('mouseenter',()=>{{r.setAttribute('stroke','var(--ink)');r.setAttribute('stroke-width','2');tooltip.textContent=details;tooltip.style.display='block';}});hit.addEventListener('mousemove',event=>{{tooltip.style.left=`${{Math.min(event.clientX+14,innerWidth-tooltip.offsetWidth-12)}}px`;tooltip.style.top=`${{Math.min(event.clientY+14,innerHeight-tooltip.offsetHeight-12)}}px`;}});hit.addEventListener('mouseleave',()=>{{r.removeAttribute('stroke');r.removeAttribute('stroke-width');tooltip.style.display='none';}});hit.addEventListener('contextmenu',async event=>{{event.preventDefault();await copyText(j.html_url);tooltip.textContent='Link copied to clipboard';tooltip.style.display='block';tooltip.style.left=`${{Math.min(event.clientX+14,innerWidth-tooltip.offsetWidth-12)}}px`;tooltip.style.top=`${{Math.min(event.clientY+14,innerHeight-tooltip.offsetHeight-12)}}px`;setTimeout(()=>tooltip.style.display='none',1200);}});}})}});
</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--workflow", default=DEFAULT_WORKFLOW)
    parser.add_argument("--run-id", type=int, help="specific run (default: latest)")
    parser.add_argument("--output", type=Path, default=Path("weekly-run-report.html"))
    args = parser.parse_args()
    run = fetch_run(args.repo, args.workflow, args.run_id)
    source = fetch_workflow(args.repo, args.workflow, run["head_sha"])
    jobs = fetch_jobs(args.repo, int(run["id"]))
    report = make_report(run, workflow_jobs(source, run["event"]), jobs)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote {args.output} for {run['html_url']}")


if __name__ == "__main__":
    main()
