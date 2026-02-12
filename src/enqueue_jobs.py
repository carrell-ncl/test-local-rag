import json
import subprocess

with open("job.json", "r") as f:
    jobs = json.load(f)

if not isinstance(jobs, list):
    raise SystemExit("jobs.json must be a JSON list of job objects")

for job in jobs:
    if not isinstance(job, dict):
        raise SystemExit("Each item in jobs.json must be a JSON object")

    subprocess.run(
        ["python", "-m", "src.ingest_worker", "--enqueue", json.dumps(job)],
        check=True,
    )

print(f"Enqueued {len(jobs)} jobs.")
