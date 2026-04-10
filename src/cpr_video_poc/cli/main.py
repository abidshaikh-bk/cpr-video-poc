from __future__ import annotations

import argparse
import json

from cpr_video_poc.orchestration.job_store import JobStore
from cpr_video_poc.orchestration.worker import run_worker
from cpr_video_poc.pipeline.generate import run_generation
from cpr_video_poc.settings import load_settings


def _print_output(data: object, as_json: bool) -> None:
    if not as_json and isinstance(data, str):
        print(data)
        return
    print(json.dumps(data, indent=2))


def _print_job_list(items: list[dict[str, object]]) -> None:
    if not items:
        print("No jobs found.")
        return
    for item in items:
        manifest = item["manifest"]
        status = item["status"]
        prompt = str(manifest.get("prompt", "")).strip().replace("\n", " ")
        prompt = prompt[:80] + ("..." if len(prompt) > 80 else "")
        print(f'{item["job_id"]} | {status.get("state")} | {prompt}')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cpr-video-poc")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate", help="Generate a CPR training video locally")
    generate_parser.add_argument("--prompt", required=True, help="Text prompt")
    generate_parser.add_argument("--seed", type=int, default=None, help="Optional seed")
    generate_parser.add_argument("--config", default=None, help="Path to project config YAML")
    generate_parser.add_argument("--backend-config", default=None, help="Path to backend config YAML")
    generate_parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full structured result instead of only the run directory",
    )

    submit_parser = subparsers.add_parser("submit", help="Submit a generation job into shared storage")
    submit_parser.add_argument("--prompt", required=True, help="Text prompt")
    submit_parser.add_argument("--seed", type=int, default=None, help="Optional seed")
    submit_parser.add_argument("--shared-root", required=True, help="Path to the shared job root")
    submit_parser.add_argument("--config", default=None, help="Path to project config YAML")
    submit_parser.add_argument("--backend-config", default=None, help="Path to backend config YAML")
    submit_parser.add_argument("--json", action="store_true", help="Print the full job payload as JSON")

    status_parser = subparsers.add_parser("status", help="Inspect jobs stored in the shared job root")
    status_parser.add_argument("--shared-root", required=True, help="Path to the shared job root")
    status_parser.add_argument("--job-id", default=None, help="Optional job id for a single-job view")
    status_parser.add_argument("--json", action="store_true", help="Print JSON output")

    requeue_parser = subparsers.add_parser("requeue", help="Move a failed or stale job back to queued")
    requeue_parser.add_argument("--shared-root", required=True, help="Path to the shared job root")
    requeue_parser.add_argument("--job-id", required=True, help="Job id to requeue")
    requeue_parser.add_argument("--json", action="store_true", help="Print JSON output")

    worker_parser = subparsers.add_parser("worker", help="Process queued jobs from shared storage")
    worker_parser.add_argument("--shared-root", required=True, help="Path to the shared job root")
    worker_parser.add_argument("--worker-id", default=None, help="Optional worker identifier")
    worker_parser.add_argument("--once", action="store_true", help="Process at most one job and exit")
    worker_parser.add_argument("--max-jobs", type=int, default=None, help="Maximum jobs to process before exit")
    worker_parser.add_argument("--poll-interval", type=int, default=15, help="Queue poll interval in seconds")
    worker_parser.add_argument(
        "--heartbeat-interval",
        type=int,
        default=30,
        help="Heartbeat interval in seconds while a job is running",
    )
    worker_parser.add_argument(
        "--stale-after",
        type=int,
        default=300,
        help="Reclaim running jobs whose heartbeat is older than this many seconds",
    )
    worker_parser.add_argument("--json", action="store_true", help="Print JSON output")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "generate":
        result = run_generation(
            prompt=args.prompt,
            seed=args.seed,
            project_config_path=args.config,
            backend_config_path=args.backend_config,
        )
        _print_output(result if args.json else result["run_dir"], as_json=args.json)
        return

    if args.command == "submit":
        settings = load_settings(args.config, args.backend_config)
        store = JobStore(args.shared_root)
        created = store.create_job(
            prompt=args.prompt,
            seed=args.seed,
            settings=settings,
        )
        output = {
            "job_id": created["job_id"],
            "job_dir": str(created["job_dir"]),
            "state": created["status"]["state"],
        }
        _print_output(output if args.json else created["job_id"], as_json=args.json)
        return

    if args.command == "status":
        store = JobStore(args.shared_root)
        if args.job_id:
            payload = {
                "job_id": args.job_id,
                "manifest": store.read_manifest(args.job_id),
                "status": store.read_status(args.job_id),
            }
            _print_output(payload, as_json=args.json)
        else:
            payload = store.list_jobs()
            if args.json:
                _print_output(payload, as_json=True)
            else:
                _print_job_list(payload)
        return

    if args.command == "requeue":
        store = JobStore(args.shared_root)
        payload = store.requeue_job(args.job_id)
        _print_output(payload if args.json else args.job_id, as_json=args.json)
        return

    if args.command == "worker":
        result = run_worker(
            shared_root=args.shared_root,
            worker_id=args.worker_id,
            poll_interval_seconds=args.poll_interval,
            heartbeat_interval_seconds=args.heartbeat_interval,
            stale_after_seconds=args.stale_after,
            once=args.once,
            max_jobs=args.max_jobs,
        )
        _print_output(result if args.json else str(result["processed_count"]), as_json=args.json)
        return


if __name__ == "__main__":
    main()
