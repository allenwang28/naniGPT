"""Perfetto trace export and local serving.

Handles Chrome trace export (gzip-compressed), optional wandb upload,
and a local HTTP server with CORS headers so Perfetto UI can fetch
traces from localhost — nothing leaves your machine.

Usage:

    from nanigpt.profiling.perfetto import export_trace, serve_traces

    gz_path = export_trace(prof, start_step=10, end_step=12, trace_dir=Path("traces"))
    serve_traces([gz_path])   # blocks until fetched or Ctrl+C
"""

import gzip
import http.server
import logging
import os
import socketserver
import threading
from pathlib import Path

import torch.profiler

log = logging.getLogger(__name__)

_PERFETTO_PORT = 9001


def export_trace(
    prof: torch.profiler.profile,
    start_step: int,
    end_step: int,
    trace_dir: Path,
) -> Path:
    """Export Chrome trace to a gzipped file and optionally upload to wandb.

    export_chrome_trace() must run before the profiler exits, so it happens inline.
    Returns the path to the gzipped trace file.
    """
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = trace_dir / f"trace_steps_{start_step}_{end_step}.json"
    prof.export_chrome_trace(str(trace_path))

    # Compress to gzip
    gz_path = trace_path.with_suffix(".json.gz")
    with open(trace_path, "rb") as f_in, gzip.open(gz_path, "wb") as f_out:
        f_out.write(f_in.read())
    trace_path.unlink()  # remove uncompressed

    log.info(f"Trace saved: {gz_path}")

    # Upload to wandb in background if available
    _upload_wandb_trace(gz_path, start_step, end_step)

    return gz_path


def _upload_wandb_trace(gz_path: Path, start_step: int, end_step: int) -> None:
    """Upload a trace file to wandb as an artifact in a background thread."""
    try:
        import wandb

        if wandb.run is None:
            return
    except Exception:
        return

    def _upload() -> None:
        try:
            import wandb

            artifact = wandb.Artifact(
                name=f"profile-trace-steps-{start_step}-{end_step}",
                type="profile",
            )
            artifact.add_file(str(gz_path))
            wandb.log_artifact(artifact)
            log.debug(f"Uploaded trace artifact for steps {start_step}-{end_step}")
        except Exception:
            log.warning("Failed to upload wandb trace artifact.", exc_info=True)

    thread = threading.Thread(target=_upload, daemon=True)
    thread.start()


class _CORSHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler with CORS headers so Perfetto UI can fetch from localhost."""

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        return super().end_headers()

    def log_message(self, format, *args):
        # Silence request logging
        pass


def serve_traces(trace_paths: list[Path]) -> None:
    """Serve trace files via local HTTP and print Perfetto UI links.

    Spins up a local HTTP server on port 9001 with CORS headers. Perfetto UI
    fetches traces from localhost — nothing leaves your machine. Blocks until
    all traces have been fetched or the user interrupts with Ctrl+C.
    """
    if not trace_paths:
        return

    # All traces should be in the same directory (traces/)
    trace_dir = trace_paths[0].parent
    filenames = {p.name for p in trace_paths}
    fetched: set[str] = set()

    class _TrackingHandler(_CORSHandler):
        def do_GET(self):
            # Track which trace files Perfetto has fetched
            requested = self.path.lstrip("/")
            if requested in filenames:
                fetched.add(requested)
            return super().do_GET()

    orig_dir = Path.cwd()
    try:
        os.chdir(trace_dir)
        socketserver.TCPServer.allow_reuse_address = True
        with socketserver.TCPServer(("127.0.0.1", _PERFETTO_PORT), _TrackingHandler) as httpd:
            log.info("Serving traces for Perfetto UI. Open these links in your browser:")
            for p in trace_paths:
                url = f"https://ui.perfetto.dev/#!/?url=http://127.0.0.1:{_PERFETTO_PORT}/{p.name}"
                log.info(f"  {url}")
            log.info("Press Ctrl+C to stop serving.")

            try:
                while fetched != filenames:
                    httpd.handle_request()
            except KeyboardInterrupt:
                pass

            log.info("All traces fetched. Server stopped.")
    except OSError as e:
        log.warning(f"Could not start Perfetto server on port {_PERFETTO_PORT}: {e}")
        log.info("Traces are saved locally. Open ui.perfetto.dev and drag-drop the files:")
        for p in trace_paths:
            log.info(f"  {p}")
    finally:
        os.chdir(orig_dir)
