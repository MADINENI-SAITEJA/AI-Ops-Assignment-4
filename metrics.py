"""Application metrics; run this file to start the independent alert receiver."""

import ipaddress
import logging
import threading

import psutil
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Summary,
    ProcessCollector,
    PlatformCollector,
    GCCollector,
    start_http_server,
)


class Metrics:
    def __init__(self):
        self.registry = CollectorRegistry()

        ProcessCollector(registry=self.registry)
        PlatformCollector(registry=self.registry)
        GCCollector(registry=self.registry)

        def counter(name, description, labels=()):
            return Counter(
                name, description, labels, registry=self.registry
            )

        def gauge(name, description, labels=()):
            return Gauge(
                name, description, labels, registry=self.registry
            )

        def histogram(name, description, labels, buckets):
            return Histogram(
                name,
                description,
                labels,
                buckets=buckets,
                registry=self.registry,
            )

        # Counters: cumulative event counts.
        self.requests = counter(
            "caption_requests_total",
            "Completed upload requests",
            ("mode", "outcome"),
        )

        self.images = counter(
            "caption_images_total",
            "Image attempts including decoding failures",
            ("mode", "outcome"),
        )

        self.errors = counter(
            "caption_errors_total",
            "Errors by processing stage",
            ("stage",),
        )

        self.upload_bytes = counter(
            "caption_uploaded_bytes_total",
            "Submitted upload bytes",
            ("mode",),
        )

        self.clients = counter(
            "caption_client_requests_total",
            "Submitted requests by bounded browser session",
            ("mode", "session_id", "ip_class"),
        )

        self.rejected = counter(
            "caption_rejected_uploads_total",
            "Rejected uploads",
            ("mode", "reason"),
        )

        # Gauges: values that can rise or fall.
        self.active = gauge(
            "caption_active_requests",
            "Uploads currently being processed",
            ("mode",),
        )

        self.pending = gauge(
            "caption_pending_images",
            "Images remaining in active uploads",
            ("mode",),
        )

        self.inference_active = gauge(
            "caption_inference_active",
            "Currently executing model calls",
        )

        self.ready = gauge(
            "caption_model_ready",
            "Model loaded successfully: one means ready",
        )

        self.parameters = gauge(
            "caption_model_parameter_bytes",
            "Model parameter and buffer bytes",
        )

        self.rss = gauge(
            "caption_process_resident_memory_bytes",
            "Resident memory of the entire application",
        )

        self.cpu = gauge(
            "caption_process_cpu_percent",
            "Application CPU percent of total guest CPU capacity",
        )

        self.last_success = gauge(
            "caption_last_success_timestamp_seconds",
            "Unix time of the most recent successful caption",
        )

        # Histograms: distributions used for averages and percentiles.
        self.request_time = histogram(
            "caption_request_duration_seconds",
            "Upload duration including validation loading and processing",
            ("mode",),
            (0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300, 600, 1800),
        )

        self.inference_time = histogram(
            "caption_inference_duration_seconds",
            "Model generation duration only",
            ("mode",),
            (0.1, 0.25, 0.5, 1, 2, 3, 5, 10, 20, 30, 60, 120),
        )

        self.inference_cpu = Counter(
            "caption_inference_cpu_seconds_total",
            "Process CPU seconds during serialized model generation",
            ("mode",), registry=self.registry,
        )

        self.queue_time = histogram(
            "caption_queue_wait_seconds",
            "Time waiting for the model inference lock",
            ("mode",),
            (0.001, 0.01, 0.1, 0.5, 1, 5, 10, 30, 60, 300),
        )

        # Summaries expose observation count and sum.
        self.request_summary = Summary(
            "caption_request_seconds",
            "Summary of complete upload durations",
            ("mode",),
            registry=self.registry,
        )

        self.pixels = Summary(
            "caption_image_pixels",
            "Original decoded image pixel counts",
            ("mode",),
            registry=self.registry,
        )

        self.batch_size = Summary(
            "caption_batch_images",
            "Number of selected images per valid ZIP",
            registry=self.registry,
        )

        self._sessions = set()
        self._session_lock = threading.Lock()

        self.ready.set(0)
        self.inference_active.set(0)

        for mode in ("single", "bulk"):
            self.inference_cpu.labels(mode).inc(0)
            self.active.labels(mode).set(0)
            self.pending.labels(mode).set(0)
            self.upload_bytes.labels(mode).inc(0)

            for outcome in ("success", "partial", "error"):
                self.requests.labels(mode, outcome).inc(0)

            for outcome in ("success", "error"):
                self.images.labels(mode, outcome).inc(0)

        for stage in ("validation", "decode", "model_load", "inference"):
            self.errors.labels(stage).inc(0)

    def track_client(self, session_id, address, mode):
        # Limit distinct session IDs to prevent unlimited time-series growth.
        with self._session_lock:
            if session_id in self._sessions or len(self._sessions) < 32:
                self._sessions.add(session_id)
                label = session_id
            else:
                label = "overflow"

        try:
            ip = ipaddress.ip_address(address)
            if ip.is_loopback:
                category = "loopback"
            elif ip.is_private:
                category = "private"
            else:
                category = "public"
        except (ValueError, TypeError):
            category = "unknown"

        self.clients.labels(mode, label, category).inc()

    def start_server(self, port=18000, address="127.0.0.1"):
        server, thread = start_http_server(
            port,
            addr=address,
            registry=self.registry,
        )

        self._stop = threading.Event()

        def monitor():
            try:
                process = psutil.Process()
                process.cpu_percent(None)
            except psutil.Error as exc:
                self.rss.set(float("nan"))
                self.cpu.set(float("nan"))
                logging.warning("Resource sampling unavailable: %s", exc)
                return

            while not self._stop.wait(2):
                try:
                    self.rss.set(process.memory_info().rss)
                    self.cpu.set(
                        process.cpu_percent(None)
                        / max(psutil.cpu_count() or 1, 1)
                    )
                except psutil.Error:
                    self.rss.set(float("nan"))
                    self.cpu.set(float("nan"))

        self._resource_thread = threading.Thread(
            target=monitor,
            daemon=True,
        )
        self._resource_thread.start()

        return server, thread


# Independent Alertmanager notification receiver
import json
import threading
import time
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from prometheus_client import (
    CollectorRegistry,
    Counter,
    generate_latest,
    CONTENT_TYPE_LATEST,
)


def make_server(
    host="127.0.0.1",
    port=18001,
    log_path="runtime/alerts.jsonl",
):
    registry = CollectorRegistry()

    notifications = Counter(
        "caption_alert_notifications_total",
        "Alert notifications delivered to the local webhook",
        ("alertname", "status", "severity"),
        registry=registry,
    )

    log = Path(log_path)
    log.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def reply(self, code, body, content_type="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/metrics":
                self.reply(
                    200,
                    generate_latest(registry),
                    CONTENT_TYPE_LATEST,
                )
            elif self.path == "/health":
                self.reply(200, b'{"status":"ok"}')
            else:
                self.reply(404, b'{"error":"not found"}')

        def do_POST(self):
            if self.path != "/alerts":
                self.reply(404, b'{"error":"not found"}')
                return

            try:
                size = int(self.headers.get("Content-Length", "0"))

                if not 0 < size <= 1024 * 1024:
                    raise ValueError("Invalid payload size")

                payload = json.loads(self.rfile.read(size))
                alerts = payload["alerts"]

                if not isinstance(alerts, list):
                    raise ValueError("alerts must be a list")

                events = []

                for alert in alerts:
                    labels = alert.get("labels", {})

                    events.append(
                        (
                            labels.get("alertname", "unknown"),
                            alert.get(
                                "status",
                                payload.get("status", "unknown"),
                            ),
                            labels.get("severity", "unknown"),
                        )
                    )

                with lock:
                    with log.open("a", encoding="utf-8") as handle:
                        handle.write(
                            json.dumps(
                                {
                                    "received_at": time.time(),
                                    "payload": payload,
                                }
                            ) + "\n"
                        )

                    for event in events:
                        notifications.labels(*event).inc()

                self.reply(200, b'{"received":true}')

            except (ValueError, KeyError, TypeError, AttributeError):
                self.reply(400, b'{"error":"invalid alert payload"}')

    return ThreadingHTTPServer((host, port), Handler)


if __name__ == "__main__":
    print(
        "Alert receiver: http://127.0.0.1:18001/alerts",
        flush=True,
    )
    make_server().serve_forever()
