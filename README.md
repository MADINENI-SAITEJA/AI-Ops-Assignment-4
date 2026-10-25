# DA3408 Assignment 4 - CPU Image Captioning and Monitoring

A Streamlit application generates captions with
`Salesforce/blip-image-captioning-base` on the CPU. It supports individual
images and ZIP batches. Prometheus collects application and Ubuntu guest
metrics, Grafana visualizes them, and Alertmanager sends notifications to an
independent local receiver.

## Project files

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit interface, CPU model, image validation and ZIP processing |
| `metrics.py` | Application metric definitions and independent alert receiver |
| `prometheus.yml` | Scrape targets, evaluation interval and Alertmanager connection |
| `alerts.yml` | Five alert rules |
| `alertmanager.yml` | Webhook routing and resolved notifications |
| `grafana_dashboard.json` | Portable dashboard export: 29 panels in five sections |
| `README.md` | Setup, startup commands and usage |
| `AI_disclosure.md` | AI assistance and student contributions |

## Environment and prerequisites

The tested environment is an Ubuntu ARM64 guest in UTM, Python 3.12,
CPU-only PyTorch, and two model CPU threads. **Run the commands below inside
Ubuntu**, not in the macOS copy of the submission folder.

The project is located at `~/monitoring/DA3408_A4`. If using another directory,
adjust the `cd` commands. The native monitoring programs are installed at:

- `~/monitoring/prometheus-3.15.0.linux-arm64/`
- `~/monitoring/node_exporter-1.12.1.linux-arm64/`
- `~/monitoring/alertmanager-0.34.1.linux-arm64/`

Docker must be running, with an existing Grafana container named `grafana`.
These binaries and the Docker container are external prerequisites, not
files included in this repository. For another installation, use binaries
for that machine's architecture and adjust their paths. Official setup
references: [Prometheus downloads](https://prometheus.io/download/),
[Grafana in Docker](https://grafana.com/docs/grafana/latest/setup-grafana/installation/docker/),
and [uv installation](https://docs.astral.sh/uv/getting-started/installation/).

### Python setup - first run only

Skip this step when the tested `.venv` already exists and its dependencies
are installed. The dependency list is kept here instead of a separate
requirements file.

```bash
cd "$HOME/monitoring/DA3408_A4"

uv python install 3.12
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --torch-backend=cpu \
  torch==2.14.1+cpu \
  streamlit==1.66.1 \
  transformers==4.57.6 \
  Pillow==12.3.0 \
  prometheus-client==0.26.0 \
  psutil==7.2.2 \
  requests==2.34.2 \
  safetensors==0.8.0
```

The first model load requires an internet connection to download the model
files, including an approximately 990 MB checkpoint. Later loads reuse the
local Hugging Face cache. The application explicitly places the model on
the CPU and disables gradient tracking during generation.

## Run the project

Use two terminals. Start each service once; the commands assume its previous
instance has stopped. Terminal 1 remains occupied by Streamlit. Terminal 2
starts the monitoring services in the background and remains available for
checks. The blocks below start and verify the finished project; no source
patching or dashboard-generation scripts are needed.

### Terminal 1 - Streamlit

```bash
cd "$HOME/monitoring/DA3408_A4"

export CAPTION_CPU_THREADS=2
export OMP_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false

.venv/bin/python -m streamlit run app.py \
  --server.port=8501 \
  --server.address=0.0.0.0 \
  --server.maxUploadSize=100 \
  --server.fileWatcherType=none \
  --browser.gatherUsageStats=false \
  --theme.base=light \
  --theme.primaryColor='#0072B2'
```

Open [Streamlit](http://localhost:8501) in the Ubuntu browser. Opening the
page initializes the separate application metrics server on port **18000**.
Click **Load model / warm up** and wait for **Model ready** before a
performance demonstration. Keep this terminal running.

### Terminal 2 - validate configuration

```bash
cd "$HOME/monitoring/DA3408_A4"

mkdir -p runtime/prometheus runtime/alertmanager

"$HOME/monitoring/prometheus-3.15.0.linux-arm64/promtool" \
  check config "$PWD/prometheus.yml"

"$HOME/monitoring/alertmanager-0.34.1.linux-arm64/amtool" \
  check-config "$PWD/alertmanager.yml"
```

Continue after both checks succeed. Promtool should also report **five
alert rules** in `alerts.yml`.

### Terminal 2 - start monitoring

```bash
nohup .venv/bin/python -u metrics.py \
  > runtime/alert_receiver.log 2>&1 < /dev/null &
echo $! > runtime/alert_receiver.pid

nohup "$HOME/monitoring/node_exporter-1.12.1.linux-arm64/node_exporter" \
  --web.listen-address=127.0.0.1:9100 \
  > runtime/node_exporter.log 2>&1 < /dev/null &
echo $! > runtime/node_exporter.pid

nohup "$HOME/monitoring/alertmanager-0.34.1.linux-arm64/alertmanager" \
  --config.file="$PWD/alertmanager.yml" \
  --storage.path="$PWD/runtime/alertmanager" \
  --web.listen-address=127.0.0.1:9093 \
  --cluster.listen-address= \
  > runtime/alertmanager.log 2>&1 < /dev/null &
echo $! > runtime/alertmanager.pid

nohup "$HOME/monitoring/prometheus-3.15.0.linux-arm64/prometheus" \
  --config.file="$PWD/prometheus.yml" \
  --storage.tsdb.path="$PWD/runtime/prometheus" \
  --storage.tsdb.retention.time=3d \
  --web.listen-address=0.0.0.0:9090 \
  > runtime/prometheus.log 2>&1 < /dev/null &
echo $! > runtime/prometheus.pid

docker restart grafana
```

Running `metrics.py` directly starts the receiver on **18001**. The
application imports its `Metrics` class to serve application metrics on
**18000**. The separate receiver continues running when Streamlit stops.
Runtime data, process IDs and logs are created under `runtime/`.

### Terminal 2 - verify services

Wait approximately ten seconds after startup, then run:

```bash
.venv/bin/python - <<'PY'
import requests

client = requests.Session()
client.trust_env = False

def query(expression):
    response = client.get(
        "http://127.0.0.1:9090/api/v1/query",
        params={"query": expression},
        timeout=10,
    )
    response.raise_for_status()
    payload = response.json()
    assert payload["status"] == "success", payload
    return payload["data"]["result"]

targets = {
    row["metric"]["job"]: row["value"][1]
    for row in query("up")
}

for job in ("prometheus", "caption_app", "node", "alert_receiver"):
    print(job, "=", targets.get(job, "MISSING"))

ready = query('caption_model_ready{job="caption_app"}')
print("Model ready =", ready[0]["value"][1] if ready else "MISSING")

response = client.get("http://127.0.0.1:18001/health", timeout=10)
response.raise_for_status()
print("Receiver health:", response.json())
PY
```

Expected output after warming up the model:

```text
prometheus = 1
caption_app = 1
node = 1
alert_receiver = 1
Model ready = 1
Receiver health: {'status': 'ok'}
```

## Browser endpoints

Open these addresses inside Ubuntu:

| Component | Address |
| --- | --- |
| Streamlit | http://localhost:8501 |
| Application metrics | http://localhost:18000/metrics |
| Prometheus queries | http://localhost:9090/query |
| Prometheus targets | http://localhost:9090/targets |
| Prometheus alerts | http://localhost:9090/alerts |
| Node exporter | http://localhost:9100/metrics |
| Alertmanager | http://localhost:9093 |
| Receiver health | http://localhost:18001/health |
| Receiver notification metrics | http://localhost:18001/metrics |
| Grafana | http://localhost:3000 |

Prometheus listens on `0.0.0.0:9090` so Grafana in Docker can reach it.
The native exporters and receiver listen on the guest loopback interface.

## Grafana dashboard

The existing VM already has the dashboard saved in Grafana. For a new
Grafana installation:

1. Create a **Prometheus** data source named **prometheus**.
2. Set its URL to the Docker container's host gateway. In the tested default
   bridge network this is `http://172.17.0.1:9090`.
3. Set the data source scrape interval to **5s**, then save and test it.
4. Import `grafana_dashboard.json` and select **prometheus** for its data
   source input. The `${DS_PROMETHEUS}` placeholder makes the export portable.
5. Use **Last 5 minutes** and **5s** refresh for a live demonstration.

To inspect the existing Grafana container's gateway:

```bash
docker inspect grafana \
  --format '{{range .NetworkSettings.Networks}}{{printf "http://%s:9090\n" .Gateway}}{{end}}'
```

Open the [live dashboard](http://localhost:3000/d/da3408-captioning?from=now-5m&to=now&refresh=5s).
Its five sections are Health overview, Throughput and latency, Workload and
client activity, System resources and bulk correlation, and Errors and alerting.

## Application use and limits

- Select **Single image** or **Bulk ZIP**, choose a file and click **Generate captions**.
- Supported images: JPG, JPEG, PNG, WEBP and BMP; ZIP extension checks are
  case-insensitive. `.jepg` is not a supported extension.
- Limits: 100 MiB per upload, 12 MiB per image, 64 selected ZIP images,
  200 MiB of declared uncompressed image data and 16 million pixels per image.
- ZIP entries are read without extracting them. Directories, hidden files,
  macOS metadata and unsupported extensions are skipped. A ZIP with no
  supported images is rejected.
- Corrupt images produce error rows; a batch can contain both successful
  and failed images. Results can be downloaded as `captions.csv` without
  running inference again.

## Interpreting the metrics

- One ZIP submission is one request. Each selected image is a separate
  processing attempt. Repeated submissions count again.
- Counters reset when their application or receiver process restarts.
  Previously scraped Prometheus history is retained.
- Active requests, pending images and active generation are gauges.
- Inference latency measures model generation; complete upload latency also
  includes validation, waiting and any cold model load.
- Summary `_sum` and `_count` represent accumulated observations. Histogram
  p95 is an estimate based on bucket boundaries. One-minute latency queries
  can be undefined during idle periods, so gaps are expected.
- Client requests are cumulative counts since application startup, grouped
  by session, mode and IP classification. The first 32 session IDs are kept;
  subsequent IDs share `overflow`. The IP class can be `unknown`.
- Single/bulk CPU records process CPU time during serialized model calls,
  grouped by inference mode. The dashboard plots its one-minute rate as a
  percentage of guest logical CPU capacity. Other application threads can
  contribute during those intervals. This is separate from total guest CPU.
- Application RSS includes model tensors; guest RAM, RSS and model tensor
  storage overlap and should not be added together.
- A successful inference means the model returned a non-empty caption;
  this is not a semantic caption-accuracy metric.

The current expressions are stored in `grafana_dashboard.json`; query prompts
are retained in the YAML comments rather than duplicated in this README.

## Alerting

The five rules cover application scrape failure, high guest CPU, high guest
memory, low root-filesystem space and a high image-error ratio. Alertmanager
sends notifications to `http://127.0.0.1:18001/alerts`, including resolved events.
The receiver stores them in `runtime/alerts.jsonl` and exposes delivery counters.

To demonstrate a real outage, stop only Streamlit with Ctrl+C in Terminal 1.
After a failed scrape and the configured 20-second pending period, the
application-down rule fires; notification grouping adds further delivery
time. Restart Terminal 1 and reopen the application page. Successful scraping
resolves the alert, and the independent receiver records the recovery.

For startup problems, inspect the relevant `runtime/*.log` file and
http://localhost:9090/targets. A port-in-use error usually means an earlier
service instance is still running. Prometheus target health and model
readiness are separate checks.

