"""DA3408 CPU image captioning application."""

import os

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration


MODEL_ID = "Salesforce/blip-image-captioning-base"


class CpuCaptioner:
    def __init__(self):
        # Limit CPU threads so the VM can also run the monitoring services.
        requested_threads = int(
            os.environ.get("CAPTION_CPU_THREADS", "2")
        )
        threads = max(
            1,
            min(requested_threads, os.cpu_count() or 1),
        )
        torch.set_num_threads(threads)

        self.processor = BlipProcessor.from_pretrained(
            MODEL_ID,
            use_fast=False,
        )

        # This model repository provides PyTorch .bin weights.
        self.model = BlipForConditionalGeneration.from_pretrained(
            MODEL_ID,
            use_safetensors=False,
        )

        # Explicitly use CPU and disable training behaviour.
        self.model = self.model.to("cpu").eval()

        # Tensor storage only; total application memory is measured separately.
        self.parameter_bytes = sum(
            parameter.numel() * parameter.element_size()
            for parameter in self.model.parameters()
        )
        self.parameter_bytes += sum(
            buffer.numel() * buffer.element_size()
            for buffer in self.model.buffers()
        )

    def caption(self, image):
        image = image.convert("RGB")

        inputs = self.processor(
            images=image,
            return_tensors="pt",
        )

        # Disable gradient tracking to reduce inference memory and overhead.
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=40,
                num_beams=1,
                do_sample=False,
            )

        caption = self.processor.decode(
            output[0],
            skip_special_tokens=True,
        ).strip()

        if not caption:
            raise RuntimeError("The model returned an empty caption.")

        return caption


# Image validation and processing
import io
import threading
import time
import warnings
import zipfile
from pathlib import PurePosixPath

from PIL import Image, ImageOps



MAX_UPLOAD = 100 * 1024 * 1024
MAX_IMAGE_BYTES = 12 * 1024 * 1024
MAX_UNCOMPRESSED = 200 * 1024 * 1024
MAX_IMAGES = 64
MAX_PIXELS = 16_000_000

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class UploadError(ValueError):
    pass


def decode_image(data):
    if len(data) > MAX_IMAGE_BYTES:
        raise UploadError("Image exceeds 12 MiB.")

    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)

        with Image.open(io.BytesIO(data)) as source:
            pixels = source.width * source.height

            if pixels > MAX_PIXELS:
                raise UploadError("Image exceeds 16 million pixels.")

            # Correct camera orientation and use three colour channels.
            image = ImageOps.exif_transpose(source).convert("RGB")

            # Bound intermediate image size before model preprocessing.
            image.thumbnail((1024, 1024))

    return image, pixels


def zip_entries(archive):
    entries = []

    for info in archive.infolist():
        path = PurePosixPath(info.filename)

        if info.is_dir():
            continue

        # Ignore hidden files and macOS metadata.
        if any(
            part.startswith(".") or part == "__MACOSX"
            for part in path.parts
        ):
            continue

        if path.suffix.lower() not in EXTENSIONS:
            continue

        if info.flag_bits & 1:
            raise UploadError("Encrypted ZIP entries are not supported.")

        if info.file_size > MAX_IMAGE_BYTES:
            raise UploadError(
                f"{path.name}: image exceeds 12 MiB."
            )

        entries.append(info)

        if len(entries) > MAX_IMAGES:
            raise UploadError(
                "ZIP contains more than 64 images. "
                "Split it into smaller batches."
            )

    if not entries:
        raise UploadError(
            "ZIP contains no JPG, PNG, WEBP or BMP images."
        )

    if sum(item.file_size for item in entries) > MAX_UNCOMPRESSED:
        raise UploadError(
            "Selected ZIP contents exceed 200 MiB uncompressed."
        )

    return entries


class CaptionService:
    def __init__(self, metrics, factory=CpuCaptioner):
        self.metrics = metrics
        self.factory = factory
        self.engine = None

        self.load_lock = threading.Lock()
        self.inference_lock = threading.Lock()

    def load(self):
        # Multiple browser sessions share one model instance.
        with self.load_lock:
            if self.engine is None:
                try:
                    engine = self.factory()
                    self.metrics.parameters.set(engine.parameter_bytes)
                    self.engine = engine
                    self.metrics.ready.set(1)
                except Exception:
                    self.metrics.errors.labels("model_load").inc()
                    self.metrics.ready.set(0)
                    raise

        return self.engine

    def _one(self, data, name, mode):
        m = self.metrics

        try:
            image, pixels = decode_image(data)
            m.pixels.labels(mode).observe(pixels)
        except Exception as exc:
            m.errors.labels("decode").inc()
            m.images.labels(mode, "error").inc()

            return {
                "file": name,
                "status": "error",
                "caption": "",
                "seconds": 0.0,
                "error": str(exc),
            }

        waiting = time.perf_counter()

        # Serialize model calls across browser sessions.
        with self.inference_lock:
            m.queue_time.labels(mode).observe(
                time.perf_counter() - waiting
            )

            started = time.perf_counter()
            m.inference_active.inc()

            cpu_started = time.process_time()

            try:
                caption = self.engine.caption(image)

                if not caption:
                    raise RuntimeError("Model returned an empty caption.")

                elapsed = time.perf_counter() - started

                m.images.labels(mode, "success").inc()
                m.last_success.set(time.time())

                return {
                    "file": name,
                    "status": "success",
                    "caption": caption,
                    "seconds": round(elapsed, 3),
                    "error": "",
                }

            except Exception as exc:
                m.images.labels(mode, "error").inc()
                m.errors.labels("inference").inc()

                return {
                    "file": name,
                    "status": "error",
                    "caption": "",
                    "seconds": round(
                        time.perf_counter() - started, 3
                    ),
                    "error": str(exc),
                }

            finally:
                m.inference_cpu.labels(mode).inc(
                    max(0.0, time.process_time() - cpu_started)
                )
                m.inference_time.labels(mode).observe(
                    time.perf_counter() - started
                )
                m.inference_active.dec()

    def process(
        self,
        data,
        name,
        mode,
        session_id,
        address=None,
        progress=None,
    ):
        if mode not in ("single", "bulk"):
            raise ValueError("Unknown processing mode.")

        m = self.metrics

        m.track_client(session_id, address, mode)
        m.upload_bytes.labels(mode).inc(len(data))
        m.active.labels(mode).inc()

        started = time.perf_counter()
        outcome = "error"
        remaining = 0
        archive = None

        try:
            if len(data) > MAX_UPLOAD:
                raise UploadError("Upload exceeds 100 MiB.")

            if mode == "bulk":
                try:
                    archive = zipfile.ZipFile(io.BytesIO(data))
                    entries = zip_entries(archive)
                except (zipfile.BadZipFile, EOFError) as exc:
                    raise UploadError(
                        "Invalid or incomplete ZIP file."
                    ) from exc

                m.batch_size.observe(len(entries))

            else:
                if len(data) > MAX_IMAGE_BYTES:
                    raise UploadError(
                        "Single image exceeds 12 MiB."
                    )

                entries = [name]

            self.load()

            remaining = len(entries)
            m.pending.labels(mode).inc(remaining)

            rows = []

            for index, entry in enumerate(entries):
                item_name = (
                    entry.filename if mode == "bulk" else name
                )

                try:
                    if mode == "bulk":
                        # Read bounded data directly; never extract to disk.
                        with archive.open(entry) as stream:
                            image_data = stream.read(
                                MAX_IMAGE_BYTES + 1
                            )
                    else:
                        image_data = data

                    row = self._one(
                        image_data, item_name, mode
                    )

                except Exception as exc:
                    m.images.labels(mode, "error").inc()
                    m.errors.labels("decode").inc()

                    row = {
                        "file": item_name,
                        "status": "error",
                        "caption": "",
                        "seconds": 0.0,
                        "error": str(exc),
                    }

                rows.append(row)

                m.pending.labels(mode).dec()
                remaining -= 1

                if progress is not None:
                    progress(index + 1, len(entries))

            successes = sum(
                row["status"] == "success"
                for row in rows
            )

            if successes == len(rows):
                outcome = "success"
            elif successes:
                outcome = "partial"
            else:
                outcome = "error"

            return rows

        except UploadError:
            m.errors.labels("validation").inc()
            m.rejected.labels(mode, "validation").inc()
            raise

        finally:
            if archive is not None:
                archive.close()

            if remaining:
                m.pending.labels(mode).dec(remaining)

            m.active.labels(mode).dec()

            elapsed = time.perf_counter() - started

            m.request_time.labels(mode).observe(elapsed)
            m.request_summary.labels(mode).observe(elapsed)
            m.requests.labels(mode, outcome).inc()


from metrics import Metrics
import csv
import io
import uuid
import streamlit as st


def main():
    st.set_page_config(
        page_title="DA3408 Image Captioning",
        page_icon="📷",
        layout="wide",
    )


    @st.cache_resource
    def get_service():
        # Create one registry and metrics server for the app process.
        # Reuse them across browser sessions and Streamlit reruns.
        metrics = Metrics()
        metrics.start_server(port=18000)

        return CaptionService(metrics)


    try:
        service = get_service()
    except OSError as exc:
        st.error(
            f"Could not start the metrics endpoint: {exc}. "
            "Check whether another application is using port 18000."
        )
        st.stop()


    if "client_session" not in st.session_state:
        st.session_state.client_session = uuid.uuid4().hex[:12]


    st.title("CPU Image Captioning")
    st.caption(
        "DA3408 Assignment 4 | Single images and ZIP batches "
        "| Prometheus instrumentation"
    )


    with st.sidebar:
        st.subheader("Runtime")
        st.write("Device: CPU")
        st.caption(f"Model: {MODEL_ID}")
        st.caption(
            f"Browser session: {st.session_state.client_session}"
        )

        if st.button("Load model / warm up"):
            try:
                with st.spinner("Loading the captioning model..."):
                    service.load()

                st.success("Model loaded successfully.")

            except Exception as exc:
                st.error(f"Model loading failed: {exc}")

        if service.engine is not None:
            st.success("Model ready")
        else:
            st.info("Model has not been loaded yet.")

        st.caption(
            "ZIP images are processed sequentially. "
            "Browser sessions share one model."
        )

        st.markdown(
            "[Metrics](http://localhost:18000/metrics)"
            " · [Grafana](http://localhost:3000)"
        )


    choice = st.radio(
        "Processing mode",
        ["Single image", "Bulk ZIP"],
        horizontal=True,
    )

    mode = "single" if choice == "Single image" else "bulk"

    if mode == "single":
        st.caption(
            "Supported: JPG, PNG, WEBP and BMP. "
            "Maximum: 12 MiB and 16 million pixels."
        )
    else:
        st.caption(
            "Maximum ZIP: 100 MiB, 64 images and "
            "200 MiB of uncompressed image data."
        )


    with st.form("caption_form"):
        upload = st.file_uploader(
            "Choose an image" if mode == "single"
            else "Choose a ZIP containing images",
            type=["jpg", "jpeg", "png", "webp", "bmp"]
            if mode == "single" else ["zip"],
            key=f"upload_{mode}",
        )

        submitted = st.form_submit_button(
            "Generate captions",
            type="primary",
        )


    if submitted:
        if upload is None:
            st.warning("Choose a file first.")

        else:
            st.session_state.pop("results", None)

            progress = st.progress(
                0,
                text="Preparing the request...",
            )

            def update_progress(completed, total):
                progress.progress(
                    completed / total,
                    text=f"Processed {completed}/{total} images",
                )

            try:
                with st.spinner("Generating captions on CPU..."):
                    rows = service.process(
                        data=upload.getvalue(),
                        name=upload.name,
                        mode=mode,
                        session_id=st.session_state.client_session,
                        address=st.context.ip_address,
                        progress=update_progress,
                    )

                st.session_state.results = rows
                st.session_state.results_mode = mode

            except UploadError as exc:
                st.error(str(exc))

            except Exception as exc:
                st.error(f"Processing failed: {exc}")

            finally:
                progress.empty()


    rows = st.session_state.get("results")

    if rows:
        st.subheader("Most recent results")
        st.caption(
            f"Upload mode: {st.session_state.results_mode}. "
            "Times below measure model generation per image."
        )

        successes = sum(
            row["status"] == "success"
            for row in rows
        )

        first, second, third = st.columns(3)

        first.metric("Images attempted", len(rows))
        second.metric("Captions generated", successes)
        third.metric("Image failures", len(rows) - successes)

        if len(rows) == 1 and rows[0]["status"] == "success":
            st.write("**Caption:**", rows[0]["caption"])

        st.dataframe(
            rows,
            hide_index=True,
            width="stretch",
        )

        csv_buffer = io.StringIO()

        writer = csv.DictWriter(
            csv_buffer,
            fieldnames=[
                "file",
                "status",
                "caption",
                "seconds",
                "error",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

        st.download_button(
            "Download captions CSV",
            data=csv_buffer.getvalue().encode("utf-8"),
            file_name="captions.csv",
            mime="text/csv",
        )

        st.caption(
            "Results remain available across page interactions. "
            "Downloading the CSV does not run inference again."
        )


if __name__ == "__main__":
    main()
