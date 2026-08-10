"""Interactive Shelfmatic product-detection demo."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

from src.inference_utils import (
    PROJECT_ROOT,
    detections_to_csv_bytes,
    extract_detections,
    image_to_jpeg_bytes,
    load_deployment_config,
    render_result,
    report_to_json_bytes,
)


SAMPLE_IMAGE = PROJECT_ROOT / "data" / "samples" / "soft_drink_shelf.jpg"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


st.set_page_config(
    page_title="Shelfmatic Product Detection",
    page_icon="📦",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def load_model(model_path: str, model_sha256: str) -> YOLO:
    """Cache the detector while invalidating it when the checkpoint changes."""
    del model_sha256
    return YOLO(model_path, task="detect")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def analyze_image(
    model: YOLO,
    image: Image.Image,
    filename: str,
    inference: dict[str, Any],
    model_sha256: str,
    show_details: bool,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    image_bgr = np.ascontiguousarray(np.asarray(image)[..., ::-1])
    result = model.predict(
        source=image_bgr,
        imgsz=int(inference["imgsz"]),
        conf=float(inference["confidence"]),
        iou=float(inference["iou"]),
        max_det=int(inference["max_det"]),
        verbose=False,
    )[0]
    elapsed_ms = (time.perf_counter() - started_at) * 1000

    detections = extract_detections(result)
    confidences = [item["confidence"] for item in detections]
    summary = {
        "detection_count": len(detections),
        "average_confidence": (
            None if not confidences else round(sum(confidences) / len(confidences), 6)
        ),
        "minimum_confidence": None if not confidences else min(confidences),
        "maximum_confidence": None if not confidences else max(confidences),
        "elapsed_ms": round(elapsed_ms, 2),
    }
    report = {
        "model": "YOLO11s",
        "model_sha256": model_sha256,
        "settings": inference,
        "image": {
            "filename": filename,
            "width": image.width,
            "height": image.height,
        },
        "summary": summary,
        "detections": detections,
    }
    return {
        "annotated_image": render_result(result, show_details=show_details),
        "report": report,
        "json_bytes": report_to_json_bytes(report),
        "csv_bytes": detections_to_csv_bytes(detections),
    }


def read_uploaded_image(uploaded_file: Any) -> Image.Image:
    if uploaded_file.size > MAX_UPLOAD_BYTES:
        raise ValueError("Dosya 20 MB sınırını aşıyor.")
    try:
        return Image.open(uploaded_file).convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError("Dosya geçerli bir JPG veya PNG görüntüsü değil.") from error


def main() -> None:
    config = load_deployment_config()
    inference = config["inference"]
    model_path = PROJECT_ROOT / config["model"]["path"]
    if not model_path.is_file():
        st.error(f"Model dosyası bulunamadı: {model_path}")
        st.stop()

    model_sha256 = sha256_file(model_path)

    st.title("Shelfmatic")
    st.caption("Raf fotoğraflarında YOLO11s ile ürün yüzü tespiti")

    metric_columns = st.columns(4)
    metric_columns[0].metric("Model", "YOLO11s")
    metric_columns[1].metric("Görüntü boyutu", str(inference["imgsz"]))
    metric_columns[2].metric("Confidence", f"{inference['confidence']:.2f}")
    metric_columns[3].metric("NMS IoU", f"{inference['iou']:.2f}")

    with st.sidebar:
        st.header("Analiz ayarları")
        st.success("Final deployment ayarları etkin")
        source_mode = st.radio(
            "Görüntü kaynağı",
            ("Örnek raf fotoğrafı", "Kendi fotoğrafımı yükle"),
        )
        show_details = st.checkbox(
            "Kutularda sınıf ve güven skorunu göster",
            value=False,
            help="Yoğun raflarda daha okunaklı olması için varsayılan olarak kapalıdır.",
        )
        st.divider()
        st.caption(f"Checkpoint: `{model_sha256[:12]}…`")
        st.caption("Desteklenen dosyalar: JPG, JPEG, PNG · En fazla 20 MB")

    uploaded_file = None
    image = None
    filename = SAMPLE_IMAGE.name
    if source_mode == "Örnek raf fotoğrafı":
        image = Image.open(SAMPLE_IMAGE).convert("RGB")
    else:
        uploaded_file = st.file_uploader(
            "Analiz edilecek raf fotoğrafını seç",
            type=("jpg", "jpeg", "png"),
        )
        if uploaded_file is not None:
            try:
                image = read_uploaded_image(uploaded_file)
                filename = uploaded_file.name
            except ValueError as error:
                st.error(str(error))

    if image is None:
        st.info("Devam etmek için bir raf fotoğrafı yükle.")
        st.stop()

    preview_column, action_column = st.columns((3, 1), vertical_alignment="bottom")
    with preview_column:
        st.image(image, caption=f"Kaynak: {filename}", width="stretch")
    with action_column:
        st.markdown("##### Analize hazır")
        st.write(f"{image.width} × {image.height} px")
        run_analysis = st.button(
            "Ürünleri tespit et",
            type="primary",
            width="stretch",
        )

    request_digest = hashlib.sha256(image.tobytes())
    request_digest.update(filename.encode("utf-8"))
    request_digest.update(str(show_details).encode("ascii"))
    request_digest.update(model_sha256.encode("ascii"))
    request_digest.update(repr(sorted(inference.items())).encode("utf-8"))
    request_key = request_digest.hexdigest()
    if run_analysis:
        with st.spinner("YOLO11s rafı inceliyor…"):
            model = load_model(str(model_path), model_sha256)
            st.session_state["shelfmatic_analysis"] = {
                "request_key": request_key,
                "result": analyze_image(
                    model=model,
                    image=image,
                    filename=filename,
                    inference=inference,
                    model_sha256=model_sha256,
                    show_details=show_details,
                ),
            }

    cached_analysis = st.session_state.get("shelfmatic_analysis")
    if not cached_analysis or cached_analysis["request_key"] != request_key:
        st.stop()

    analysis = cached_analysis["result"]
    report = analysis["report"]
    summary = report["summary"]

    st.divider()
    st.subheader("Analiz sonucu")
    result_metrics = st.columns(4)
    result_metrics[0].metric("Tespit edilen ürün", summary["detection_count"])
    result_metrics[1].metric(
        "Ortalama güven",
        "—"
        if summary["average_confidence"] is None
        else f"{summary['average_confidence'] * 100:.1f}%",
    )
    result_metrics[2].metric(
        "En yüksek güven",
        "—"
        if summary["maximum_confidence"] is None
        else f"{summary['maximum_confidence'] * 100:.1f}%",
    )
    result_metrics[3].metric("Toplam işlem", f"{summary['elapsed_ms']:.0f} ms")

    original_column, result_column = st.columns(2)
    with original_column:
        st.markdown("##### Orijinal")
        st.image(image, width="stretch")
    with result_column:
        st.markdown("##### YOLO11s sonucu")
        st.image(analysis["annotated_image"], width="stretch")

    stem = Path(filename).stem
    download_columns = st.columns(3)
    download_columns[0].download_button(
        "Kutulu JPG indir",
        data=image_to_jpeg_bytes(analysis["annotated_image"]),
        file_name=f"{stem}_detected.jpg",
        mime="image/jpeg",
        width="stretch",
    )
    download_columns[1].download_button(
        "JSON raporu indir",
        data=analysis["json_bytes"],
        file_name=f"{stem}_detections.json",
        mime="application/json",
        width="stretch",
    )
    download_columns[2].download_button(
        "CSV listesini indir",
        data=analysis["csv_bytes"],
        file_name=f"{stem}_detections.csv",
        mime="text/csv",
        width="stretch",
    )

    with st.expander("Tespit ayrıntıları"):
        if report["detections"]:
            st.dataframe(report["detections"], width="stretch", hide_index=True)
        else:
            st.info("Bu görüntüde deployment eşiğini geçen ürün tespiti yok.")

    st.warning(
        "Bu PoC ürün yüzlerini tespit eder. SKU kimliği, raf sırası ve planogram "
        "uyumluluk yüzdesi bir sonraki geliştirme aşamasıdır.",
        icon="ℹ️",
    )


if __name__ == "__main__":
    main()
