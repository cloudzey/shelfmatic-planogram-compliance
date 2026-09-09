"""Interactive Shelfmatic product-detection demo."""

from __future__ import annotations

import hashlib
import json
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

from src.planogram_compliance import (
    compare_planogram,
    parse_expected_facings,
)

from src.shelf_layout import build_shelf_layout


SAMPLE_GALLERY_MANIFEST = PROJECT_ROOT / "data" / "samples" / "gallery.json"
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


@st.cache_data(show_spinner=False)
def load_sample_gallery() -> list[dict[str, str]]:
    """Load the attributed demo images and verify their local paths."""
    with SAMPLE_GALLERY_MANIFEST.open(encoding="utf-8") as manifest_file:
        gallery = json.load(manifest_file)

    if not isinstance(gallery, list) or not gallery:
        raise ValueError("Örnek görsel galerisi boş veya geçersiz.")

    required_fields = {
        "id",
        "label",
        "category",
        "path",
        "creator",
        "license",
        "license_url",
        "source_url",
    }
    seen_ids: set[str] = set()
    for sample in gallery:
        if not isinstance(sample, dict) or not required_fields.issubset(sample):
            raise ValueError("Örnek görsel galerisinde eksik alan var.")
        if sample["id"] in seen_ids:
            raise ValueError(f"Yinelenen örnek görsel kimliği: {sample['id']}")
        seen_ids.add(sample["id"])
        if not (PROJECT_ROOT / sample["path"]).is_file():
            raise FileNotFoundError(f"Örnek görsel bulunamadı: {sample['path']}")

    return gallery


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

    raw_detections = extract_detections(result)
    shelf_layout = build_shelf_layout(raw_detections)

    detections = [
        detection
        for row in shelf_layout["rows"]
        for detection in row["detections"]
    ]

    detections.extend(shelf_layout["unassigned_detections"])

    confidences = [item["confidence"] for item in detections]

    shelf_layout_report = {
        "row_count": shelf_layout["row_count"],
        "row_tolerance": shelf_layout["row_tolerance"],
        "min_detections_per_row": shelf_layout["min_detections_per_row"],
        "unassigned_count": shelf_layout["unassigned_count"],
        "rows": [
            {
                "row_index": row["row_index"],
                "center_y_norm": row["center_y_norm"],
                "top_y_norm": row["top_y_norm"],
                "bottom_y_norm": row["bottom_y_norm"],
                "detection_count": row["detection_count"],
                "slot_ids": [
                    detection["slot_id"]
                    for detection in row["detections"]
                ],
            }
            for row in shelf_layout["rows"]
        ],
    }

    summary = {
        "detection_count": len(detections),
        "shelf_row_count": shelf_layout["row_count"],
        "review_required_count": shelf_layout["unassigned_count"],
        "average_confidence": (
            None
            if not confidences
            else round(sum(confidences) / len(confidences), 6)
        ),
        "minimum_confidence": (
            None if not confidences else min(confidences)
        ),
        "maximum_confidence": (
            None if not confidences else max(confidences)
        ),
        "elapsed_ms": round(elapsed_ms, 2),
    }

    report = {
        "schema_version": 1,
        "model": "YOLO11s",
        "model_sha256": model_sha256,
        "settings": inference,
        "image": {
            "filename": filename,
            "width": image.width,
            "height": image.height,
        },
        "summary": summary,
        "shelf_layout": shelf_layout_report,
        "detections": detections,
    }

    return {
        "annotated_image": render_result(
            result,
            show_details=show_details,
        ),
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
    try:
        sample_gallery = load_sample_gallery()
    except (OSError, ValueError, json.JSONDecodeError) as error:
        st.error(f"Örnek görsel galerisi yüklenemedi: {error}")
        st.stop()

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
        selected_sample = None
        if source_mode == "Örnek raf fotoğrafı":
            sample_by_id = {sample["id"]: sample for sample in sample_gallery}
            selected_sample_id = st.selectbox(
                "Örnek görsel",
                options=list(sample_by_id),
                format_func=lambda sample_id: (
                    f"{sample_by_id[sample_id]['label']} · "
                    f"{sample_by_id[sample_id]['category']}"
                ),
            )
            selected_sample = sample_by_id[selected_sample_id]
            st.caption(f"{len(sample_gallery)} lisanslı raf örneği")
        show_details = st.checkbox(
            "Kutularda sınıf ve güven skorunu göster",
            value=False,
            help="Yoğun raflarda daha okunaklı olması için varsayılan olarak kapalıdır.",
        )
        st.subheader("Hedef planogram")
        expected_facings_text = st.text_input(
            "Satır başına beklenen ürün sayısı",
            value="",
            placeholder="Örn. 15,16,17",
            help=(
                "Üst raftan alt rafa doğru beklenen "
                "ürün yüzü sayılarını virgülle ayır."
            ),
        )
        st.divider()
        st.caption(f"Checkpoint: `{model_sha256[:12]}…`")
        st.caption("Desteklenen dosyalar: JPG, JPEG, PNG · En fazla 20 MB")

    uploaded_file = None
    image = None
    filename = ""
    if source_mode == "Örnek raf fotoğrafı":
        sample_path = PROJECT_ROOT / selected_sample["path"]
        image = Image.open(sample_path).convert("RGB")
        filename = sample_path.name
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
        if selected_sample is not None:
            st.caption(
                f"Fotoğraf: [{selected_sample['creator']}]"
                f"({selected_sample['source_url']}) · "
                f"Lisans: [{selected_sample['license']}]"
                f"({selected_sample['license_url']})"
            )
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
    result_metrics =st.columns(5)

    result_metrics[0].metric(
        "Tespit edilen ürün",
        summary["detection_count"],
    )
    result_metrics[1].metric(
        "Raf satırı",
        summary["shelf_row_count"],
    )
    result_metrics[2].metric(
        "Ortalama güven",
        "—"
        if summary["average_confidence"] is None
        else f"{summary['average_confidence'] * 100:.1f}%",
    )
    result_metrics[3].metric(
        "En yüksek güven",
        "—"
        if summary["maximum_confidence"] is None
        else f"{summary['maximum_confidence'] * 100:.1f}%",
    )
    result_metrics[4].metric(
        "Toplam işlem", 
        f"{summary['elapsed_ms']:.0f} ms",
        )

    if summary["review_required_count"] > 0:
        st.info(
        f"{summary['review_required_count']} tespit raf satırlarına "
        "güvenli biçimde atanamadığı için insan kontrolüne ayrıldı.",
        icon="ℹ️",
        )

    compliance = None

    if expected_facings_text.strip():
        try:
            expected_facings = parse_expected_facings(
                expected_facings_text
            )
            compliance = compare_planogram(
                report["shelf_layout"],
                expected_facings,
            )
        except ValueError as error:
            st.error(f"Hedef planogram geçersiz: {error}")

    export_report = dict(report)

    if compliance is not None:
        export_report["planogram_compliance"] = compliance

        st.markdown("##### Planogram uyumluluğu")

        compliance_metrics = st.columns(4)
        compliance_metrics[0].metric(
            "Uyumluluk",
            f"{compliance['compliance_percent']:.1f}%",
        )
        compliance_metrics[1].metric(
            "Beklenen ürün yüzü",
            compliance["expected_facing_count"],
        )
        compliance_metrics[2].metric(
            "Eksik ürün yüzü",
            compliance["missing_facing_count"],
        )
        compliance_metrics[3].metric(
            "Fazla ürün yüzü",
            compliance["extra_facing_count"],
        )

        if compliance["is_compliant"]:
            st.success(
                "Raf yapısı hedef planogramla uyumlu.",
                icon="✅",
            )
        else:
            st.warning(
                "Raf yapısında hedef planograma göre "
                "farklılıklar bulundu.",
                icon="⚠️",
            )

        status_labels = {
            "compliant": "Uyumlu",
            "missing_facings": "Eksik ürün",
            "extra_facings": "Fazla ürün",
            "missing_row": "Eksik raf",
            "unexpected_row": "Beklenmeyen raf",
        }

        comparison_rows = [
            {
                "Raf": row["row_index"],
                "Beklenen": row["expected_facings"],
                "Tespit edilen": row["detected_facings"],
                "Fark": row["difference"],
                "Durum": status_labels[row["status"]],
            }
            for row in compliance["rows"]
        ]

        st.dataframe(
            comparison_rows,
            width="stretch",
            hide_index=True,
        )

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
        data=report_to_json_bytes(export_report),
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

        st.info(
        "Planogram karşılaştırması raf satırı ve ürün yüzü sayıları "
        "üzerinden yapısal olarak gerçekleştirilir. Model SKU veya "
        "marka kimliği tahmin etmez.",
        icon="ℹ️",
    )


if __name__ == "__main__":
    main()
