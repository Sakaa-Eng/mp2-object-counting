from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Menghitung mobil pada foto parkiran.")
    parser.add_argument("--input", type=Path, default=ROOT / "input" / "parking.jpg")
    parser.add_argument("--output", type=Path, default=ROOT / "output")
    return parser.parse_args()


def parking_line_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 150), (179, 80, 255))
    edges = cv2.Canny(white, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=120, minLineLength=160, maxLineGap=30
    )

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            dx, dy = x2 - x1, y2 - y1
            angle = abs(np.degrees(np.arctan2(dy, dx)))
            angle = min(angle, 180 - angle)
            if np.hypot(dx, dy) > 150 and (angle < 8 or abs(angle - 90) < 8):
                cv2.line(mask, (x1, y1), (x2, y2), 255, 13)

    return cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)


def otsu_contours(
    image: np.ndarray, line_mask: np.ndarray
) -> tuple[dict[str, np.ndarray | float], np.ndarray]:
    lightness = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 0]
    blurred = cv2.GaussianBlur(lightness, (7, 7), 0)
    threshold, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    without_lines = cv2.bitwise_and(binary, cv2.bitwise_not(line_mask))
    opened = cv2.morphologyEx(
        without_lines,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    cleaned = cv2.morphologyEx(
        opened,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_mask = np.zeros_like(cleaned)
    image_area = cleaned.size
    for contour in contours:
        area = cv2.contourArea(contour)
        if 80 < area < image_area * 0.08:
            cv2.drawContours(contour_mask, [contour], -1, 255, thickness=-1)

    stages = {
        "lightness": lightness,
        "blurred": blurred,
        "binary": binary,
        "cleaned": cleaned,
        "threshold": float(threshold),
    }
    return stages, contour_mask


def appearance_evidence(image: np.ndarray, line_mask: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    saturation, value = hsv[:, :, 1], hsv[:, :, 2]
    lightness = lab[:, :, 0]

    background = cv2.GaussianBlur(gray, (0, 0), 55)
    local_difference = cv2.absdiff(gray, background)
    contrast = local_difference > 42
    colored = (saturation > 45) & (value > 45)
    dark_detail = (lightness < 75) & (
        (local_difference > 25) | (saturation > 35)
    )
    evidence = ((contrast | colored | dark_detail).astype(np.uint8)) * 255
    evidence = cv2.bitwise_and(evidence, cv2.bitwise_not(line_mask))
    evidence = cv2.morphologyEx(
        evidence,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    return cv2.dilate(
        evidence,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    )


def local_maxima(
    score_map: np.ndarray,
    threshold: float,
    suppress_width: int,
    suppress_height: int,
) -> list[tuple[int, int, float]]:
    working = score_map.copy()
    height, width = working.shape
    peaks = []
    while True:
        _, score, _, (x, y) = cv2.minMaxLoc(working)
        if score < threshold:
            break
        peaks.append((x, y, float(score)))
        working[
            max(0, y - suppress_height // 2) : min(height, y + suppress_height // 2),
            max(0, x - suppress_width // 2) : min(width, x + suppress_width // 2),
        ] = 0
    return peaks


def horizontal_cars(
    image: np.ndarray, evidence: np.ndarray, contour_mask: np.ndarray
) -> tuple[list[dict], np.ndarray]:
    height, width = image.shape[:2]
    car_width = max(80, round(width * 0.136))
    car_height = max(45, round(height * 0.096))

    combined = (
        0.85 * (evidence.astype(np.float32) / 255)
        + 0.15 * (contour_mask.astype(np.float32) / 255)
    )
    score_map = cv2.boxFilter(
        combined,
        -1,
        (car_width, car_height),
        normalize=True,
        borderType=cv2.BORDER_REPLICATE,
    )
    peaks = local_maxima(
        score_map,
        threshold=0.655,
        suppress_width=round(car_width * 1.43),
        suppress_height=round(car_height * 1.43),
    )

    detections = []
    for x, y, score in peaks:
        if y > height - car_height * 0.25:
            continue
        detections.append(
            {
                "box": (
                    max(0, x - car_width // 2),
                    max(0, y - car_height // 2),
                    min(width, x + car_width // 2),
                    min(height, y + car_height // 2),
                ),
                "score": score,
                "type": "horizontal",
            }
        )
    return detections, score_map


def vertical_red_car(image: np.ndarray) -> tuple[list[dict], np.ndarray]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_or(
        cv2.inRange(hsv, (0, 70, 80), (12, 255, 255)),
        cv2.inRange(hsv, (170, 70, 80), (179, 255, 255)),
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)),
        iterations=2,
    )

    detections = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = image.shape[0] * image.shape[1]
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area > image_area * 0.001 and height / max(width, 1) > 1.5:
            pad_x, pad_y = round(width * 0.08), round(height * 0.04)
            detections.append(
                {
                    "box": (
                        max(0, x - pad_x),
                        max(0, y - pad_y),
                        min(image.shape[1], x + width + pad_x),
                        min(image.shape[0], y + height + pad_y),
                    ),
                    "score": float(area),
                    "type": "vertical",
                }
            )
    return detections, mask


def draw_result(image: np.ndarray, detections: list[dict]) -> np.ndarray:
    result = image.copy()
    detections.sort(key=lambda item: (item["box"][1], item["box"][0]))
    for number, detection in enumerate(detections, start=1):
        x1, y1, x2, y2 = detection["box"]
        color = (0, 165, 255) if detection["type"] == "vertical" else (0, 255, 0)
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 4)
        cv2.putText(
            result,
            str(number),
            (x1 + 5, max(28, y1 + 28)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )

    cv2.rectangle(result, (1050, 18), (2045, 112), (25, 25, 25), -1)
    cv2.putText(
        result,
        f"Jumlah mobil: {len(detections)}",
        (1100, 84),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.6,
        (0, 0, 255),
        4,
        cv2.LINE_AA,
    )
    return result


def run(input_path: Path, output_dir: Path) -> int:
    image = cv2.imread(str(input_path))
    if image is None:
        raise FileNotFoundError(f"Gambar tidak ditemukan: {input_path}")
    steps = output_dir / "steps"
    steps.mkdir(parents=True, exist_ok=True)

    lines = parking_line_mask(image)
    otsu_stages, contour_mask = otsu_contours(image, lines)
    evidence = appearance_evidence(image, lines)
    horizontal, score_map = horizontal_cars(image, evidence, contour_mask)
    vertical, red_mask = vertical_red_car(image)
    detections = horizontal + vertical
    result = draw_result(image, detections)

    heatmap = cv2.applyColorMap(
        cv2.normalize(score_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        cv2.COLORMAP_JET,
    )
    outputs = {
        "gambar1_hasil_original.png": image,
        "gambar2_hasil_lab_lightness.png": otsu_stages["lightness"],
        "gambar3_hasil_otsu_threshold.png": otsu_stages["binary"],
        "gambar4_hasil_parking_line_mask.png": lines,
        "gambar5_hasil_morphology.png": otsu_stages["cleaned"],
        "gambar6_hasil_filtered_contours.png": contour_mask,
        "gambar7_hasil_appearance_evidence.png": evidence,
        "gambar8_hasil_density_heatmap.png": heatmap,
        "gambar9_hasil_red_car_mask.png": red_mask,
    }
    for filename, step_image in outputs.items():
        cv2.imwrite(str(steps / filename), step_image)
    cv2.imwrite(str(output_dir / "total_mobil.png"), result)
    (output_dir / "count.txt").write_text(
        f"Jumlah mobil terdeteksi: {len(detections)}\n"
        f"Horizontal: {len(horizontal)}\n"
        f"Vertikal merah: {len(vertical)}\n"
        f"Threshold Otsu: {otsu_stages['threshold']:.0f}\n",
        encoding="utf-8",
    )
    return len(detections)


def main() -> None:
    args = parse_args()
    count = run(args.input, args.output)
    print(f"Jumlah mobil terdeteksi: {count}")
    print(f"Hasil tersimpan di: {args.output / 'total_mobil.png'}")


if __name__ == "__main__":
    main()
