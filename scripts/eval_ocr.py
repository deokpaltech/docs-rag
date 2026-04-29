"""OCR 품질 평가 스크립트.

현재 OCR 파이프라인(PaddleOCR PP-StructureV3 + 6단계 입구 필터)의 품질을 필터
통과율 / confidence 분포 / 샘플 리뷰로 측정. 필터 임계값(settings.py
OCR_*) 튜닝의 근거 데이터를 제공.

사용법:
    uv run python scripts/eval_ocr.py
    uv run python scripts/eval_ocr.py --doc "자녀"
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "output" / "raw"
EVAL_DIR = PROJECT_ROOT / "data" / "eval" / "ocr"

# 필터 로직 (src/v1/utils/ocr.py와 동일)
_SPECIAL_ONLY_RE = re.compile(r'^[\s\-–—_·…=|/\\:;,.!?@#$%^&*(){}[\]<>]+$')
_MATH_NOISE_RE = re.compile(r'^[x_{}()\-\d\s\^]+$')
_HANGUL_RE = re.compile(r'[가-힣]')
_ALPHA_RE = re.compile(r'[a-zA-Z]')


def _is_meaningful(text: str) -> tuple[bool, str]:
    stripped = text.strip()
    if not stripped:
        return False, "empty"
    if len(stripped) < 10:
        return False, "too_short"
    if _SPECIAL_ONLY_RE.match(stripped):
        return False, "special_only"
    if _MATH_NOISE_RE.match(stripped):
        return False, "math_noise"
    hangul = len(_HANGUL_RE.findall(stripped))
    alpha = len(_ALPHA_RE.findall(stripped))
    total = len(stripped.replace(" ", "").replace("\n", ""))
    if total > 0 and (hangul + alpha) / total < 0.3:
        return False, "low_lang_ratio"
    return True, "passed"


def _natural_key(s: str):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]


def find_all_doc_dirs() -> list[Path]:
    return sorted([d for d in RAW_DIR.iterdir() if d.is_dir() and d.name.endswith("_images")])


def evaluate_doc(images_dir: Path) -> dict:
    """한 문서의 OCR 품질 평가."""
    doc_name = images_dir.name.replace("_images", "")
    image_files = sorted(
        [f for f in images_dir.iterdir() if f.suffix == ".png" and "_ocr" not in f.name and "_prep" not in f.name],
        key=lambda x: _natural_key(x.name),
    )

    total = len(image_files)
    ocr_count = 0
    passed = 0
    dropped = 0
    drop_reasons = {}
    all_scores = []
    low_conf_samples = []  # confidence 낮은 통과 샘플

    for img_path in image_files:
        ocr_json = images_dir / f"{img_path.stem}_ocr.json"
        if not ocr_json.exists():
            continue
        ocr_count += 1

        with open(ocr_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        scores = data.get("rec_scores", [])
        all_scores.extend(scores)

        blocks = data.get("parsing_blocks", [])
        combined = "\n".join(b.get("content", "") for b in blocks)
        ok, reason = _is_meaningful(combined)

        if ok:
            passed += 1
            if scores:
                avg = sum(scores) / len(scores)
                low_conf_samples.append((avg, img_path.name, combined[:80]))
        else:
            dropped += 1
            drop_reasons[reason] = drop_reasons.get(reason, 0) + 1

    # confidence 분포
    conf_dist = {}
    for lo, hi, label in [(0.9, 1.01, "0.9+"), (0.8, 0.9, "0.8-0.9"), (0.5, 0.8, "0.5-0.8"), (0.0, 0.5, "<0.5")]:
        conf_dist[label] = sum(1 for s in all_scores if lo <= s < hi)

    # 낮은 confidence 통과 샘플 (상위 5개)
    low_conf_samples.sort()
    worst = [(name, f"{conf:.2f}", text) for conf, name, text in low_conf_samples[:5]]

    return {
        "doc_name": doc_name,
        "total_images": total,
        "ocr_processed": ocr_count,
        "skipped": total - ocr_count,
        "filter_passed": passed,
        "filter_dropped": dropped,
        "drop_reasons": drop_reasons,
        "avg_confidence": sum(all_scores) / len(all_scores) if all_scores else 0,
        "confidence_dist": conf_dist,
        "worst_samples": worst,
    }


def print_results(results: list[dict]):
    """결과 출력."""
    print(f"\n{'='*70}")
    print(f"  OCR 품질 평가 결과 ({len(results)}개 문서)")
    print(f"{'='*70}\n")

    total_images = 0
    total_ocr = 0
    total_passed = 0
    total_dropped = 0

    for r in results:
        total_images += r["total_images"]
        total_ocr += r["ocr_processed"]
        total_passed += r["filter_passed"]
        total_dropped += r["filter_dropped"]

        pass_rate = r["filter_passed"] / r["ocr_processed"] * 100 if r["ocr_processed"] > 0 else 0

        print(f"  [{r['doc_name'][:60]}]")
        print(f"    이미지: {r['total_images']}개 → OCR: {r['ocr_processed']}개 → 필터통과: {r['filter_passed']}개 ({pass_rate:.0f}%)")
        print(f"    평균 confidence: {r['avg_confidence']:.2f}")

        # confidence 분포 바 차트
        if r["confidence_dist"]:
            total_s = sum(r["confidence_dist"].values())
            for label, count in r["confidence_dist"].items():
                pct = count / total_s * 100 if total_s > 0 else 0
                bar = "█" * int(pct / 3)
                print(f"      {label:>7s}: {count:3d} ({pct:4.1f}%) {bar}")

        # 제거 사유
        if r["drop_reasons"]:
            reasons_str = ", ".join(f"{k}:{v}" for k, v in sorted(r["drop_reasons"].items(), key=lambda x: -x[1]))
            print(f"    제거 사유: {reasons_str}")

        # 가장 confidence 낮은 통과 샘플
        if r["worst_samples"]:
            print(f"    주의 샘플 (confidence 낮은 통과):")
            for name, conf, text in r["worst_samples"][:3]:
                print(f"      {name} (conf={conf}): {text!r}")
        print()

    # 전체 요약
    overall_pass = total_passed / total_ocr * 100 if total_ocr > 0 else 0
    print(f"{'─'*70}")
    print(f"  전체 요약: {total_images}개 이미지 → {total_ocr}개 OCR → {total_passed}개 통과 ({overall_pass:.0f}%)")
    print(f"{'─'*70}\n")


def save_results(results: list[dict]):
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EVAL_DIR / "ocr_eval_result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  결과 저장: {out_path}\n")


def main():
    parser = argparse.ArgumentParser(description="OCR 품질 평가")
    parser.add_argument("--doc", default=None, help="문서 키워드 (미지정 시 전체)")
    args = parser.parse_args()

    doc_dirs = find_all_doc_dirs()
    if args.doc:
        doc_dirs = [d for d in doc_dirs if args.doc in d.name]

    if not doc_dirs:
        print("평가할 문서가 없습니다.")
        sys.exit(1)

    print(f"=== OCR 품질 평가 ===")
    print(f"대상: {len(doc_dirs)}개 문서\n")

    results = []
    for d in doc_dirs:
        print(f"  평가 중: {d.name[:50]}...")
        results.append(evaluate_doc(d))

    print_results(results)
    save_results(results)


if __name__ == "__main__":
    main()
