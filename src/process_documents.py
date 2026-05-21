"""
PDF → Markdown batch converter (memory-optimised)
Uses Docling for layout + table structure recognition.

Requirements:
    pip install docling

Usage:
    python pdf_to_markdown.py --input-dir ./pdfs --output-dir ./markdown
"""

import argparse
import gc
import sys
import time
from pathlib import Path

try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        TableFormerMode,
    )
except ImportError:
    print(
        "Docling is not installed.\n"
        "Install it with:  pip install docling\n"
    )
    sys.exit(1)


def build_converter(*, accurate_tables: bool = False) -> DocumentConverter:
    """
    Minimal memory footprint converter.
    - No OCR (text-based PDFs only)
    - No images or figures
    - Low render scale
    - FAST table mode by default (use accurate_tables=True only if needed)
    """
    pipeline_options = PdfPipelineOptions()

    # Tables only — no images, no OCR
    pipeline_options.do_table_structure = True
    accurate_tables = True
    pipeline_options.table_structure_options.mode = (
        TableFormerMode.ACCURATE if accurate_tables else TableFormerMode.FAST
    )
    pipeline_options.table_structure_options.do_cell_matching = True

    pipeline_options.do_ocr = False                  # biggest memory saving
    pipeline_options.generate_picture_images = False  # no figure images
    pipeline_options.images_scale = 0.5              # half resolution = 4x less image memory

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )


def convert_pdf(pdf_path: Path, output_dir: Path, converter: DocumentConverter) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / (pdf_path.stem + ".md")

    print(f"  Converting: {pdf_path.name}", end="", flush=True)
    t0 = time.monotonic()

    result = converter.convert(str(pdf_path))

    # image_placeholder="" suppresses any <!-- image --> / figure stubs in output
    markdown = result.document.export_to_markdown(image_placeholder="")

    front_matter = (
        f"---\n"
        f"source: {pdf_path.name}\n"
        f"---\n\n"
    )
    out_path.write_text(front_matter + markdown, encoding="utf-8")

    elapsed = time.monotonic() - t0
    print(f" → {out_path.name}  ({elapsed:.1f}s)")

    # Explicitly free the result object before moving to the next file
    del result
    gc.collect()

    return out_path


def batch_convert(pdf_paths: list[Path], output_dir: Path, *, accurate_tables: bool, skip_existing: bool):
    if not pdf_paths:
        print("No PDF files found.")
        return

    print(f"\nBuilding converter (accurate_tables={accurate_tables}, ocr=False)...")
    converter = build_converter(accurate_tables=accurate_tables)

    ok, failed = [], []

    print(f"\nProcessing {len(pdf_paths)} file(s) → {output_dir}\n")

    for i, pdf_path in enumerate(pdf_paths, 1):
        print(f"[{i}/{len(pdf_paths)}]", end=" ")

        out_path = output_dir / (pdf_path.stem + ".md")
        if skip_existing and out_path.exists():
            print(f"  Skipping (already exists): {out_path.name}")
            ok.append(out_path)
            continue

        try:
            convert_pdf(pdf_path, output_dir, converter)
            ok.append(pdf_path)
        except Exception as exc:
            print(f" ✗ FAILED — {exc}")
            failed.append(pdf_path)
            gc.collect()  # free memory even on failure

    print(f"\n{'─'*50}")
    print(f"  Done.  {len(ok)} succeeded,  {len(failed)} failed.")
    if failed:
        print("\n  Failed files:")
        for f in failed:
            print(f"    • {f}")
    print(f"{'─'*50}\n")


def main():
    parser = argparse.ArgumentParser(description="Convert PDFs to Markdown using Docling.")
    parser.add_argument("files", nargs="*", type=Path, help="PDF file(s) to convert.")
    parser.add_argument("--input-dir",  "-i", type=Path, default=None,  help="Directory of PDF files.")
    parser.add_argument("--output-dir", "-o", type=Path, default=Path("./markdown_output"), help="Output directory (default: ./markdown_output).")
    parser.add_argument("--accurate",   action="store_true", help="Use ACCURATE table mode (better quality, more memory).")
    parser.add_argument("--skip-existing", action="store_true", help="Skip files already converted.")
    args = parser.parse_args()

    # Collect PDFs
    paths: list[Path] = []

    for p in args.files:
        if p.exists() and p.suffix.lower() == ".pdf":
            paths.append(p.resolve())

    if args.input_dir:
        if not args.input_dir.is_dir():
            print(f"Error: {args.input_dir} is not a directory.")
            sys.exit(1)
        paths.extend(sorted(p.resolve() for p in args.input_dir.glob("*.pdf")))

    # Deduplicate
    paths = list(dict.fromkeys(paths))

    if not paths:
        print("No PDF files to process.")
        sys.exit(0)

    batch_convert(paths, args.output_dir, accurate_tables=args.accurate, skip_existing=args.skip_existing)


if __name__ == "__main__":
    main()