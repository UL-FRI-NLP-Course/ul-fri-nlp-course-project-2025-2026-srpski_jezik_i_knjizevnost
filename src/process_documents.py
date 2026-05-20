import pymupdf4llm 
import pymupdf

def _rect_overlap_area(a: pymupdf.Rect, b: pymupdf.Rect) -> float:
    inter = a & b
    if inter.is_empty:
        return 0.0
    return float(inter.width * inter.height)


def _is_mostly_inside(inner: pymupdf.Rect, outer: pymupdf.Rect, threshold: float = 0.6) -> bool:
    overlap = _rect_overlap_area(inner, outer)
    area = float(inner.width * inner.height)
    if area <= 0:
        return False
    return (overlap / area) >= threshold


def _to_markdown_table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    col_count = max(len(r) for r in rows)

    def norm_cell(v: str) -> str:
        if v is None:
            return ""
        return str(v).replace("\n", " ").replace("|", "\\|").strip()

    normalized = []
    for row in rows:
        padded = [norm_cell(cell) for cell in row] + [""] * (col_count - len(row))
        normalized.append(padded)

    header = normalized[0]
    separators = ["---"] * col_count
    body_rows = normalized[1:]

    md_lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(separators) + " |",
    ]
    for row in body_rows:
        md_lines.append("| " + " | ".join(row) + " |")
    return "\n".join(md_lines)


def document_to_markdown(input_path: Path, output_path: Path, name: str) -> None:
    doc = pymupdf.open(input_path)
    md_content = pymupdf4llm.to_markdown(doc)
    with open(output_path / name, "w", encoding="utf-8") as out:
        out.write(md_content)
    doc.close()