from pathlib import Path
import fitz  # PyMuPDF

def pdf_to_png(pdf_path, output_dir, dpi=300):
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)

    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        out_file = output_dir / f"{pdf_path.stem}_page_{i+1}.png"
        pix.save(out_file)
        print(f"Saved {out_file}")

    doc.close()

input_folder = Path(".")
output_folder = Path("png_output")

for pdf_file in input_folder.glob("*.pdf"):
    pdf_to_png(pdf_file, output_folder, dpi=300)