from PyPDF2 import PdfReader
from typing import Optional


class DocumentProcessor:
    def load_pdf(self, file_path: str) -> Optional[str]:
        all_text = ""
        try:
            pdf = PdfReader(file_path)
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    all_text += page_text

            return self._clean_text(all_text)

        except FileNotFoundError:
            print(f"File not found at {file_path}")
            return None

        except Exception as e:
            print(f"Error reading PDF : {e}")
            return None

    def _clean_text(self, text: str) -> str:
        cleaned_text = text.strip()

        return cleaned_text


if __name__ == "__main__":
    processor = DocumentProcessor()
    text = processor.load_pdf("data/sample_pdfs/test.pdf")

    if text:
        print(f"Extracted {len(text)} characters")
        print(f"First 200 words : {text[:200]}")
    else:
        print("Failed to load PDF")
