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
        text = " ".join(text.split())
        lines = text.split("\n")
        clean_lines = [line.strip() for line in lines if line.strip()]
        text = "\n".join(clean_lines)

        return text
