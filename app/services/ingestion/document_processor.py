# app/services/ingestion/document_processor.py
"""
Advanced document processor using Docling for intelligent chunking.
Supports multiple file formats: PDF, images, Word docs, HTML, etc.
"""

import os
from typing import List, Dict, Any
from docling.document_converter import DocumentConverter
from app.core.config import settings


class DocumentProcessor:
    """
    Process documents using Docling for structure-aware chunking.
    Better than simple text splitting - preserves document structure.
    """

    def __init__(self):
        self.converter = DocumentConverter()
        self.chunk_size = settings.KB_CHUNK_SIZE  # 800
        self.chunk_overlap = settings.KB_CHUNK_OVERLAP  # 100

    async def process_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Process a file and return structured chunks.

        Args:
            file_path: Path to the file

        Returns:
            List of dicts with keys: 'text', 'metadata'
        """
        import asyncio

        ext = os.path.splitext(file_path)[1].lower()
        if ext in {".txt", ".md"}:
            return await self._process_plain_text(file_path)

        # Convert document using Docling (runs in thread to avoid blocking)
        result = await asyncio.to_thread(
            self.converter.convert,
            file_path
        )

        # Extract text with structure
        markdown_content = result.document.export_to_markdown()

        # Parse markdown into intelligent chunks
        chunks = self._parse_markdown_to_chunks(markdown_content, file_path)

        # Fallback if markdown parsing yields too few chunks
        if len(chunks) < 2:
            chunks = self._fallback_text_extraction(result.document, file_path)

        return chunks

    def _parse_markdown_to_chunks(self, markdown: str, file_path: str) -> List[Dict[str, Any]]:
        """
        Parse markdown content into structured chunks.
        Respects document sections and headers.
        """
        chunks = []
        lines = markdown.split('\n')

        current_section = "GENERAL"
        current_chunk = ""
        page_number = 1  # Approximate

        for line in lines:
            line = line.strip()

            if not line:
                continue

            # Detect headers (markdown: # Header or ## Header)
            if line.startswith('#'):
                # Save previous chunk if substantial
                if current_chunk and len(current_chunk) > 100:
                    chunks.append({
                        'text': current_chunk.strip(),
                        'metadata': {
                            'filename': os.path.basename(file_path),
                            'section': current_section,
                            'page': page_number,
                            'type': 'section_content'
                        }
                    })
                    current_chunk = ""

                # Extract section name
                current_section = line.lstrip('#').strip()
                continue

            # Detect ALL CAPS lines as section headers
            if line.isupper() and len(line) < 100 and len(line.split()) <= 5:
                if current_chunk and len(current_chunk) > 100:
                    chunks.append({
                        'text': current_chunk.strip(),
                        'metadata': {
                            'filename': os.path.basename(file_path),
                            'section': current_section,
                            'page': page_number,
                            'type': 'section_content'
                        }
                    })
                    current_chunk = ""

                current_section = line
                continue

            # Add line to current chunk
            current_chunk += line + " "

            # Split if chunk gets too large
            if len(current_chunk) >= self.chunk_size:
                chunks.append({
                    'text': current_chunk.strip(),
                    'metadata': {
                        'filename': os.path.basename(file_path),
                        'section': current_section,
                        'page': page_number,
                        'type': 'section_content'
                    }
                })
                current_chunk = ""
                page_number += 1  # Approximate page tracking

        # Add remaining chunk
        if current_chunk and len(current_chunk) > 100:
            chunks.append({
                'text': current_chunk.strip(),
                'metadata': {
                    'filename': os.path.basename(file_path),
                    'section': current_section,
                    'page': page_number,
                    'type': 'section_content'
                }
            })

        return chunks

    def _fallback_text_extraction(self, document, file_path: str) -> List[Dict[str, Any]]:
        """
        Fallback: simple paragraph-based extraction if markdown parsing fails.
        """
        chunks = []

        try:
            full_text = document.export_to_text()
            paragraphs = [p.strip() for p in full_text.split('\n\n') if p.strip()]

            current_section = "GENERAL"
            for i, para in enumerate(paragraphs):
                # Skip very short paragraphs
                if len(para) < 100:
                    # Check if it's a section header
                    if para.isupper() and len(para) < 100:
                        current_section = para
                    continue

                chunks.append({
                    'text': para,
                    'metadata': {
                        'filename': os.path.basename(file_path),
                        'section': current_section,
                        'page': (i // 3) + 1,  # Rough estimate
                        'type': 'paragraph'
                    }
                })

        except Exception as e:
            print(f"Fallback extraction error: {e}")
            # Last resort: return whole text as one chunk
            try:
                text = document.export_to_text()
                if len(text) > 200:
                    chunks.append({
                        'text': text[:2000],  # Limit to 2000 chars
                        'metadata': {
                            'filename': os.path.basename(file_path),
                            'section': 'FULL_DOCUMENT',
                            'page': 1,
                            'type': 'full_text'
                        }
                    })
            except Exception as e2:
                print(f"Final fallback failed: {e2}")

        return chunks

    async def _process_plain_text(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Process plain text/markdown files without Docling.
        Splits into overlapping chunks.
        """
        import aiofiles

        async with aiofiles.open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = await f.read()

        if not text or len(text.strip()) < 50:
            return []

        return self._split_text_to_chunks(text, file_path)

    def _split_text_to_chunks(self, text: str, file_path: str) -> List[Dict[str, Any]]:
        chunks = []
        cleaned = " ".join(text.split())
        step = max(self.chunk_size - self.chunk_overlap, 1)
        index = 0
        page_number = 1

        for start in range(0, len(cleaned), step):
            chunk_text = cleaned[start:start + self.chunk_size]
            if len(chunk_text.strip()) < 50:
                continue
            chunks.append({
                "text": chunk_text.strip(),
                "metadata": {
                    "filename": os.path.basename(file_path),
                    "section": "PLAIN_TEXT",
                    "page": page_number,
                    "type": "plain_text"
                }
            })
            index += 1
            if index % 3 == 0:
                page_number += 1

        return chunks
