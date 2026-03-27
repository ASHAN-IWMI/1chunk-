import os
import fitz
import re
import pickle
import hashlib
import time
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer, CrossEncoder
import google.generativeai as genai
import logging
import requests
import json
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken
import functools
from source_normalizer import normalize_source_name

_embedding_model_cache = None
def get_cached_embedding_model():
    global _embedding_model_cache
    if _embedding_model_cache is None:
        print("🔄 [CACHE MISS] Loading SentenceTransformer Embedding Model into memory...")
        _embedding_model_cache = SentenceTransformer("BAAI/bge-base-en-v1.5")
        _embedding_model_cache.eval()
        print("✅ [CACHE SET] Embedding Model loaded successfully!")
    return _embedding_model_cache

_cross_encoder_cache = None
def get_cached_cross_encoder():
    global _cross_encoder_cache
    if _cross_encoder_cache is None:
        print("🔄 [CACHE MISS] Loading CrossEncoder Model into memory...")
        _cross_encoder_cache = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        print("✅ [CACHE SET] CrossEncoder loaded successfully!")
    return _cross_encoder_cache

_cross_encoder_stage2_cache: Dict[str, CrossEncoder] = {}
def get_cached_cross_encoder_stage2(model_name: str):
    name = (model_name or "").strip()
    if not name:
        return None
    if name not in _cross_encoder_stage2_cache:
        print(f"ðŸ”„ [CACHE MISS] Loading Stage-2 CrossEncoder Model into memory: {name}")
        _cross_encoder_stage2_cache[name] = CrossEncoder(name)
        print("âœ… [CACHE SET] Stage-2 CrossEncoder loaded successfully!")
    return _cross_encoder_stage2_cache[name]

_embedding_model_cache_by_name: Dict[str, SentenceTransformer] = {}
def get_cached_embedding_model():
    model_name = os.getenv("RAGPIPELINE1_EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5").strip() or "BAAI/bge-base-en-v1.5"
    if model_name not in _embedding_model_cache_by_name:
        print(f"Loading SentenceTransformer Embedding Model into memory: {model_name}")
        _embedding_model_cache_by_name[model_name] = SentenceTransformer(model_name)
        _embedding_model_cache_by_name[model_name].eval()
        print("Embedding Model loaded successfully!")
    return _embedding_model_cache_by_name[model_name]

_cross_encoder_cache_by_name: Dict[str, CrossEncoder] = {}
def get_cached_cross_encoder(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    name = (model_name or "").strip() or "cross-encoder/ms-marco-MiniLM-L-6-v2"
    if name not in _cross_encoder_cache_by_name:
        print(f"Loading CrossEncoder Model into memory: {name}")
        _cross_encoder_cache_by_name[name] = CrossEncoder(name)
        print("CrossEncoder loaded successfully!")
    return _cross_encoder_cache_by_name[name]

_retrievers_cache = {}
def get_cached_retrievers(index_files_tuple, k=24):
    cache_key = (index_files_tuple, k)
    if cache_key in _retrievers_cache:
        return _retrievers_cache[cache_key]
    print(f"🔄 [CACHE MISS] Loading and stacking FAISS and BM25 databases from {len(index_files_tuple)} files...")
    all_documents = []
    all_embeddings = []
    loaded_files = []
    
    for index_file in index_files_tuple:
        if os.path.exists(index_file):
            try:
                with open(index_file, "rb") as f:
                    data = pickle.load(f)
                all_documents.extend(data["documents"])
                all_embeddings.append(data["embeddings"])
                loaded_files.append(index_file)
            except Exception as e:
                logging.getLogger(__name__).error(f"Failed to load index file '{index_file}': {e}")
                
    if not loaded_files:
        return None, None, None, None
        
    documents = all_documents
    if len(all_embeddings) > 1:
        embeddings = np.vstack(all_embeddings)
    else:
        embeddings = all_embeddings[0]
        
    texts = [doc.page_content for doc in documents]
    emb_model = get_cached_embedding_model()
    
    faiss_index = FAISS.from_embeddings(
        text_embeddings=list(zip(texts, embeddings)),
        embedding=lambda x: emb_model.encode(x, normalize_embeddings=True),
        metadatas=[doc.metadata for doc in documents]
    )
    # INCREASE: k from 30 to 50 for deeper reranking pool
    faiss_retriever = faiss_index.as_retriever(search_kwargs={"k": k})
    
    bm25_retriever = BM25Retriever.from_documents(documents)
    # INCREASE: k from 30 to 50 for deeper reranking pool
    bm25_retriever.k = k
    
    print(f"✅ [CACHE SET] Vector databases stacked and loaded into memory! (Total chunks: {len(documents)})")
    result = (documents, embeddings, faiss_retriever, bm25_retriever)
    _retrievers_cache[cache_key] = result
    return result

# Import agentic tools
from gemini_tools import ToolExecutor, get_tool_schemas_for_gemini, format_tool_result_for_prompt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SemanticChunker:
    """
    Very simple semantic-ish chunker:
    1) First split into small fixed chunks.
    2) Then merge adjacent chunks whose embeddings are very similar.
    """

    def __init__(
        self,
        embedding_model: SentenceTransformer,
        base_chunk_size: int = 512,
        base_overlap: int = 50,
        sim_threshold: float = 0.85,
    ):
        from langchain_text_splitters import CharacterTextSplitter

        self.embedding_model = embedding_model
        self.base_splitter = CharacterTextSplitter(
            chunk_size=base_chunk_size,
            chunk_overlap=base_overlap,
            separator="\n\n",
        )
        self.sim_threshold = sim_threshold

    def split_text(self, text: str) -> list[str]:
        # 1) base split
        mini_chunks = self.base_splitter.split_text(text)
        if len(mini_chunks) <= 1:
            return mini_chunks

        # 2) embed all mini-chunks
        embs = self.embedding_model.encode(
            mini_chunks, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True
        )

        # 3) merge adjacent chunks with high cosine similarity
        merged_chunks: list[str] = []
        current = mini_chunks[0]
        current_emb = embs[0]

        for i in range(1, len(mini_chunks)):
            # Round similarity to 6 decimal places to prevent floating-point drift across CPUs
            sim = round(float(np.dot(current_emb, embs[i])), 6)
            if sim >= self.sim_threshold:
                # same topic: merge
                current = current + " " + mini_chunks[i]
                # update embedding as average of both (approx)
                current_emb = (current_emb + embs[i]) / 2.0
            else:
                merged_chunks.append(current)
                current = mini_chunks[i]
                current_emb = embs[i]

        merged_chunks.append(current)
        return merged_chunks

class EnsembleRetriever(BaseRetriever):
    """Ensemble retriever that combines multiple retrievers."""
    
    retrievers: List[BaseRetriever]
    weights: List[float]
    
    def __init__(self, retrievers: List[BaseRetriever], weights: List[float] = None):
        # Calculate weights if not provided
        if weights is None:
            weights = [1.0 / len(retrievers)] * len(retrievers)
        
        # Call parent __init__ with the fields as keyword arguments
        super().__init__(retrievers=retrievers, weights=weights)
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Get relevant documents from all retrievers."""
        all_docs = []
        doc_scores = {}
        
        for retriever, weight in zip(self.retrievers, self.weights):
            # FIXED: Try both invoke() and get_relevant_documents() for compatibility
            try:
                docs = retriever.invoke(query)  # New LangChain method
            except AttributeError:
                try:
                    docs = retriever.get_relevant_documents(query)  # Old method
                except AttributeError:
                    logger.warning(f"Retriever {type(retriever)} has no compatible method")
                    continue
            
            for i, doc in enumerate(docs):
                doc_id = doc.page_content
                score = weight * (1.0 / (i + 1))  # Reciprocal rank fusion
                
                if doc_id in doc_scores:
                    doc_scores[doc_id]['score'] += score
                else:
                    doc_scores[doc_id] = {'doc': doc, 'score': score}
        
        # Sort by score and return documents
        sorted_docs = sorted(doc_scores.values(), key=lambda x: x['score'], reverse=True)
        return [item['doc'] for item in sorted_docs]
    
    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """Async version - for now just call sync version."""
        return self._get_relevant_documents(query)
    
    def invoke(self, query: str) -> List[Document]:
        """Public invoke method for new LangChain compatibility."""
        return self._get_relevant_documents(query)

class RelevanceChecker:
    """
    RelevanceChecker: rerank, threshold, optional contextual compression.
    """

    def __init__(
        self,
        embedding_model: SentenceTransformer,
        cross_encoder_name: Optional[str] = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        cross_encoder_stage2_name: Optional[str] = None,
        stage2_top_n: int = 20,
        threshold: float = 0.70,
        min_docs: int = 2,
        max_docs: int = 6,
        enable_compression: bool = True,
        compression_top_sentences: int = 3,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        fallback_to_cosine: bool = True
    ):
        self.embedding_model = embedding_model
        self.cross_encoder = None
        self.cross_encoder_stage2 = None
        self.cross_encoder_name = cross_encoder_name
        self.cross_encoder_stage2_name = cross_encoder_stage2_name
        self.stage2_top_n = stage2_top_n

        if cross_encoder_name:
            try:
                self.cross_encoder = get_cached_cross_encoder(cross_encoder_name)
                logger.info(f"Loaded CrossEncoder from cache: {cross_encoder_name}")
            except Exception as e:
                logger.warning(f"Failed to load cross-encoder '{cross_encoder_name}': {e}")
                self.cross_encoder = None

        if cross_encoder_stage2_name:
            try:
                self.cross_encoder_stage2 = get_cached_cross_encoder_stage2(cross_encoder_stage2_name)
                if self.cross_encoder_stage2 is not None:
                    logger.info(f"Loaded Stage-2 CrossEncoder from cache: {cross_encoder_stage2_name}")
            except Exception as e:
                logger.warning(f"Failed to load stage-2 cross-encoder '{cross_encoder_stage2_name}': {e}")
                self.cross_encoder_stage2 = None

        self.threshold = threshold
        self.min_docs = min_docs
        self.max_docs = max_docs
        self.enable_compression = enable_compression
        self.compression_top_sentences = compression_top_sentences
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.fallback_to_cosine = fallback_to_cosine

    # -------------------------
    # Public API
    # -------------------------
    def filter_documents(
        self,
        question: str,
        docs: List,
        doc_embeddings: Optional[np.ndarray] = None
    ) -> List[Tuple[object, float]]:

        logger.info(f"Filtering {len(docs)} retrieved chunks for question: {question}")

        if not docs:
            logger.warning("No documents retrieved.")
            return []

        # 1) Scoring
        if self.cross_encoder is not None:
            scored = self._score_with_crossencoder(question, docs)
        else:
            scored = self._score_with_cosine(question, docs, doc_embeddings)

        # Optional Stage-2 rerank on top-N (precision boost)
        if self.cross_encoder_stage2 is not None and scored:
            top_docs = [d for d, _ in scored[: self.stage2_top_n]]
            try:
                rescored = self._score_with_crossencoder_model(self.cross_encoder_stage2, question, top_docs)
                scored = rescored + scored[self.stage2_top_n :]
                logger.info(f"Stage-2 rerank applied on top {min(self.stage2_top_n, len(top_docs))} candidates.")
            except Exception as e:
                logger.warning(f"Stage-2 rerank failed: {e}")

        # Log all scored chunks
        for i, (doc, score) in enumerate(scored):
            logger.info(f"[Score] Rank={i+1} Score={score:.4f} Content={doc.page_content[:150]}...")

        # 2) Thresholding
        filtered = [(d, s) for d, s in scored if s >= self.threshold]
        logger.info(f"Chunks above threshold {self.threshold}: {len(filtered)}")

        if len(filtered) < self.min_docs:
            logger.info(f"Below min_docs={self.min_docs}, selecting top {self.min_docs} anyway.")
            filtered = scored[: self.min_docs]

        filtered = filtered[: self.max_docs]

        # Log filtered results
        for doc, score in filtered:
            logger.info(f"[Selected] Score={score:.4f} Content={doc.page_content[:150]}...")

        # 3) Compression
        if self.enable_compression:
            compressed = []
            for doc, score in filtered:
                logger.info(f"Compressing chunk (score={score:.4f}): {doc.page_content[:150]}...")
                compressed_doc = self._compress_document(question, doc, top_k=self.compression_top_sentences)
                logger.info(f"[Compressed Result] {compressed_doc.page_content[:200]}...")
                compressed.append((compressed_doc, score))
            filtered = compressed

        return filtered

    # -------------------------
    # Internal scoring helpers
    # -------------------------
    def _score_with_crossencoder(self, question: str, docs: List) -> List[Tuple[object, float]]:
        return self._score_with_crossencoder_model(self.cross_encoder, question, docs)

    def _score_with_crossencoder_model(self, model, question: str, docs: List) -> List[Tuple[object, float]]:
        cross_input = [(question, doc.page_content) for doc in docs]
        try:
            scores = model.predict(cross_input, batch_size=self.batch_size)
            logger.info("Cross-encoder scoring successful.")
        except Exception as e:
            logger.warning(f"Cross-encoder failed: {e}, falling back to cosine.")
            return self._score_with_cosine(question, docs)

        scores = self._minmax_normalize(np.array(scores))
        return sorted(list(zip(docs, scores.tolist())), key=lambda x: x[1], reverse=True)


    def _score_with_cosine(self, question: str, docs: List, doc_embeddings: Optional[np.ndarray] = None):
        logger.info("Using cosine similarity scoring...")

        q_emb = self.embedding_model.encode([question], show_progress_bar=False, convert_to_numpy=True)
        if self.normalize_embeddings:
            q_emb = self._l2_normalize(q_emb)

        if doc_embeddings is None:
            texts = [d.page_content for d in docs]
            doc_embeddings = self.embedding_model.encode(texts, batch_size=self.batch_size,
                                                         show_progress_bar=False, convert_to_numpy=True)
        if self.normalize_embeddings:
            doc_embeddings = self._l2_normalize(doc_embeddings)

        sims = np.dot(doc_embeddings, q_emb.T).reshape(-1)
        sims = (sims + 1.0) / 2.0  # normalize to 0–1
        # Round to 6 decimal places for deterministic thresholding across CPUs
        sims = np.round(sims, 6)

        return sorted(list(zip(docs, sims.tolist())), key=lambda x: x[1], reverse=True)

    # -------------------------
    # Contextual compression
    # -------------------------
    def _compress_document(self, question: str, doc, top_k: int = 3):
        text = doc.page_content
        sentences = self._split_sentences(text)

        logger.info(f"Splitting into {len(sentences)} sentences for compression.")

        if not sentences:
            return doc

        q_emb = self.embedding_model.encode([question], show_progress_bar=False, convert_to_numpy=True)
        sent_embs = self.embedding_model.encode(sentences, batch_size=self.batch_size, show_progress_bar=False, convert_to_numpy=True)

        if self.normalize_embeddings:
            q_emb = self._l2_normalize(q_emb)
            sent_embs = self._l2_normalize(sent_embs)

        sims = np.dot(sent_embs, q_emb.T).reshape(-1)

        # Log top sentences
        idx_scores = list(enumerate(sims))
        idx_scores_sorted = sorted(idx_scores, key=lambda x: x[1], reverse=True)
        for i, (idx, score) in enumerate(idx_scores_sorted[:top_k]):
            logger.info(f"[Compression Sentence #{i+1}] Score={score:.4f} Sentence={sentences[idx][:200]}")

        top_idx = [i for i, _ in idx_scores_sorted[:top_k]]
        top_idx.sort()

        compressed_text = " ".join([sentences[i] for i in top_idx]).strip()

        compressed_doc = type(doc)(
            page_content=compressed_text,
            metadata={**getattr(doc, "metadata", {})}
        )
        return compressed_doc

    # -------------------------
    # Utilities
    # -------------------------
    @staticmethod
    def _minmax_normalize(arr):
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn + 1e-12)

    @staticmethod
    def _l2_normalize(x):
        denom = np.linalg.norm(x, axis=1, keepdims=True)
        denom[denom == 0] = 1e-12
        return x / denom

    @staticmethod
    def _split_sentences(text: str):
        parts = re.split(r'(?<=[\.\?\!])\s+', text)
        return [p.strip() for p in parts if p.strip()]
class ConversationManager:
    """Manages conversation history with model-aware token limits"""
    
    MODEL_LIMITS = {
        "gemini": 32768,      # Gemini API (Gemma 3 12B IT)
    }
    
    # 
    #     Args:
    #         llm_type: "gemini" for Gemma 3 12B IT via Gemini API
    #         reserve_tokens: Tokens to reserve for system prompt + retrieved docs + response
    #     
    def __init__(self, llm_type: str = "gemini", reserve_tokens: int = 8000):
        if llm_type not in self.MODEL_LIMITS:
            logger.warning(f"Unknown llm_type: {llm_type}. Defaulting to 'gemini'.")
            llm_type = "gemini"
        
        self.llm_type = llm_type
        self.max_context = self.MODEL_LIMITS[llm_type]
        self.reserve_tokens = reserve_tokens
        
        # Available tokens for conversation history
        self.available_for_history = self.max_context - reserve_tokens
        
        self.history: List[Dict[str, str]] = []
        
        # Initialize tokenizer
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning(f"Could not load tiktoken: {e}. Using character-based fallback.")
            self.tokenizer = None
        
        
        logger.info(f"ConversationManager initialized: {self.llm_type}, "
                   f"max={self.max_context}, available_for_history={self.available_for_history}")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Fallback: rough estimation (1 token ≈ 4 characters)
            return len(text) // 4
    
    def count_messages_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Count tokens in message list"""
        total = 0
        for msg in messages:
            # Count content tokens
            total += self.count_tokens(msg["content"])
            # Add overhead for message formatting (~4 tokens per message)
            total += 4
        return total
    
    def add_exchange(self, user_message: str, assistant_message: str):
        """Add a Q&A pair to history with automatic truncation"""
        # Add new messages
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": assistant_message})
        
        # Truncate if needed
        self._truncate_to_fit()
    
    def _truncate_to_fit(self):
        """Remove oldest messages until history fits in available token budget"""
        current_tokens = self.count_messages_tokens(self.history)
        
        # Keep removing oldest Q&A pairs until we fit
        while current_tokens > self.available_for_history and len(self.history) > 2:
            # Remove oldest Q&A pair (first 2 messages)
            removed = self.history[:2]
            self.history = self.history[2:]
            
            removed_tokens = self.count_messages_tokens(removed)
            current_tokens -= removed_tokens
            
            logger.info(f"Truncated conversation: removed {removed_tokens} tokens, "
                       f"remaining={current_tokens}/{self.available_for_history}")
        
        # Log if we're getting close to limit
        if current_tokens > self.available_for_history * 0.8:
            pairs = len(self.history) // 2
            logger.warning(f"Conversation history at 80% capacity: {current_tokens} tokens, {pairs} pairs")
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get current conversation history"""
        return self.history.copy()

    def set_history(self, messages: List[Dict[str, str]]):
        """Set history from external message list, ensuring token limits are respected"""
        self.history = []
        for msg in messages:
            if msg.get("role") in ["user", "assistant"]:
                self.history.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        self._truncate_to_fit()
    
    def get_history_tokens(self) -> int:
        """Get current token count of history"""
        return self.count_messages_tokens(self.history)
    
    def clear(self):
        """Clear conversation history"""
        self.history = []
        logger.info("Conversation history cleared")
    
    def get_stats(self) -> Dict:
        """Get conversation statistics"""
        pairs = len(self.history) // 2
        tokens = self.count_messages_tokens(self.history)
        
        return {
            "total_exchanges": pairs,
            "history_tokens": tokens,
            "available_tokens": self.available_for_history,
            "utilization_percent": round((tokens / self.available_for_history) * 100, 1),
            "model": self.llm_type,
            "max_context": self.max_context
        }
#sentence-transformers/all-mpnet-base-v2
class PDFExtractor:
    """Handles PDF extraction with layout preservation"""
    
    def __init__(self):
        self.header_footer_margin = 50
        self.min_text_length = 50
        self.heading_font_threshold = 13
        self.enable_ocr = os.getenv("RAGPIPELINE1_ENABLE_OCR", "0").strip() == "1"
        self.ocr_lang = os.getenv("RAGPIPELINE1_OCR_LANG", "eng").strip() or "eng"
        try:
            self.ocr_dpi = int(os.getenv("RAGPIPELINE1_OCR_DPI", "300"))
        except (TypeError, ValueError):
            self.ocr_dpi = 300

    @staticmethod
    def _df_to_markdown_table(df: pd.DataFrame) -> str:
        safe_df = df.fillna("").astype(str)
        headers = [str(h).strip() for h in safe_df.columns.tolist()]
        rows = safe_df.values.tolist()

        def esc(cell: str) -> str:
            return (cell or "").replace("|", "\\|").strip()

        header_line = "| " + " | ".join(esc(h) for h in headers) + " |"
        sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
        row_lines = ["| " + " | ".join(esc(str(c)) for c in row) + " |" for row in rows]
        return "\n".join([header_line, sep_line, *row_lines])
    
    def extract_pdf(self, pdf_path: str) -> List[Dict]:
        """
        Extract content from PDF with structure preservation
        Returns: List of content blocks with metadata
        """
        try:
            content_blocks = self._extract_with_layout(pdf_path)
            
            if not content_blocks:
                logger.warning(f"Layout extraction failed, using fallback for {pdf_path}")
                content_blocks = self._fallback_extraction(pdf_path)
            
            # Merge small adjacent blocks
            content_blocks = self._merge_blocks(content_blocks)
            
            return content_blocks
        
        except Exception as e:
            logger.error(f"PDF extraction failed for {pdf_path}: {e}")
            return []
    
    def _extract_with_layout(self, pdf_path: str) -> List[Dict]:
        """Extract text with layout and structure preservation"""
        doc = fitz.open(pdf_path)
        all_content = []
        current_h1, current_h2, current_h3 = "", "", ""
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_height = page.rect.height
            
            # Get text blocks with layout info
            blocks = page.get_text("dict")["blocks"]
            try:
                blocks = sorted(blocks, key=lambda b: (b.get("bbox", [0, 0, 0, 0])[1], b.get("bbox", [0, 0, 0, 0])[0]))
            except Exception:
                pass
            
            for block in blocks:
                if "lines" not in block:
                    continue
                
                bbox = block["bbox"]
                
                # Skip headers and footers
                if (bbox[1] < self.header_footer_margin or 
                    bbox[3] > page_height - self.header_footer_margin):
                    continue
                
                # Extract text and font information
                text_lines = []
                font_sizes = []
                
                for line in block["lines"]:
                    line_text = ""
                    for span in line["spans"]:
                        line_text += span["text"]
                        font_sizes.append(span["size"])
                    
                    if line_text.strip():
                        text_lines.append(line_text.strip())
                
                if not text_lines:
                    continue
                
                text = " ".join(text_lines)
                
                if len(text.strip()) < self.min_text_length:
                    continue
                
                # Detect headings by font size
                avg_font_size = np.mean(font_sizes) if font_sizes else 11
                content_type = "heading" if avg_font_size > self.heading_font_threshold else "paragraph"

                heading_level = None
                if content_type == "heading":
                    if avg_font_size >= (self.heading_font_threshold + 6):
                        heading_level = 1
                    elif avg_font_size >= (self.heading_font_threshold + 3):
                        heading_level = 2
                    else:
                        heading_level = 3

                    if heading_level == 1:
                        current_h1, current_h2, current_h3 = text, "", ""
                    elif heading_level == 2:
                        current_h2, current_h3 = text, ""
                    else:
                        current_h3 = text

                section_path = " > ".join([h for h in [current_h1, current_h2, current_h3] if h])
                
                all_content.append({
                    "text": text,
                    "page": page_num + 1,
                    "type": content_type,
                    "bbox": bbox,
                    "heading_level": heading_level,
                    "h1": current_h1,
                    "h2": current_h2,
                    "h3": current_h3,
                    "section_path": section_path,
                })
            
            # Extract tables separately
            tables = self._extract_tables(page, page_num + 1)
            if tables:
                section_path = " > ".join([h for h in [current_h1, current_h2, current_h3] if h])
                for t in tables:
                    t["h1"] = current_h1
                    t["h2"] = current_h2
                    t["h3"] = current_h3
                    t["section_path"] = section_path
            all_content.extend(tables)
        
        doc.close()
        return all_content
    
    def _extract_tables(self, page, page_num: int) -> List[Dict]:
        """Extract tables from page using PyMuPDF"""
        tables = []
        
        try:
            tabs = page.find_tables()
            
            for i, table in enumerate(tabs):
                df = table.to_pandas()
                
                if df.empty:
                    continue
                
                md = self._df_to_markdown_table(df)
                table_text = f"Table {i+1} (markdown):\n{md}"
                
                tables.append({
                    "text": table_text,
                    "page": page_num,
                    "type": "table",
                    "bbox": table.bbox
                })
        
        except Exception as e:
            logger.debug(f"Table extraction failed on page {page_num}: {e}")
        
        return tables
    
    def _fallback_extraction(self, pdf_path: str) -> List[Dict]:
        """Simple fallback if advanced extraction fails"""
        doc = fitz.open(pdf_path)
        content = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")

            # Optional OCR fallback for scanned/image-only pages.
            if self.enable_ocr and not (text or "").strip():
                try:
                    import pytesseract
                    from PIL import Image

                    pix = page.get_pixmap(dpi=self.ocr_dpi)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    text = pytesseract.image_to_string(img, lang=self.ocr_lang) or ""
                except Exception as e:
                    logger.debug(f"OCR failed on page {page_num + 1}: {e}")
            
            if text.strip():
                content.append({
                    "text": text.strip(),
                    "page": page_num + 1,
                    "type": "text"
                })
        
        doc.close()
        return content
    
    def _merge_blocks(self, content_blocks: List[Dict]) -> List[Dict]:
        """Merge small adjacent blocks on same page"""
        if not content_blocks:
            return []
        
        merged = []
        current_block = None
        
        for block in content_blocks:
            # Always keep tables separate
            if block["type"] == "table":
                if current_block:
                    merged.append(current_block)
                    current_block = None
                merged.append(block)
                continue
            
            if current_block is None:
                current_block = block.copy()
                continue
            
            # Merge if same page and combined text not too long
            if (block["page"] == current_block["page"] and 
                len(current_block["text"]) + len(block["text"]) < 800):
                current_block["text"] += " " + block["text"]
            else:
                merged.append(current_block)
                current_block = block.copy()
        
        if current_block:
            merged.append(current_block)
        
        return merged
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize extracted text"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\b\d{1,3}\s*$', '', text)
        text = re.sub(r'[^\w\s\.\,\:\;\-\(\)\[\]\"\'\%\$\#\@\!\?\/\&\+\=\*]', '', text)
        text = re.sub(r'\s+([.,;:!?)])', r'\1', text)
        text = re.sub(r'([(])\s+', r'\1', text)
        return text.strip()


class RAGPipeline1:
    """RAGPipeline1: LLM query expansion -> bge-large embeddings -> top-k retrieval -> BGE reranker -> top 3 chunks -> Gemma 3 12B IT grounded answer."""
    def __init__(
        self,
        pdf_folder: str,
        index_file: str,
        model_params: dict,
        reserve_tokens: int = 8000,
        gemini_rotator=None,
    ):
        # Paths
        self.pdf_folder = pdf_folder
        self.index_file = index_file
        
        # Configure Gemini API with first key or rotator
        self.gemini_rotator = gemini_rotator
        if self.gemini_rotator:
            idx, key = self.gemini_rotator.get_next_key()
            genai.configure(api_key=key)
            self.current_key_idx = idx
            masked_key = f"{key[:4]}...{key[-4:]}" if len(key) > 8 else "****"
            print(f"🚀 [Marsh Fast] Initialized with Google Gemini API Key #{idx + 1} ({masked_key})")
        elif "google_api_key" in model_params or "groq_api_key" in model_params:
            key = model_params.get("groq_api_key") or model_params.get("google_api_key")
            genai.configure(api_key=key)
            self.current_key_idx = None
            masked_key = f"{key[:4]}...{key[-4:]}" if len(key) > 8 else "****"
            print(f"🚀 [Marsh Fast] Initialized with static Google Gemini API Key ({masked_key})")
        else:
            raise ValueError("Either model_params['groq_api_key'], model_params['google_api_key'], or gemini_rotator is required")
        
        # Initialize Gemini model with Gemma 3 12B IT
        # Using models/gemma-3-12b-it for higher-quality answers
        requested_model_name = (model_params.get("model_name") or "").strip()
        self.model_name = requested_model_name if requested_model_name and not requested_model_name.startswith("models/") else "openai/gpt-oss-20b"
        
        # Create generative model - NOT using tools parameter for Gemma manual loop
        # Enforce determinism with temperature 0.0
        # Added safety settings to prevent false positives when analyzing environmental policy/penalties
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        
        self.llm_client = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config={
                # ── DETERMINISM LOCK ──────────────────────────────────────────
                # temperature=0.0 + top_k=1 + top_p=1.0 → pure greedy decoding.
                # The model ALWAYS picks the single highest-probability token.
                # This guarantees byte-for-byte identical answers across any
                # device, browser, OS, or session for the same conversation state.
                "temperature": 0.0,
                "top_p": 1.0,           # disable nucleus sampling
                "top_k": 1,             # greedy: always pick #1 token
                "max_output_tokens": 4500,  # allow detailed, comprehensive answers
            },
            safety_settings=safety_settings
        )
        self.groq_api_base = model_params.get("groq_api_base", "https://api.groq.com/openai/v1/chat/completions")
        self.groq_api_keys = self._collect_groq_api_keys(model_params)
        try:
            self.groq_max_tokens = int(model_params.get("groq_max_tokens", os.getenv("RAGPIPELINE1_GROQ_MAX_TOKENS", "8000")))
        except (TypeError, ValueError):
            self.groq_max_tokens = 8000
        try:
            self.groq_retry_base_delay = float(os.getenv("RAGPIPELINE1_GROQ_RETRY_BASE_DELAY", "1.5"))
        except (TypeError, ValueError):
            self.groq_retry_base_delay = 1.5
        try:
            self.groq_retry_max_attempts = int(os.getenv("RAGPIPELINE1_GROQ_RETRY_MAX_ATTEMPTS", "0"))
        except (TypeError, ValueError):
            self.groq_retry_max_attempts = 0
        try:
            self.final_context_docs = int(os.getenv("RAGPIPELINE1_FINAL_CONTEXT_DOCS", "5"))
        except (TypeError, ValueError):
            self.final_context_docs = 5
        self.use_local_groq_rotation = bool(self.groq_api_keys) or ("groq_api_key" in model_params)
        if self.groq_api_keys:
            self.current_api_key = self.groq_api_keys[0]
            self.current_key_idx = 0
        elif "groq_api_key" in model_params:
            self.current_api_key = model_params["groq_api_key"]
        elif "google_api_key" in model_params:
            self.current_api_key = model_params["google_api_key"]
        else:
            self.current_api_key = None
        if self.current_api_key is None and "key" in locals():
            self.current_api_key = key
        
        self.last_retrieved_docs = []
        self.last_intent = None
        
        logger.info(f"Initialized Deterministic Gemini model: {self.model_name}")
        
        # Initialize conversation manager for Gemini
        self.conversation_manager = ConversationManager(
            llm_type="gemini",
            reserve_tokens=reserve_tokens
        )
        
        # Initialize tool executor (will be set after retrievers are built)
        self.tool_executor = None
        
        # Models
        self.embedding_model = get_cached_embedding_model()
        # Architecture for RAGPipeline1 only:
        # User Query -> LLM Query Expansion -> bge-large embedding -> FAISS/BM25 retrieval
        # -> top-k retrieval -> BGE reranker -> top 3 chunks -> Gemma 3 12B IT -> Grounded Answer
        stage2_name = os.getenv("RAGPIPELINE1_STAGE2_CROSS_ENCODER", "").strip() or None
        try:
            stage2_top_n = int(os.getenv("RAGPIPELINE1_STAGE2_TOP_N", "8"))
        except (TypeError, ValueError):
            stage2_top_n = 8
        self.relevance_checker = RelevanceChecker(
            embedding_model=self.embedding_model,
            cross_encoder_name=os.getenv("RAGPIPELINE1_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            cross_encoder_stage2_name=stage2_name,
            stage2_top_n=stage2_top_n,
            threshold=0.68,
            min_docs=3,
            max_docs=3,
            enable_compression=False,
            compression_top_sentences=3
        )
        
        self.pdf_extractor = PDFExtractor()
        
        # Text splitter
        # self.text_splitter = RecursiveCharacterTextSplitter(
        #     chunk_size=700,
        #     chunk_overlap=150,
        #     length_function=len,
        #     separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
        #     is_separator_regex=False,
        # )

        #Simple sliding window chunker.
        # from langchain_text_splitters import CharacterTextSplitter

        # self.text_splitter = CharacterTextSplitter(
        #     chunk_size=512,
        #     chunk_overlap=50,
        #     separator="\n\n",   
        # )

        #Simple fixed length chunker.
        # from langchain_text_splitters import CharacterTextSplitter

        # self.text_splitter = CharacterTextSplitter(
        #     chunk_size=512,
        #     chunk_overlap=0,
        #     separator="\n\n",   
        # )

        #Semantic-ish chunker.
        self.text_splitter = SemanticChunker(
            embedding_model=self.embedding_model,
            base_chunk_size=512,
            base_overlap=50,
            sim_threshold=0.85,
        )
        """The Semantica actually worked better than the sliding window chunker."""
        """Fixed length is shape. But it migth not be good to pick up on context."""
        """Sliding window chunker actually made it better. I'll have to tweak more with the chunk_size and chunk_overlap."""
        """When I checked the chunks found in recursive splitter it did have the required texts, but the recursive splitter
           seems to split them at random points leading to a losss of context."""
        
        # Storage
        self.documents = []
        self.embeddings = None
        self.faiss_retriever = None
        self.bm25_retriever = None
        self.hybrid_retriever = None

        # Retrieval enhancements (additive; keep legacy path available)
        self.enable_enhanced_retrieval = os.getenv("RAGPIPELINE1_ENHANCED_RETRIEVAL", "1").strip() != "0"
        self.enable_hyde = os.getenv("RAGPIPELINE1_ENABLE_HYDE", "1").strip() != "0"
        self.enable_entity_prefilter = os.getenv("RAGPIPELINE1_ENABLE_ENTITY_PREFILTER", "1").strip() != "0"
        self.enable_similarity_dedupe = os.getenv("RAGPIPELINE1_ENABLE_SIM_DEDUPE", "1").strip() != "0"
        self.enable_intent_classification = os.getenv("RAGPIPELINE1_ENABLE_INTENT_CLASSIFICATION", "1").strip() != "0"
        self.enable_query_rewrite = os.getenv("RAGPIPELINE1_ENABLE_QUERY_REWRITE", "0").strip() != "0"
        self.enable_entity_query_boost = os.getenv("RAGPIPELINE1_ENABLE_ENTITY_QUERY_BOOST", "1").strip() != "0"
        self.enable_adaptive_retrieval = os.getenv("RAGPIPELINE1_ENABLE_ADAPTIVE_RETRIEVAL", "1").strip() != "0"
        self.enable_source_diversity = os.getenv("RAGPIPELINE1_ENABLE_SOURCE_DIVERSITY", "1").strip() != "0"
        self.enable_sibling_injection = os.getenv("RAGPIPELINE1_ENABLE_SIBLINGS", "1").strip() != "0"
        self.enable_parent_windowing = os.getenv("RAGPIPELINE1_ENABLE_PARENT_WINDOW", "1").strip() != "0"
        try:
            self.parent_max_tokens = int(os.getenv("RAGPIPELINE1_PARENT_MAX_TOKENS", "450"))
        except (TypeError, ValueError):
            self.parent_max_tokens = 450

        self.enable_eval_logging = os.getenv("RAGPIPELINE1_ENABLE_EVAL_LOGGING", "1").strip() != "0"

        try:
            self.max_candidate_pool = int(os.getenv("RAGPIPELINE1_CANDIDATE_POOL", "24"))
        except (TypeError, ValueError):
            self.max_candidate_pool = 24

        try:
            self.retrieval_depth = int(os.getenv("RAGPIPELINE1_RETRIEVAL_DEPTH", "24"))
        except (TypeError, ValueError):
            self.retrieval_depth = 24

        try:
            self.max_query_variants = int(os.getenv("RAGPIPELINE1_MAX_QUERY_VARIANTS", "3"))
        except (TypeError, ValueError):
            self.max_query_variants = 3

        try:
            self.max_sub_queries = int(os.getenv("RAGPIPELINE1_MAX_SUB_QUERIES", "2"))
        except (TypeError, ValueError):
            self.max_sub_queries = 2

        try:
            self.complex_candidate_pool = int(os.getenv("RAGPIPELINE1_COMPLEX_CANDIDATE_POOL", "40"))
        except (TypeError, ValueError):
            self.complex_candidate_pool = 40

        try:
            self.complex_retrieval_depth = int(os.getenv("RAGPIPELINE1_COMPLEX_RETRIEVAL_DEPTH", "40"))
        except (TypeError, ValueError):
            self.complex_retrieval_depth = 40

        try:
            self.complex_final_context_docs = int(os.getenv("RAGPIPELINE1_COMPLEX_FINAL_CONTEXT_DOCS", "6"))
        except (TypeError, ValueError):
            self.complex_final_context_docs = 6

        try:
            self.fast_track_iterations = int(os.getenv("RAGPIPELINE1_FAST_TRACK_ITERS", "4"))
        except (TypeError, ValueError):
            self.fast_track_iterations = 4

        try:
            self.research_iterations = int(os.getenv("RAGPIPELINE1_RESEARCH_ITERS", "8"))
        except (TypeError, ValueError):
            self.research_iterations = 8

        self.enable_direct_grounded_answer = os.getenv("RAGPIPELINE1_ENABLE_DIRECT_ANSWER", "1").strip() != "0"

        try:
            self.dedupe_sim_threshold = float(os.getenv("RAGPIPELINE1_DEDUPE_SIM_THRESHOLD", "0.97"))
        except (TypeError, ValueError):
            self.dedupe_sim_threshold = 0.97

        self.rrf_weights = (0.55, 0.30, 0.15)  # semantic, BM25, HyDE

        # Chunk post-processing (only impacts newly-built indexes)
        self.chunk_sentence_guard = os.getenv("RAGPIPELINE1_CHUNK_SENTENCE_GUARD", "1").strip() != "0"
        self.chunk_sentence_overlap = os.getenv("RAGPIPELINE1_CHUNK_SENTENCE_OVERLAP", "1").strip() != "0"

        # Lazy caches
        self._content_emb_map = None
        self.metadata_index = None
        self.gazetteer = None
        self.chunk_lookup = None
    
    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        parts = re.split(r'(?<=[\.\?\!])\s+', text or "")
        return [p.strip() for p in parts if p.strip()]

    def _postprocess_chunks(self, chunks: List[str]) -> List[str]:
        if not chunks:
            return chunks

        out = [c.strip() for c in chunks if (c or "").strip()]
        if len(out) <= 1:
            return out

        if self.chunk_sentence_guard:
            guarded = []
            carry = ""
            for i, chunk in enumerate(out):
                text = (carry + " " + chunk).strip() if carry else chunk.strip()
                carry = ""

                # For non-final chunks, avoid ending mid-sentence by pushing trailing fragment forward.
                if i < (len(out) - 1) and text and not re.search(r"[.!?][\"']?\\s*$", text):
                    matches = list(re.finditer(r"[.!?][\"']?\\s+", text))
                    cut = None
                    for m in reversed(matches):
                        if m.end() >= int(len(text) * 0.6):
                            cut = m.end()
                            break
                    if cut is not None:
                        carry = text[cut:].strip()
                        text = text[:cut].strip()

                if text:
                    guarded.append(text)

            if carry and guarded:
                guarded[-1] = (guarded[-1] + " " + carry).strip()
            out = guarded

        if self.chunk_sentence_overlap and len(out) > 1:
            overlapped = [out[0]]
            for i in range(1, len(out)):
                prev_sentences = self._split_sentences(overlapped[i - 1])
                last_sentence = prev_sentences[-1] if prev_sentences else ""
                cur = out[i]
                if last_sentence and not cur.startswith(last_sentence):
                    cur = (last_sentence + " " + cur).strip()
                overlapped.append(cur)
            out = overlapped

        return out

    @staticmethod
    def _extract_legal_anchors(text: str) -> Dict[str, object]:
        t = text or ""
        anchors: Dict[str, object] = {}

        act_match = re.search(r'\\b([A-Z][A-Za-z]+(?:\\s+[A-Z][A-Za-z]+){0,6}\\s+(?:Act|Ordinance|Regulations|Rules))\\b', t)
        if act_match:
            anchors["act_name"] = act_match.group(1).strip()

        sections = re.findall(r'\\b(?:section|sec\\.|s\\.)\\s*(\\d+[A-Za-z]?)\\b', t, flags=re.IGNORECASE)
        if sections:
            anchors["sections"] = sorted(set(sections), key=lambda x: (len(x), x))

        articles = re.findall(r'\\b(?:article|art\\.)\\s*(\\d+[A-Za-z]?)\\b', t, flags=re.IGNORECASE)
        if articles:
            anchors["articles"] = sorted(set(articles), key=lambda x: (len(x), x))

        return anchors

    @staticmethod
    def _parent_sig(text: str) -> str:
        # Stable content signature for sibling lookup without storing huge strings as keys.
        b = (text or "").encode("utf-8", errors="ignore")
        return hashlib.sha1(b).hexdigest()[:16]

    def build_index(self, progress_callback=None, status_callback=None):
        """Build index from PDFs in folder"""
        # Deterministic: sort file names to ensure identical indexing on all machines
        pdf_files = sorted([f for f in os.listdir(self.pdf_folder) if f.lower().endswith(".pdf")])
        
        if not pdf_files:
            raise ValueError("No PDF files found in folder")
        
        all_documents = []
        
        for i, pdf_file in enumerate(pdf_files):
            if status_callback:
                status_callback(f"Processing: {pdf_file} ({i+1}/{len(pdf_files)})")
            
            pdf_path = os.path.join(self.pdf_folder, pdf_file)
            
            try:
                # Extract content blocks
                content_blocks = self.pdf_extractor.extract_pdf(pdf_path)
                
                # Create smart chunks
                documents = self._create_chunks(content_blocks, pdf_file)
                all_documents.extend(documents)
                
                logger.info(f"Extracted {len(documents)} chunks from {pdf_file}")
            
            except Exception as e:
                logger.error(f"Failed to process {pdf_file}: {e}")
                continue
            
            if progress_callback:
                progress_callback((i + 1) / len(pdf_files))
        
        if not all_documents:
            raise ValueError("No content extracted from PDFs")
        
        if status_callback:
            status_callback(f"Encoding {len(all_documents)} chunks...")
        
        # Encode documents
        texts = [doc.page_content for doc in all_documents]
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        # Build retrievers
        self._build_retrievers(all_documents, texts, embeddings)
        
        # Save index
        self.documents = all_documents
        self.embeddings = embeddings
        self._content_emb_map = None
        self._rebuild_metadata_indexes()
        self._save_index()
        
        if status_callback:
            status_callback(f"✅ Indexed {len(all_documents)} chunks from {len(pdf_files)} PDFs")
        
        return len(all_documents)
    
    def _create_chunks(self, content_blocks: List[Dict], pdf_name: str) -> List[Document]:
        """
        Create smart chunks from content blocks with Hierarchical (Parent-Child) support.
        Stores the full block (Parent) in the metadata of each chunk (Child).
        """
        documents = []
        
        for block in content_blocks:
            text = self.pdf_extractor.clean_text(block["text"])
            
            if len(text) < 50:
                continue

            base_meta = {
                "source": pdf_name,
                "page": block.get("page"),
                "type": block.get("type"),
                "h1": block.get("h1", ""),
                "h2": block.get("h2", ""),
                "h3": block.get("h3", ""),
                "section_path": block.get("section_path", ""),
            }
            base_meta.update(self._extract_legal_anchors(text))
            
            # Parent context is the full block text
            parent_text = text
            base_meta["parent_sig"] = self._parent_sig(parent_text)
            
            # Keep tables and short content as single chunks
            if block["type"] == "table" or len(text) < 600:
                documents.append(Document(
                    page_content=text,
                    metadata={
                        **base_meta,
                        "parent_text": parent_text  # Self-parent for short chunks
                    }
                ))
            else:
                # Split long content (Children)
                chunks = self.text_splitter.split_text(text)
                chunks = self._postprocess_chunks(chunks)
                
                for i, chunk in enumerate(chunks):
                    documents.append(Document(
                        page_content=chunk,
                        metadata={
                            **base_meta,
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                            "parent_text": parent_text  # Reference to full block
                        }
                    ))
        
        return documents
    
    def _build_retrievers(self, documents: List[Document], texts: List[str], embeddings: np.ndarray):
        """Build FAISS and BM25 retrievers"""
        # FAISS retriever
        faiss_index = FAISS.from_embeddings(
            text_embeddings=list(zip(texts, embeddings)),
            embedding=lambda x: self.embedding_model.encode(x, normalize_embeddings=True),
            metadatas=[doc.metadata for doc in documents]
        )
        self.faiss_retriever = faiss_index.as_retriever(search_kwargs={"k": self.retrieval_depth})
        
        # BM25 retriever
        self.bm25_retriever = BM25Retriever.from_documents(documents)
        self.bm25_retriever.k = self.retrieval_depth
        
        # Hybrid retriever
        self.hybrid_retriever = EnsembleRetriever(
            retrievers=[self.faiss_retriever, self.bm25_retriever],
            # REBALANCE: 0.85/0.15 -> 0.70/0.30 to favor keywords/Act numbers/terms
            weights=[0.70, 0.30]
        )
        
        # Initialize tool executor
        from gemini_tools import ToolExecutor
        self.tool_executor = ToolExecutor(self)
        logger.info("ToolExecutor initialized for RAGPipeline1")
    
    def _save_index(self):
        """Save index to disk"""
        with open(self.index_file, "wb") as f:
            pickle.dump({
                "documents": self.documents,
                "embeddings": self.embeddings,
                "model": "sentence-transformers/all-mpnet-base-v2",
                "metadata_index": self.metadata_index,
                "gazetteer": self.gazetteer,
                "version": 2,
            }, f)
    
    def load_index(self):
        """Load existing index from disk (supports single file or list of files) using cache"""
        index_files = self.index_file if isinstance(self.index_file, list) else [self.index_file]
        
        docs_and_rets = get_cached_retrievers(tuple(index_files), k=self.retrieval_depth)
        
        if docs_and_rets[0] is None:
            logger.error("No index files were loaded successfully")
            return False
            
        self.documents, self.embeddings, self.faiss_retriever, self.bm25_retriever = docs_and_rets
        self._content_emb_map = None
        self._rebuild_metadata_indexes()
        
        # Hybrid retriever
        self.hybrid_retriever = EnsembleRetriever(
            retrievers=[self.faiss_retriever, self.bm25_retriever],
            weights=[0.70, 0.30]
        )
        
        # Initialize tool executor
        from gemini_tools import ToolExecutor
        self.tool_executor = ToolExecutor(self)
        logger.info("ToolExecutor initialized directly from cache for RAGPipeline1")
        
        logger.info(f"Loaded merged index with {len(self.documents)} chunks from {len(index_files)} files (Cached)")
        return True
    
    def _normalize_query(self, question: str) -> str:
        """
        Canonical query preprocessing: strips filler words so that semantically
        identical questions from different typings are normalized before retrieval.
        Only used for retrieval; the original question is still fed to the LLM.
        """
        q = question.strip()
        # Remove common filler prefixes that add no retrieval signal
        filler_patterns = [
            r'^(?:please|kindly)\s+',
            r'^(?:can you|could you|would you|will you)\s+',
            r'^(?:tell me|explain|describe|show me|give me|provide)\s+(?:about\s+|me\s+)?',
            r'^(?:what is|what are|what were|what was)\s+',
            r'^(?:how does|how do|how can|how to)\s+',
            r'^(?:i want to know|i need to know|i would like to know)\s+',
        ]
        q_lower = q.lower()
        for pattern in filler_patterns:
            q_lower = re.sub(pattern, '', q_lower, flags=re.IGNORECASE).strip()
        # Return the cleaned lowercase version for retrieval queries
        return q_lower if q_lower else q.lower()

    def _decompose_query(self, question: str) -> List[str]:
        """
        Decompose a complex question into specific sub-queries for precision retrieval.
        Uses Gemma 3 12B IT to perform the decomposition.
        """
        prompt = f"""You are a high-precision query decomposer. 
Break down the user's question into 1-3 specific, factual sub-queries that can be answered by looking at documentation.
If the question is simple, return ONLY the original question.
DO NOT provide any conversational filler. Return each sub-query on a new line.

User Question: {question}
Sub-queries:"""
        
        # Use a temporary client or the shared tool-less model
        try:
            response = self._safe_generate_content(prompt)
            if not response or "Error:" in response:
                return [question]
            
            sub_queries = [q.strip("- ").strip() for q in response.split('\n') if q.strip()]
            logger.info(f"Decomposed query into: {sub_queries}")
            return sub_queries if sub_queries else [question]
        except Exception as e:
            logger.warning(f"Query decomposition failed: {e}")
            return [question]

    def _expand_queries(self, question: str) -> List[str]:
        """
        Generate 2-3 retrieval-oriented reformulations using an LLM prompt template.
        Falls back to deterministic template expansions if the LLM call fails.
        """
        base = question.strip()
        normalized = self._normalize_query(base)
        expansions = []

        prompt = f"""You are a retrieval query expander for a RAG system.
Generate up to {max(2, self.max_query_variants)} short search queries for the user question.
Rules:
- Focus on terms likely to appear in source documents.
- Include legal/regulatory keywords only when relevant.
- Keep each query on its own line.
- Do not number the lines.
- Include the original meaning exactly.

User Question: {base}
Expanded Queries:"""

        try:
            llm_output = (self._safe_generate_content(prompt, max_retries=3) or "").strip()
            if llm_output and "Error:" not in llm_output:
                expansions.extend([q.strip("- ").strip() for q in llm_output.splitlines() if q.strip()])
        except Exception:
            pass

        expansions.extend([
            normalized,
            base,
            f"{normalized} act regulations section policy",
            f"{normalized} penalties fines list rules",
        ])
        # Deduplicate while preserving order
        seen, unique = set(), []
        for q in expansions:
            q_stripped = q.strip().lower()
            if q_stripped and q_stripped not in seen:
                seen.add(q_stripped)
                unique.append(q.strip())
        return unique[: max(1, self.max_query_variants) ]

    def _boost_queries_with_entities(self, question: str, queries: List[str], entities: Dict[str, object]) -> List[str]:
        if not self.enable_entity_query_boost:
            return queries

        boosted = list(queries or [])
        acts = [a.strip() for a in (entities.get("acts") or []) if str(a).strip()]
        sections = [s.strip() for s in (entities.get("sections") or []) if str(s).strip()]
        years = [y.strip() for y in (entities.get("years") or []) if str(y).strip()]

        if acts:
            for act in acts[:2]:
                boosted.append(act)
                boosted.append(f"{question} {act}")
                boosted.append(f"{act} provisions requirements penalties")

        if acts and sections:
            for act in acts[:2]:
                for sec in sections[:3]:
                    boosted.append(f"{act} section {sec}")
                    boosted.append(f"{question} {act} section {sec}")
        elif sections:
            for sec in sections[:3]:
                boosted.append(f"{question} section {sec}")
                boosted.append(f"section {sec} penalties duties requirements")

        if years:
            for year in years[:2]:
                boosted.append(f"{question} {year}")

        seen, unique = set(), []
        for q in boosted:
            key = (q or "").strip().lower()
            if key and key not in seen:
                seen.add(key)
                unique.append((q or "").strip())
        return unique[: max(self.max_query_variants + 3, len(queries or []))]

    def _adaptive_retrieval_settings(self, question: str, intent: str, entities: Dict[str, object]) -> Dict[str, int]:
        if not self.enable_adaptive_retrieval:
            return {
                "retrieval_depth": self.retrieval_depth,
                "candidate_pool": self.max_candidate_pool,
                "final_context_docs": self.final_context_docs,
                "rerank_max_docs": self.relevance_checker.max_docs,
            }

        q = (question or "").lower()
        is_complex = (
            intent in {"comparative", "procedural"}
            or bool((entities.get("acts") or []) or (entities.get("sections") or []))
            or len(q) > 140
            or any(token in q for token in ["explain", "difference", "compare", "penalties", "requirements", "process"])
        )

        if not is_complex:
            return {
                "retrieval_depth": self.retrieval_depth,
                "candidate_pool": self.max_candidate_pool,
                "final_context_docs": self.final_context_docs,
                "rerank_max_docs": self.relevance_checker.max_docs,
            }

        rerank_max_docs = max(self.relevance_checker.max_docs, min(self.complex_final_context_docs, 8))
        return {
            "retrieval_depth": max(self.retrieval_depth, self.complex_retrieval_depth),
            "candidate_pool": max(self.max_candidate_pool, self.complex_candidate_pool),
            "final_context_docs": max(self.final_context_docs, self.complex_final_context_docs),
            "rerank_max_docs": rerank_max_docs,
        }

    def _should_decompose_query(self, question: str, intent: str) -> bool:
        q = (question or "").strip()
        if not q:
            return False
        if intent in {"comparative", "procedural"}:
            return True
        if len(q) > 140:
            return True
        complexity_markers = [
            " and ", " or ", " compare ", " difference ", " versus ", " vs ",
            " if ", " when ", " how ", " why ", " explain ", " steps ", " process ",
        ]
        return any(marker in q.lower() for marker in complexity_markers)

    def _should_use_hyde(self, question: str, intent: str, entities: Dict[str, object]) -> bool:
        if not self.enable_hyde:
            return False
        if entities.get("acts") or entities.get("sections"):
            return False
        q = (question or "").lower()
        if intent in {"comparative", "procedural"}:
            return True
        return len(q) > 100 or any(token in q for token in ["explain", "describe", "why", "how"])

    @staticmethod
    def _invoke_retriever(retriever, query: str) -> List[Document]:
        try:
            return retriever.invoke(query)  # LangChain new API
        except Exception:
            try:
                return retriever.get_relevant_documents(query)  # legacy API
            except Exception:
                return []

    def _extract_query_entities(self, question: str) -> Dict[str, object]:
        q = question or ""
        acts = re.findall(r'\\b([A-Z][A-Za-z]+(?:\\s+[A-Z][A-Za-z]+){0,6}\\s+(?:Act|Ordinance|Regulations|Rules))\\b', q)
        acts = list(dict.fromkeys([a.strip() for a in acts if a.strip()]))

        sections = re.findall(r'\\b(?:section|sec\\.|s\\.)\\s*(\\d+[A-Za-z]?)\\b', q, flags=re.IGNORECASE)
        sections = list(dict.fromkeys([s.strip() for s in sections if s.strip()]))

        years = re.findall(r'\\b(19\\d{2}|20\\d{2})\\b', q)
        dates = re.findall(r'\\b(\\d{1,2}[\\/\\-]\\d{1,2}[\\/\\-](?:19\\d{2}|20\\d{2}))\\b', q)

        return {
            "acts": acts,
            "sections": sections,
            "years": list(dict.fromkeys(years)),
            "dates": list(dict.fromkeys(dates)),
        }

    @staticmethod
    def _classify_intent(question: str) -> str:
        q = (question or "").lower()
        if any(w in q for w in ["compare", "difference", "vs", "versus", "better than", "pros and cons"]):
            return "comparative"
        if any(w in q for w in ["how to", "procedure", "process", "steps", "apply", "obtain", "renew", "permit", "license", "compliance"]):
            return "procedural"
        return "factual"

    @staticmethod
    def _answer_format_instructions(question: str, intent: str, is_informative: bool = False) -> str:
        q = (question or "").lower()

        wants_table = (
            intent == "comparative"
            or any(token in q for token in [
                "compare", "comparison", "difference", "differences", "versus", "vs",
                "table", "tabulate", "side by side", "columns"
            ])
        )
        wants_bullets = (
            wants_table
            or intent == "procedural"
            or is_informative
            or any(token in q for token in [
                "list", "steps", "requirements", "points", "bullet", "bullets",
                "what are", "types of", "penalties", "fines", "duties", "obligations",
                "documents needed", "conditions", "criteria", "how to", "process"
            ])
        )

        if wants_table:
            return (
                "FORMAT RULES:\n"
                "- Use a markdown table when comparing items, options, penalties, requirements, or multiple entities.\n"
                "- Add short bullets below the table only if clarification is needed.\n"
                "- Do not use a table for simple single-fact answers.\n"
                "- Keep the answer tightly scoped to what the user asked.\n"
            )

        if wants_bullets:
            return (
                "FORMAT RULES:\n"
                "- Use bullet points for lists, steps, requirements, penalties, or multi-part answers.\n"
                "- Use short headings only when they improve clarity.\n"
                "- Do not add a table unless the question is explicitly comparative.\n"
                "- Keep the answer tightly scoped to what the user asked.\n"
            )

        return (
            "FORMAT RULES:\n"
            "- Answer in a short natural paragraph unless the question clearly asks for a list or comparison.\n"
            "- Do not add bullets or tables for a simple direct question.\n"
            "- Keep the answer tightly scoped to what the user asked.\n"
        )

    def _rewrite_query_with_history(self, question: str) -> str:
        if not self.enable_query_rewrite:
            return question

        q = (question or "").strip()
        if not q:
            return question

        # Only rewrite when the question likely references prior context.
        needs_rewrite = any(p in q.lower() for p in ["it", "this", "that", "they", "those", "these", "what about", "and what about", "penalties", "fines"])
        if not needs_rewrite:
            return question

        history = getattr(self.conversation_manager, "history", []) or []
        if len(history) < 2:
            return question

        # Use the last exchange as context for disambiguation.
        last_user = ""
        last_assistant = ""
        for msg in reversed(history):
            if msg.get("role") == "assistant" and not last_assistant:
                last_assistant = msg.get("content", "")
            if msg.get("role") == "user" and not last_user:
                last_user = msg.get("content", "")
            if last_user and last_assistant:
                break

        prompt = (
            "Rewrite the user's new question into a fully self-contained, explicit query that can be used for document retrieval. "
            "Resolve pronouns and implicit references using the prior exchange. "
            "If you can infer a specific Act name or Section number, include it explicitly. "
            "Return only the rewritten query.\n\n"
            f"Previous user question: {last_user}\n"
            f"Previous assistant answer (may be partial): {last_assistant}\n\n"
            f"New user question: {q}\n\n"
            "Rewritten retrieval query:"
        )

        try:
            rewritten = (self._safe_generate_content(prompt) or "").strip()
            if rewritten and "Error:" not in rewritten and len(rewritten) <= 400:
                return rewritten
        except Exception:
            pass

        return question

    def _doc_matches_entities(self, doc: Document, entities: Dict[str, object]) -> bool:
        if not entities:
            return True

        acts = [a.lower() for a in (entities.get("acts") or [])]
        sections = set((entities.get("sections") or []))

        meta = getattr(doc, "metadata", {}) or {}
        hay = " ".join([
            str(meta.get("act_name", "")),
            str(meta.get("section_path", "")),
            str(meta.get("h1", "")),
            str(meta.get("h2", "")),
            str(meta.get("h3", "")),
            str(doc.page_content or ""),
        ]).lower()

        if acts:
            if not any(a in hay for a in acts):
                return False

        if sections:
            meta_sections = set(meta.get("sections") or [])
            if meta_sections and (meta_sections & sections):
                return True

            # Fallback to text match when section metadata is absent.
            if any(re.search(rf'\\b(?:section|sec\\.|s\\.)\\s*{re.escape(s)}\\b', hay, flags=re.IGNORECASE) for s in sections):
                return True

            # If sections were requested and we can't match them, treat as non-match.
            return False

        return True

    def _filter_docs_by_entities(self, docs: List[Document], entities: Dict[str, object]) -> List[Document]:
        if not docs or not entities or not self.enable_entity_prefilter:
            return docs

        filtered = [d for d in docs if self._doc_matches_entities(d, entities)]
        return filtered if filtered else docs

    def _generate_hyde_hypothesis(self, question: str) -> str:
        if not self.enable_hyde:
            return ""

        prompt = (
            "Write a short, plausible excerpt (80-140 words) that would appear in the source documents and answer the question. "
            "Use formal/legal phrasing and include likely Act/Section references if relevant. "
            "Return only the excerpt text.\n\n"
            f"Question: {question}\n\nExcerpt:"
        )
        try:
            txt = (self._safe_generate_content(prompt) or "").strip()
            return txt if txt and "Error:" not in txt else ""
        except Exception:
            return ""

    def _weighted_rrf_fuse(self, ranked_lists: List[List[Document]], weights: Tuple[float, ...], k: int = 60) -> List[Document]:
        scores: Dict[str, Dict] = {}
        for w, ranked in zip(weights, ranked_lists):
            if not ranked:
                continue
            for rank, doc in enumerate(ranked, start=1):
                key = doc.page_content
                if key not in scores:
                    scores[key] = {"doc": doc, "score": 0.0}
                scores[key]["score"] = round(scores[key]["score"] + (w * (1.0 / (k + rank))), 6)
        sorted_items = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in sorted_items]

    def _ensure_content_emb_map(self):
        if self._content_emb_map is not None:
            return
        if self.embeddings is None or not self.documents:
            self._content_emb_map = {}
            return
        try:
            self._content_emb_map = {doc.page_content: self.embeddings[i] for i, doc in enumerate(self.documents)}
        except Exception:
            self._content_emb_map = {}

    def _dedupe_docs(self, docs: List[Document], max_docs: int) -> List[Document]:
        if not docs:
            return docs

        # Always do exact-text dedupe for stability.
        seen = set()
        uniq = []
        for d in docs:
            key = re.sub(r"\\s+", " ", (d.page_content or "")).strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            uniq.append(d)
            if len(uniq) >= max_docs:
                break

        if not self.enable_similarity_dedupe or len(uniq) <= 1:
            return uniq

        self._ensure_content_emb_map()
        if not self._content_emb_map:
            return uniq

        selected = []
        selected_vecs = []
        for d in uniq:
            vec = self._content_emb_map.get(d.page_content)
            if vec is None:
                selected.append(d)
                continue

            if selected_vecs:
                mat = np.vstack(selected_vecs)
                sims = np.dot(mat, vec)
                if float(np.max(sims)) >= self.dedupe_sim_threshold:
                    continue

            selected.append(d)
            selected_vecs.append(vec)
            if len(selected) >= max_docs:
                break

        return selected

    def _rebuild_metadata_indexes(self):
        by_source: Dict[str, List[int]] = {}
        by_type: Dict[str, List[int]] = {}
        by_act: Dict[str, List[int]] = {}
        by_section: Dict[str, List[int]] = {}
        chunk_lookup: Dict[Tuple[str, int, str, int], Document] = {}

        acts = set()
        sections = set()

        for i, doc in enumerate(self.documents or []):
            meta = getattr(doc, "metadata", {}) or {}
            source = str(meta.get("source", "")).strip()
            dtype = str(meta.get("type", "")).strip()
            act_name = str(meta.get("act_name", "")).strip()
            sec_list = meta.get("sections") or []

            try:
                page_num = int(meta.get("page")) if meta.get("page") is not None else None
            except (TypeError, ValueError):
                page_num = None

            parent_sig = str(meta.get("parent_sig", "")).strip()
            chunk_index = meta.get("chunk_index")
            if source and page_num is not None and parent_sig and isinstance(chunk_index, int):
                chunk_lookup[(source.lower(), page_num, parent_sig, chunk_index)] = doc

            if source:
                by_source.setdefault(source.lower(), []).append(i)
            if dtype:
                by_type.setdefault(dtype.lower(), []).append(i)
            if act_name:
                by_act.setdefault(act_name.lower(), []).append(i)
                acts.add(act_name)
            for s in sec_list:
                s_norm = str(s).strip()
                if s_norm:
                    by_section.setdefault(s_norm.lower(), []).append(i)
                    sections.add(s_norm)

        self.metadata_index = {
            "by_source": by_source,
            "by_type": by_type,
            "by_act": by_act,
            "by_section": by_section,
        }
        self.gazetteer = {
            "acts": sorted(acts),
            "sections": sorted(sections, key=lambda x: (len(x), x)),
        }
        self.chunk_lookup = chunk_lookup

    def _candidate_indices_for_entities(self, entities: Dict[str, object]) -> Optional[np.ndarray]:
        if not entities or not self.metadata_index:
            return None

        acts = [a.lower() for a in (entities.get("acts") or []) if a]
        sections = [s.lower() for s in (entities.get("sections") or []) if s]

        act_sets = []
        for a in acts:
            idxs = self.metadata_index["by_act"].get(a)
            if idxs:
                act_sets.append(set(idxs))

        sec_sets = []
        for s in sections:
            idxs = self.metadata_index["by_section"].get(s)
            if idxs:
                sec_sets.append(set(idxs))

        # If both act and section are present, intersect for a hard filter.
        if act_sets and sec_sets:
            act_union = set().union(*act_sets)
            sec_union = set().union(*sec_sets)
            inter = sorted(act_union & sec_union)
            return np.array(inter, dtype=np.int32) if inter else None

        if act_sets:
            union = sorted(set().union(*act_sets))
            return np.array(union, dtype=np.int32) if union else None

        if sec_sets:
            union = sorted(set().union(*sec_sets))
            return np.array(union, dtype=np.int32) if union else None

        return None

    def _vector_search(self, query: str, candidate_indices: Optional[np.ndarray] = None, k: int = 50) -> List[Document]:
        # If we don't have embeddings in memory, fall back to FAISS retriever.
        if self.embeddings is None or not self.documents:
            return self._invoke_retriever(self.faiss_retriever, query)

        try:
            q_emb = self.embedding_model.encode([query], normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
            q = q_emb.reshape(-1)
        except Exception:
            return self._invoke_retriever(self.faiss_retriever, query)

        if candidate_indices is not None and len(candidate_indices) > 0:
            embs = self.embeddings[candidate_indices]
            sims = np.dot(embs, q)
            pairs = list(zip(candidate_indices.tolist(), sims.tolist()))
        else:
            sims = np.dot(self.embeddings, q)
            pairs = list(enumerate(sims.tolist()))

        pairs.sort(key=lambda x: (-x[1], x[0]))
        top = pairs[:k]
        return [self.documents[i] for i, _ in top]

    def _count_tokens(self, text: str) -> int:
        try:
            return int(self.conversation_manager.count_tokens(text or ""))
        except Exception:
            return len((text or "")) // 4

    def _get_sibling_chunks(self, meta: Dict[str, object]) -> Tuple[Optional[Document], Optional[Document]]:
        if not self.enable_sibling_injection or not self.chunk_lookup:
            return None, None

        source = str(meta.get("source", "")).strip().lower()
        try:
            page_num = int(meta.get("page")) if meta.get("page") is not None else None
        except (TypeError, ValueError):
            page_num = None
        parent_sig = str(meta.get("parent_sig", "")).strip()
        chunk_index = meta.get("chunk_index")
        if not source or page_num is None or not parent_sig or not isinstance(chunk_index, int):
            return None, None

        prev_doc = self.chunk_lookup.get((source, page_num, parent_sig, chunk_index - 1))
        next_doc = self.chunk_lookup.get((source, page_num, parent_sig, chunk_index + 1))
        return prev_doc, next_doc

    def _select_relevant_window(self, question: str, text: str) -> str:
        if not self.enable_parent_windowing:
            return text

        if self._count_tokens(text) <= self.parent_max_tokens:
            return text

        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return text

        try:
            q_emb = self.embedding_model.encode([question], normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True).reshape(-1)
            sent_embs = self.embedding_model.encode(sentences, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
            sims = np.dot(sent_embs, q_emb).reshape(-1)
            best = int(np.argmax(sims))
        except Exception:
            best = 0
            sims = np.zeros(len(sentences), dtype=np.float32)

        left = right = best
        cur_tokens = self._count_tokens(sentences[best])

        while cur_tokens < self.parent_max_tokens and (left > 0 or right < (len(sentences) - 1)):
            cand_left = sims[left - 1] if left > 0 else -1e9
            cand_right = sims[right + 1] if right < (len(sentences) - 1) else -1e9

            if cand_right >= cand_left and right < (len(sentences) - 1):
                next_tokens = self._count_tokens(sentences[right + 1])
                if cur_tokens + next_tokens > self.parent_max_tokens:
                    break
                right += 1
                cur_tokens += next_tokens
            elif left > 0:
                next_tokens = self._count_tokens(sentences[left - 1])
                if cur_tokens + next_tokens > self.parent_max_tokens:
                    break
                left -= 1
                cur_tokens += next_tokens
            else:
                break

        return " ".join(sentences[left : right + 1]).strip()

    def _assemble_parent_context(self, question: str, doc: Document) -> str:
        meta = getattr(doc, "metadata", {}) or {}
        source = str(meta.get("source", "Unknown"))
        page = meta.get("page", "?")
        section_path = str(meta.get("section_path", "")).strip()
        header = f"[{normalize_source_name(source)}, Page {page}]"
        if section_path:
            header = header + f" {section_path}"

        parent_text = meta.get("parent_text", getattr(doc, "page_content", "")) or ""
        parent_text = self._select_relevant_window(question, str(parent_text))

        prev_doc, next_doc = self._get_sibling_chunks(meta)
        sibling_texts = []
        if prev_doc is not None:
            sibling_texts.append(prev_doc.page_content)
        if next_doc is not None:
            sibling_texts.append(next_doc.page_content)

        if sibling_texts:
            parent_text = (parent_text + "\n\n[Adjacent chunks]\n" + "\n\n".join(sibling_texts)).strip()

        return (header + "\n" + parent_text).strip()

    def _select_final_context_docs(
        self,
        question: str,
        filtered: List[Tuple[Document, float]],
        limit: int,
    ) -> List[Document]:
        if not filtered:
            return []

        seen_parents = set()
        used_sources = set()
        selected: List[Document] = []
        fallback: List[Document] = []

        for doc, score in filtered:
            metadata = getattr(doc, "metadata", {}) or {}
            parent_text = metadata.get("parent_text", getattr(doc, "page_content", ""))
            if not parent_text or parent_text in seen_parents:
                continue

            seen_parents.add(parent_text)
            parent_doc = Document(
                page_content=self._assemble_parent_context(question, doc),
                metadata=metadata.copy()
            )
            fallback.append(parent_doc)

            if not self.enable_source_diversity:
                selected.append(parent_doc)
            else:
                source_key = str(metadata.get("source", "")).strip().lower() or f"__src_{len(fallback)}"
                if source_key not in used_sources:
                    used_sources.add(source_key)
                    selected.append(parent_doc)

            if len(selected) >= limit:
                return selected[:limit]

        if len(selected) < limit:
            seen_contents = {d.page_content for d in selected}
            for doc in fallback:
                if doc.page_content in seen_contents:
                    continue
                selected.append(doc)
                seen_contents.add(doc.page_content)
                if len(selected) >= limit:
                    break

        return selected[:limit]

    def _log_query_event(self, payload: Dict[str, object]):
        if not self.enable_eval_logging:
            return

        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            log_dir = os.path.join(base_dir, "rag_logs")
            os.makedirs(log_dir, exist_ok=True)
            path = os.path.join(log_dir, "pipeline1.jsonl")

            event = {
                "ts": float(time.time()),
                **(payload or {}),
            }
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        except Exception:
            # Never fail the user request due to logging.
            pass

    def _rrf_fuse(self, ranked_lists: List[List[Document]], k: int = 60) -> List[Document]:
        """
        Reciprocal Rank Fusion: combine multiple ranked lists of documents.
        Each document's fused score = sum(1 / (k + rank_i)) across all lists.
        Deterministic: RRF score is purely arithmetic, no randomness.
        """
        scores: Dict[str, Dict] = {}
        for ranked in ranked_lists:
            for rank, doc in enumerate(ranked, start=1):
                key = doc.page_content  # use content as identity
                if key not in scores:
                    scores[key] = {'doc': doc, 'score': 0.0}
                # Use 6 decimal places for RRF arithmetic stability
                scores[key]['score'] = round(scores[key]['score'] + 1.0 / (k + rank), 6)
        sorted_items = sorted(scores.values(), key=lambda x: x['score'], reverse=True)
        return [item['doc'] for item in sorted_items]

    def _expand_and_retrieve(
        self,
        question: str,
        top_k: int = 15,
        entities: Optional[Dict[str, object]] = None,
        retrieval_depth: Optional[int] = None,
    ) -> List[Document]:
        """
        Multi-query retrieval with RRF fusion.
        Runs retrieval for each query expansion and merges results so that
        the same document surfaces regardless of how the question was phrased.
        """
        from concurrent.futures import ThreadPoolExecutor

        # Legacy path (kept for safety/compatibility)
        if not self.enable_enhanced_retrieval:
            queries = self._expand_queries(question)
            logger.info(f"Multi-query retrieval: {len(queries)} query variants (Parallel)")

            all_ranked_lists = []
            unique_queries = list(dict.fromkeys([q.lower() for q in queries]))  # dedupe

            def _get_docs(q):
                try:
                    msg = f"  Query variant '{q[:60]}'"
                    docs = self.hybrid_retriever.invoke(q)
                    if docs:
                        logger.info(f"{msg} returned {len(docs)} docs")
                        return docs
                except Exception as e:
                    logger.warning(f"Retrieval failed for query variant '{q}': {e}")
                return []

            with ThreadPoolExecutor(max_workers=len(unique_queries)) as executor:
                results = list(executor.map(_get_docs, unique_queries))

            all_ranked_lists = [r for r in results if r]
            if not all_ranked_lists:
                return []

            fused = self._rrf_fuse(all_ranked_lists)
            logger.info(f"RRF fusion produced {len(fused)} unique docs from {len(all_ranked_lists)} query variants")
            return fused[:60]

        # Enhanced additive path: semantic + BM25 + HyDE, fused with weighted RRF.
        queries = self._expand_queries(question)
        entities = entities or self._extract_query_entities(question)
        queries = self._boost_queries_with_entities(question, queries, entities)
        effective_depth = retrieval_depth or self.retrieval_depth
        candidate_indices = self._candidate_indices_for_entities(entities) if (entities and self.enable_entity_prefilter) else None
        logger.info(f"Enhanced retrieval: {len(queries)} query variants (+HyDE={self.enable_hyde})")

        # Dedupe variants but keep original casing/spacing for retrieval.
        unique_queries = []
        seen = set()
        for q in queries:
            key = (q or "").strip().lower()
            if key and key not in seen:
                seen.add(key)
                unique_queries.append(q)

        def _get_pair(q):
            faiss_docs = self._vector_search(q, candidate_indices=candidate_indices, k=effective_depth) if candidate_indices is not None else self._invoke_retriever(self.faiss_retriever, q)
            bm25_docs = self._invoke_retriever(self.bm25_retriever, q)
            return faiss_docs, bm25_docs

        faiss_lists: List[List[Document]] = []
        bm25_lists: List[List[Document]] = []
        if unique_queries:
            with ThreadPoolExecutor(max_workers=min(len(unique_queries), 8)) as executor:
                pairs = list(executor.map(_get_pair, unique_queries))
            for faiss_docs, bm25_docs in pairs:
                if faiss_docs:
                    faiss_lists.append(faiss_docs)
                if bm25_docs:
                    bm25_lists.append(bm25_docs)

        semantic_fused = self._rrf_fuse(faiss_lists, k=60) if faiss_lists else []
        bm25_fused = self._rrf_fuse(bm25_lists, k=60) if bm25_lists else []

        hyde_docs: List[Document] = []
        use_hyde = self._should_use_hyde(question, self.last_intent or "factual", entities)
        hyde_text = self._generate_hyde_hypothesis(question) if use_hyde else ""
        if hyde_text:
            hyde_docs = self._vector_search(hyde_text, candidate_indices=candidate_indices, k=effective_depth) if candidate_indices is not None else self._invoke_retriever(self.faiss_retriever, hyde_text)

        if entities and self.enable_entity_prefilter:
            semantic_fused = self._filter_docs_by_entities(semantic_fused, entities)
            bm25_fused = self._filter_docs_by_entities(bm25_fused, entities)
            hyde_docs = self._filter_docs_by_entities(hyde_docs, entities)

        fused = self._weighted_rrf_fuse(
            [semantic_fused, bm25_fused, hyde_docs],
            weights=self.rrf_weights,
            k=60,
        )

        fused = self._dedupe_docs(fused, max_docs=self.max_candidate_pool)
        logger.info(
            f"Enhanced fusion: semantic={len(semantic_fused)} bm25={len(bm25_fused)} hyde={len(hyde_docs)} -> fused={len(fused)}"
        )
        return fused[: self.max_candidate_pool]

    def query(self, question: str, top_k: int = 15, stream_callback=None, **kwargs) -> str:
        """Processes a query and returns the full answer string"""
        if self.hybrid_retriever is None:
            raise ValueError("Index not loaded. Call load_index() or build_index() first.")

        # Ensure top_k is int
        try:
            top_k = int(top_k)
        except (ValueError, TypeError):
            top_k = 15

        intent = self._classify_intent(question) if self.enable_intent_classification else "factual"
        self.last_intent = intent

        question_for_retrieval = self._rewrite_query_with_history(question)
        entities = self._extract_query_entities(question_for_retrieval)
        retrieval_cfg = self._adaptive_retrieval_settings(question_for_retrieval, intent, entities)

        # Decompose only when the question is genuinely complex; skip the extra LLM call for simple factual questions.
        if self._should_decompose_query(question_for_retrieval, intent):
            sub_queries = self._decompose_query(question_for_retrieval)[: self.max_sub_queries]
        else:
            sub_queries = [question_for_retrieval]
        
        # ── STEP 1: Multi-query retrieval with RRF fusion ─────────────────────
        # Expand and retrieve for each sub-query
        sub_results = []
        for sq in sub_queries:
            sq_entities = self._extract_query_entities(sq)
            merged_entities = {
                "acts": list(dict.fromkeys((entities.get("acts") or []) + (sq_entities.get("acts") or []))),
                "sections": list(dict.fromkeys((entities.get("sections") or []) + (sq_entities.get("sections") or []))),
                "years": list(dict.fromkeys((entities.get("years") or []) + (sq_entities.get("years") or []))),
                "dates": list(dict.fromkeys((entities.get("dates") or []) + (sq_entities.get("dates") or []))),
            }
            res = self._expand_and_retrieve(
                sq,
                top_k=top_k,
                entities=merged_entities,
                retrieval_depth=retrieval_cfg["retrieval_depth"],
            )
            if res:
                sub_results.append(res)
        
        # Fusion of sub-query results
        if len(sub_results) > 1:
            retrieved_docs = self._rrf_fuse(sub_results, k=60)
        elif sub_results:
            retrieved_docs = sub_results[0]
        else:
            retrieved_docs = []

        # ── STEP 2: Re-rank fused pool with CrossEncoder ──────────────────────
        # Expand pool for the relevance checker (cap before reranking)
        top_docs = retrieved_docs[: retrieval_cfg["candidate_pool"]] if retrieved_docs else []
        original_max_docs = self.relevance_checker.max_docs
        self.relevance_checker.max_docs = retrieval_cfg["rerank_max_docs"]
        try:
            filtered: List[Tuple[Document, float]] = self.relevance_checker.filter_documents(question_for_retrieval, top_docs)
        finally:
            self.relevance_checker.max_docs = original_max_docs
        
        # ── STEP 2.5: Parent Expansion (Hierarchical Retrieval) ──────────────
        filtered_docs = self._select_final_context_docs(
            question_for_retrieval,
            filtered,
            retrieval_cfg["final_context_docs"],
        )
        if not filtered_docs:
            filtered_docs = [d for d, s in filtered[: retrieval_cfg["final_context_docs"]]]

        # Capture filtered_docs for UI references
        self.last_retrieved_docs = filtered_docs

        # ── STEP 3: Generate answer via deep research ReAct loop ──────────────
        # Detect informative intent keywords in the question
        informative_keywords = [
            "informative answer", "informative", "explain in detail", "in detail",
            "comprehensive", "elaborate", "describe", "give details", "detailed"
        ]
        q_lower = question.lower()
        is_informative = any(kw in q_lower for kw in informative_keywords)
        
        # ── FALLBACK: If retrieval is sparse, use Fast Track (Direct) loop to save time ──
        if is_informative and len(filtered_docs) < 3:
            logger.info(f"Sparse retrieval ({len(filtered_docs)} docs). Falling back to Fast Track for accuracy.")
            is_informative = False

        if is_informative:
            augmented_question = (
                f"[RESEARCH DIRECTIVE: Provide a comprehensive, informative answer of 200-300 words. "
                f"Use structured headings, bullets with explanations, tables where relevant, and cite all sources.] "
                f"{question}"
            )
        else:
            augmented_question = question

        full_answer = self._generate_answer(
            augmented_question, 
            filtered_docs, 
            is_informative=is_informative, 
            stream_callback=stream_callback
        )

        # Store this exchange in history once complete
        self.conversation_manager.add_exchange(question, full_answer.strip())

        # Log conversation stats
        stats = self.conversation_manager.get_stats()
        logger.info(f"Query complete. Total exchanges: {stats['total_exchanges']}")

        try:
            used_sources = []
            for d in (self.last_retrieved_docs or []):
                m = getattr(d, "metadata", {}) or {}
                used_sources.append({
                    "source": normalize_source_name(m.get("source", "Unknown")),
                    "page": m.get("page", "?"),
                    "type": m.get("type", "text"),
                    "section_path": m.get("section_path", ""),
                })

            self._log_query_event({
                "question": question,
                "question_for_retrieval": question_for_retrieval,
                "intent": intent,
                "entities": entities,
                "retrieval_cfg": retrieval_cfg,
                "retrieved_docs": len(retrieved_docs or []),
                "final_context_docs": len(self.last_retrieved_docs or []),
                "sources": used_sources[:10],
            })
        except Exception:
            pass
        return full_answer


    def set_history(self, messages: List[Dict[str, str]]):
        """Synchronize conversation history"""
        self.conversation_manager.set_history(messages)

    def _generate_answer(self, question: str, context_docs: List[Document], is_informative: bool = False, stream_callback=None) -> str:
        """Generates answer using the Enhanced Manual ReAct Loop — Deep Research Mode"""

        format_instructions = self._answer_format_instructions(question, self.last_intent or "factual", is_informative=is_informative)

        FEW_SHOT_EXAMPLE = """
Example Reasoning Cycle:
Question: What are the specific protection rules for the Muturajawela wetland?
Thought: I have been provided with initial context chunks. I will review them first.
Reflection: Confidence=6/10. Document 1 (Muturajawela Management Plan) contains specific protection rules. I will verify the prohibition of construction before answering.
Action: verify_answer({"statement": "Zone A of Muturajawela prohibits construction and landfilling", "source_hint": "Muturajawela Management Plan"})
Observation: [verify_answer]: Verdict: SUPPORTED | Confidence: 9/10
Reflection: Confidence=9/10. Verified. Ready to write Final Answer with citations from the provided context.
Final Answer:
According to the **Muturajawela Management Plan** (p. 4):
- **Zone A** is strictly protected — construction and landfilling are prohibited [Muturajawela Management Plan, p. 4]

**Sources Used:**
- Muturajawela Management Plan, p. 4
        """
        # ── Format hierarchical context for injecting into the prompt ──────────
        formatted_docs = ""
        if context_docs:
            for i, doc in enumerate(context_docs):
                source = normalize_source_name(doc.metadata.get("source", "Unknown"))
                page = doc.metadata.get("page", "?")
                # Use "PARENT BLOCK" label to indicate high-precision grounding
                formatted_docs += f"--- HIGH-PRECISION CONTEXT BLOCK {i+1} [{source}, Page {page}] ---\n{doc.page_content}\n\n"
        else:
            formatted_docs = "No initial context blocks provided. Use tools to find information."

        # ── HIGH ACCURACY FAST PROMPT ──
        system_prompt = f"""You are 'Marsh Fast', an elite AI accuracy-first assistant.
You have been enhanced with an **Advanced High-Precision RAG Workflow** (Intent Decomposition -> Hierarchical Retrieval -> Cross-Encoder Reranking).

Your ABSOLUTE TOP PRIORITY is FATUAL ACCURACY and PINPOINT CITATION. You must be precise, fast, and 100% grounded.

═══════════════════════════════════════════════════════════════
HIGH-PRECISION PROTOCOL
═══════════════════════════════════════════════════════════════
1. ACCURACY FIRST: Never guess. Use exact names of Acts, section numbers, and monetary values.
2. CITATION RIGOR: Every sentence in the Final Answer must end with a citation like [Document Name, Page X]. No exceptions.
3. HYBRID RESEARCH: Cross-reference findings across the retrieved chunks. If evidence conflicts, explicitly flag the conflict.
4. CLAIM LISTING: Before writing the Final Answer, list EVERY claim you intend to make (numbers, dates, legal references, obligations, penalties) as a checklist.
5. VERIFY: Call **verify_answer** for EVERY numerical or legal claim in your checklist. Use **lookup_section** for Act+Section exact text when relevant. Use **cross_reference** for high-stakes claims that may conflict across sources.
6. FAITHFULNESS: If you cannot ground a sentence in the retrieved context, do not write it. Instead say what is missing and what document/section would be needed.
7. PRE-RETRIEVED CONTEXT: Use the provided context blocks as primary truth. Only call tools if context is insufficient or to verify a critical claim.

        OUTPUT FORMAT:
        - Precise, technical, and formatted for clarity.
        - Use **BOLD** for all Acts, sections, and penalties.
        - Citations must be visible: [Document, Page].

        {format_instructions}

FORMAT for reasoning:
Thought: [Reasoning]
Action: tool_name({{"arg": "val"}})
Observation: [Result]
Reflection: [Confidence]
...
Final Answer: [Precise result with citations]

# ── INITIAL RESEARCH RESULTS ──
The following documents were pre-retrieved and reranked for this question:
{formatted_docs}

{FEW_SHOT_EXAMPLE}
Begin!
"""

        history = self.conversation_manager.get_history()
        full_context = f"SYSTEM: {system_prompt}\n\n"
        for msg in history:
            role = "USER" if msg["role"] == "user" else "ASSISTANT"
            full_context += f"{role}: {msg['content']}\n"
        full_context += f"USER: {question}\n"

        current_prompt = full_context
        if self.enable_direct_grounded_answer and not is_informative and context_docs:
            allow_uncited_fallback = "gpt-oss" in (self.model_name or "").lower()
            direct_answer = self._generate_direct_grounded_answer(
                question,
                context_docs,
                stream_callback=stream_callback,
                allow_uncited_fallback=allow_uncited_fallback,
            )
            if direct_answer:
                return direct_answer

        if not is_informative and "gpt-oss" in (self.model_name or "").lower():
            return self._generate_direct_grounded_answer(
                question,
                context_docs,
                stream_callback=stream_callback,
                allow_uncited_fallback=True,
            ) or "I'm sorry, I couldn't generate a grounded answer from the available context."

        # Use dynamic iterations: shorter fast track, deeper informative path.
        max_iterations = self.research_iterations if is_informative else self.fast_track_iterations
        iteration = 0
        last_action = None
        tool_call_count = 0
        self_critique_done = False

        try:
            while iteration < max_iterations:
                iteration += 1
                logger.info(f"ReAct Iteration {iteration}/{max_iterations} for {self.model_name}")

                # ── Inject mandatory self-critique after 4+ tool calls ──────────
                if tool_call_count >= 4 and not self_critique_done:
                    self_critique_done = True
                    current_prompt += (
                        "\nSYSTEM INTERRUPT: You have made 4+ tool calls. "
                        "Write a SELF-CRITIQUE before continuing:\n"
                        "- List each confirmed fact with its source name [Source]\n"
                        "- List what is still missing (e.g. specific penalty values, dates)\n"
                        "- State your next specific research goal.\n"
                        "SELF-CRITIQUE: "
                    )

                # ── Final Call Nudge (Iteration 13) ───────────────────────────
                if iteration == 13:
                    current_prompt += (
                        "\nSYSTEM FINAL CALL: You are reaching the research limit. "
                        "Synthesize all gathered information now. If enough info exists, "
                        "write your Final Answer with full citations. If not, state what is missing.\n"
                    )

                # Call model with automatic rotation on quota limits
                full_llm_output = ""
                try:
                    full_llm_output = self._safe_generate_content(current_prompt, stream_callback=stream_callback)
                except Exception as e:
                    logger.error(f"Failed to generate content after rotation attempts: {e}")
                    return (
                        f"I encountered an error during research: {str(e)}. "
                        "This may happen if the content is flagged as sensitive or there is a connection issue."
                    )
                
                if not full_llm_output:
                    return "I'm sorry, I couldn't generate a response. Please try again."
                
                logger.info(f"Model response received ({len(full_llm_output)} chars)")

                lower_output = full_llm_output.lower()

                # ── Final Answer detection ──────────────────────────────────────
                if "final answer" in lower_output:
                    marker_match = re.search(r"final answer:?", full_llm_output, re.IGNORECASE)
                    if marker_match:
                        ans = full_llm_output[marker_match.end():].strip()
                        if ans:
                            logger.info(f"Final Answer found at iteration {iteration}")
                            return self._strip_react_trace(ans)

                # ── Fallback: direct output in later iterations ─────────────────
                if iteration >= 7 and not any(
                    m in lower_output for m in ["action:", "thought:", "reflection:", "self-critique"]
                ):
                    if len(full_llm_output) > 100:
                        logger.info("Assuming direct output is Final Answer (no markers, iter>=7).")
                        return self._strip_react_trace(full_llm_output.strip())

                # ── Tool call detection ─────────────────────────────────────────
                if "action" in lower_output:
                    # 1. Try strict end-anchored match first
                    action_match = re.search(
                        r"action:?\s*(\w+)\s*\((.*?)\)\s*$",
                        full_llm_output,
                        re.IGNORECASE | re.DOTALL | re.MULTILINE
                    )
                    # 2. Try greedy fallback if strict fails
                    if not action_match:
                        action_match = re.search(
                            r"action:?\s*(\w+)\s*\((.*)\)",
                            full_llm_output,
                            re.IGNORECASE | re.DOTALL
                        )

                    if action_match:
                        tool_name = action_match.group(1).strip()
                        args_str = action_match.group(2).strip()
                        observation = ""

                        try:
                            # ── Smart Argument Parsing (Handles JSON and Python-style) ─────
                            tool_args = {}
                            try:
                                clean_json = args_str
                                if '```' in clean_json:
                                    json_match = re.search(r'(\{.*\})', clean_json, re.DOTALL)
                                    if json_match: clean_json = json_match.group(1)
                                tool_args = json.loads(clean_json or "{}")
                            except (json.JSONDecodeError, ValueError):
                                # Fallback: Extract key-value pairs
                                logger.warning(f"JSON parse failed for args: {args_str}. Trying regex fallback.")
                                kv_pairs = re.findall(r'(\w+)\s*=\s*(?:"(.*?)"|\'(.*?)\'|(\d+))', args_str)
                                for k, v1, v2, v3 in kv_pairs:
                                    val = v1 or v2 or v3
                                    if v3: val = int(v3)
                                    tool_args[k] = val
                                if not tool_args and args_str:
                                    tool_args = {"query": args_str.strip("\"' }")}

                            current_action = f"{tool_name}({tool_args})"

                            if current_action == last_action:
                                observation = (
                                    "Error: You already performed this exact action. "
                                    "Try a different phrasing, call get_document_list, "
                                    "or write your Final Answer."
                                )
                            else:
                                last_action = current_action
                                tool_call_count += 1
                                try:
                                    tool_result = self.tool_executor.execute_tool(tool_name, tool_args)
                                    observation = format_tool_result_for_prompt(tool_name, tool_result)
                                    
                                    if ("No relevant documents found" in observation 
                                        or "No content found" in observation):
                                        observation += (
                                            "\n[System Hint]: No results. Try broader keywords or check document names."
                                        )
                                except Exception as e:
                                    observation = f"Error executing tool {tool_name}: {str(e)}"
                        except Exception as e:
                            logger.error(f"Action processing failed: {e}")
                            observation = f"Error: Could not process arguments '{args_str}'."

                        current_prompt += f"\n{full_llm_output}\nObservation: {observation}\n"
                        continue

                # No recognized marker — nudge the model ──────────────────────
                current_prompt += (
                    f"\n{full_llm_output}\n"
                    "Thought: I have not yet provided a Final Answer or called a tool correctly. "
                    "I must proceed with the next step in the ReAct cycle."
                )

            return (
                "The analysis required more steps than available. "
                "Please try a more specific question or break it into smaller parts."
            )

        except Exception as e:
            logger.error(f"Generation error: {e}")
            return f"Error: {str(e)}"

    def _generate_direct_grounded_answer(self, question: str, context_docs: List[Document], stream_callback=None, allow_uncited_fallback: bool = False) -> str:
        formatted_docs = []
        for i, doc in enumerate(context_docs[: self.final_context_docs], start=1):
            meta = getattr(doc, "metadata", {}) or {}
            source = normalize_source_name(meta.get("source", "Unknown"))
            page = meta.get("page", "?")
            formatted_docs.append(f"[Context {i}: {source}, Page {page}]\n{doc.page_content}")

        format_instructions = self._answer_format_instructions(question, self.last_intent or "factual", is_informative=False)

        prompt = (
            "You are Marsh Fast. Answer the user directly using only the provided context.\n"
            "Rules:\n"
            "- Be complete, precise, and directly answer every part of the question.\n"
            "- Every factual sentence must include a citation in the form [Document, Page X].\n"
            "- If the context is insufficient, say exactly what is missing.\n"
            "- Cover the important details before concluding. Do not stop at a partial answer.\n"
            "- Follow the FORMAT RULES exactly. Use bullets, tables, or prose according to the user's query shape.\n"
            "- Do not reveal reasoning. Do not use Thought, Action, Observation, Reflection, or Final Answer labels.\n\n"
            f"{format_instructions}\n"
            "Context:\n"
            f"{chr(10).join(formatted_docs)}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

        try:
            answer = (self._safe_generate_content(prompt, max_retries=3, stream_callback=stream_callback) or "").strip()
        except Exception:
            return ""

        if not answer or "Error:" in answer:
            return ""

        cleaned = self._strip_react_trace(answer)
        if len(cleaned) < 40:
            return ""
        if "[" not in cleaned or "Page" not in cleaned:
            return cleaned if allow_uncited_fallback else ""
        return cleaned

    def _safe_generate_content(self, prompt: str, max_retries: int = 5, stream_callback=None) -> str:
        """Helper to generate content with automatic API key rotation and real-time streaming support"""
        retries = 0
        while retries < max_retries:
            try:
                stream_enabled = bool(stream_callback)
                response = self.llm_client.generate_content(prompt, stream=stream_enabled)
                
                # Check for empty response or safety blocks
                if not response:
                    logger.warning("Empty response object received from Gemini.")
                    return "I'm sorry, I cannot provide an answer due to an empty model response."
                
                # Success - mark key as successful if using rotator
                if (not self.use_local_groq_rotation) and self.gemini_rotator is not None and self.current_key_idx is not None:
                    self.gemini_rotator.mark_key_success(self.current_key_idx)
                
                # Safely extract text (Gemini raises ValueError if parts are empty even on finish_reason=1)
                if stream_enabled:
                    full_text = ""
                    is_streaming_to_ui = False
                    for chunk in response:
                        if hasattr(chunk, 'text') and chunk.text:
                            text_chunk = chunk.text
                            full_text += text_chunk
                            if not is_streaming_to_ui:
                                marker_match = re.search(r"final answer:?\s*", full_text, re.IGNORECASE)
                                if marker_match:
                                    is_streaming_to_ui = True
                                    after_marker = full_text[marker_match.end():]
                                    if after_marker:
                                        stream_callback(after_marker)
                            else:
                                stream_callback(text_chunk)
                    if not full_text:
                        logger.warning("Gemini stream returned no text parts; delivering empty-response notice.")
                        return "I'm sorry, I couldn't generate a response this time. Please try rephrasing."
                    return full_text
                else:
                    try:
                        return response.text
                    except ValueError as ve:
                        fallback = self._extract_text_from_candidates(response)
                        if fallback:
                            logger.warning("Gemini returned empty text accessor; using candidate parts as fallback.")
                            return fallback
                        logger.warning(f"Gemini returned no text parts. Error: {ve}")
                        return "I'm sorry, I couldn't generate a response this time. Please try rephrasing."
                
            except Exception as e:
                error_str = str(e)
                # Check for quota/rate limit errors (429) or service unavailable (503/500)
                is_retryable = (
                    "429" in error_str or 
                    "503" in error_str or 
                    "500" in error_str or
                    "quota" in error_str.lower() or 
                    "limit" in error_str.lower() or
                    "unavailable" in error_str.lower() or
                    "empty response" in error_str.lower()
                )
                
                if is_retryable and self.gemini_rotator:
                    msg = f"⚠️ [Marsh Fast] API error hit for Gemini Key #{self.current_key_idx + 1}: {error_str}. Rotating key..."
                    print(f"\n{msg}")
                    logger.warning(msg)
                    
                    # Mark current key as failed (temporary)
                    if self.current_key_idx is not None:
                        self.gemini_rotator.mark_key_failed(self.current_key_idx)
                    
                    # Get next key and reconfigure
                    idx, key = self.gemini_rotator.get_next_key()
                    genai.configure(api_key=key)
                    self.current_key_idx = idx
                    
                    masked_key = f"{key[:4]}...{key[-4:]}" if len(key) > 8 else "****"
                    print(f"🔃 [Marsh Fast] Switched to Google Gemini API Key #{idx + 1} ({masked_key})")
                    
                    # Re-initialize the model with the new key (keep deterministic settings)
                    safety_settings = [
                        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                    ]
                    self.llm_client = genai.GenerativeModel(
                        model_name=self.model_name,
                        generation_config={
                            "temperature": 0.0,
                            "top_p": 1.0,
                            "top_k": 1,
                            "max_output_tokens": 4500,
                        },
                        safety_settings=safety_settings
                    )
                    
                    retries += 1
                    continue
                else:
                    # Some other non-quota error
                    logger.error(f"Unexpected error in _safe_generate_content: {e}")
                    raise e
        
        raise Exception("Exceeded max retries for Gemini API key rotation")

    def _safe_generate_content(self, prompt: str, max_retries: int = 5, stream_callback=None) -> str:
        """Helper to generate content with Groq API key rotation and optional streaming support."""
        configured_attempts = self.groq_retry_max_attempts if self.groq_retry_max_attempts > 0 else 0
        pool_size = max(1, len(getattr(self, "groq_api_keys", []) or []))
        effective_attempts = max(max_retries, configured_attempts, pool_size * 3)
        retries = 0
        last_error = None
        prompt_to_send = prompt
        toolless_retry_used = False
        while retries < effective_attempts:
            try:
                stream_enabled = bool(stream_callback)
                response = requests.post(
                    self.groq_api_base,
                    headers=self._groq_headers(),
                    json=self._groq_payload(prompt_to_send, stream=stream_enabled),
                    timeout=(30, 180),
                    stream=stream_enabled,
                )

                if response.status_code >= 400:
                    try:
                        detail = response.json()
                    except Exception:
                        detail = response.text
                    raise Exception(f"{response.status_code} {detail}")

                if self.gemini_rotator is not None and self.current_key_idx is not None:
                    self.gemini_rotator.mark_key_success(self.current_key_idx)

                if stream_enabled:
                    full_text = self._stream_groq_response(response, stream_callback)
                    if not full_text:
                        logger.warning("Groq stream returned no text parts; delivering empty-response notice.")
                        return "I'm sorry, I couldn't generate a response this time. Please try rephrasing."
                    return full_text

                data = response.json()
                text = self._extract_groq_text(data)
                if text:
                    return text
                logger.warning("Groq returned no text parts; delivering empty-response notice.")
                return "I'm sorry, I couldn't generate a response this time. Please try rephrasing."

            except Exception as e:
                last_error = e
                error_str = str(e)
                error_lower = error_str.lower()
                if ("tool_use_failed" in error_lower or "tool choice is none, but model called a tool" in error_lower) and not toolless_retry_used:
                    toolless_retry_used = True
                    prompt_to_send = self._make_toolless_retry_prompt(prompt)
                    logger.warning("Groq attempted a tool call without tools configured. Retrying once with a plain-text-only prompt.")
                    continue

                is_retryable = (
                    "429" in error_str or
                    "500" in error_str or
                    "502" in error_str or
                    "503" in error_str or
                    "504" in error_str or
                    "quota" in error_lower or
                    "rate" in error_lower or
                    "limit" in error_lower or
                    "timeout" in error_lower or
                    "unavailable" in error_lower or
                    "connection" in error_lower
                )

                if is_retryable:
                    retries += 1
                    rotated = self._rotate_groq_key()
                    sleep_s = min(12.0, self.groq_retry_base_delay * (2 ** max(0, retries - 1)))
                    if not rotated:
                        logger.warning(
                            f"Groq retryable error without fresh key rotation available. "
                            f"Retrying current key after {sleep_s:.1f}s ({retries}/{effective_attempts})."
                        )
                    else:
                        logger.warning(
                            f"Groq retryable error. Rotated key and retrying after {sleep_s:.1f}s "
                            f"({retries}/{effective_attempts})."
                        )
                    time.sleep(sleep_s)
                    continue

                logger.error(f"Unexpected error in _safe_generate_content: {e}")
                raise e

        raise Exception(
            f"Exceeded max retries for Groq API key rotation after {effective_attempts} attempts. "
            f"Last error: {last_error}"
        )

    @staticmethod
    def _make_toolless_retry_prompt(prompt: str) -> str:
        return (
            "SYSTEM OVERRIDE: Tools are unavailable for this request.\n"
            "Do NOT call any tool, function, action, JSON schema, or structured tool interface.\n"
            "Respond with plain text only.\n"
            "If the original prompt asked for Thought, Action, Observation, or Final Answer formatting, ignore the tool-call parts "
            "and provide only the final plain-text answer grounded in the provided context.\n\n"
            f"{prompt}"
        )

    def _collect_groq_api_keys(self, model_params: dict) -> List[str]:
        keys: List[str] = []
        explicit_keys = model_params.get("groq_api_keys") or []
        if isinstance(explicit_keys, str):
            explicit_keys = [explicit_keys]
        for key in explicit_keys:
            if key and "your_groq_api_key" not in key.lower() and key not in keys:
                keys.append(key)

        for single_key_name in ("groq_api_key", "google_api_key"):
            key = (model_params.get(single_key_name) or "").strip()
            if key and "your_groq_api_key" not in key.lower() and key not in keys:
                keys.append(key)

        for env_name, env_value in sorted(os.environ.items()):
            if env_name.upper().startswith("GROQ_API_KEY") and env_value and "your_groq_api_key" not in env_value.lower() and env_value not in keys:
                keys.append(env_value)
        return keys

    def _groq_headers(self) -> Dict[str, str]:
        if not self.current_api_key:
            raise ValueError("No Groq API key is configured for RAGPipeline1.")
        return {
            "Authorization": f"Bearer {self.current_api_key}",
            "Content-Type": "application/json",
        }

    def _groq_payload(self, prompt: str, stream: bool = False) -> Dict:
        return {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": self.groq_max_tokens,
            "stream": stream,
        }

    def _stream_groq_response(self, response, stream_callback) -> str:
        full_text = ""
        is_streaming_to_ui = False
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            line = raw_line.strip()
            if not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if data_str == "[DONE]":
                break
            try:
                payload = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            choices = payload.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            text_chunk = delta.get("content") or ""
            if not text_chunk:
                continue
            full_text += text_chunk
            if not is_streaming_to_ui:
                marker_match = re.search(r"final answer:?\s*", full_text, re.IGNORECASE)
                if marker_match:
                    is_streaming_to_ui = True
                    after_marker = full_text[marker_match.end():]
                    if after_marker:
                        stream_callback(after_marker)
            else:
                stream_callback(text_chunk)
        return full_text

    def _extract_groq_text(self, response_json: Dict) -> str:
        choices = response_json.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        return (message.get("content") or "").strip()

    def _rotate_groq_key(self) -> bool:
        current_display_idx = (self.current_key_idx + 1) if self.current_key_idx is not None else "?"

        groq_keys = getattr(self, "groq_api_keys", []) or []
        if len(groq_keys) > 1:
            next_idx = 0 if self.current_key_idx is None else (self.current_key_idx + 1) % len(groq_keys)
            if next_idx == self.current_key_idx:
                return False
            self.current_key_idx = next_idx
            self.current_api_key = groq_keys[next_idx]
            masked_key = f"{self.current_api_key[:4]}...{self.current_api_key[-4:]}" if len(self.current_api_key) > 8 else "****"
            print(f"Switched RAGPipeline1 to Groq API Key #{next_idx + 1} ({masked_key})")
            return True

        if (not self.use_local_groq_rotation) and self.gemini_rotator and hasattr(self.gemini_rotator, "get_next_key"):
            logger.warning(f"[RAGPipeline1] API error hit for Groq Key #{current_display_idx}. Rotating key...")
            if self.current_key_idx is not None and hasattr(self.gemini_rotator, "mark_key_failed"):
                self.gemini_rotator.mark_key_failed(self.current_key_idx)
            idx, key = self.gemini_rotator.get_next_key()
            self.current_key_idx = idx
            self.current_api_key = key
            masked_key = f"{key[:4]}...{key[-4:]}" if len(key) > 8 else "****"
            print(f"Switched RAGPipeline1 to Groq API Key #{idx + 1} ({masked_key})")
            return True

        return False

    @staticmethod
    def _extract_text_from_candidates(response) -> str:
        """Best-effort extraction when response.text raises ValueError."""
        try:
            candidates = getattr(response, "candidates", []) or []
            for cand in candidates:
                content = getattr(cand, "content", None)
                parts = getattr(content, "parts", []) if content is not None else getattr(cand, "parts", []) or []
                texts = [getattr(p, "text", "") for p in parts if getattr(p, "text", "")]
                if texts:
                    return "".join(texts)
        except Exception:
            return ""
        return ""

    @staticmethod
    def _strip_react_trace(text: str) -> str:
        """Remove any leaked ReAct chain-of-thought from the answer before returning it to the caller.
        Priority:
          1. If 'Final Answer:' appears inside the text (model repeated it), extract only what follows.
          2. Strip leading/trailing lines that start with ReAct keywords.
        """
        import re as _re
        # Pass 1 — if the model accidentally included another 'Final Answer:' marker, keep only what follows
        fa_match = _re.search(r"final answer:?", text, _re.IGNORECASE)
        if fa_match:
            candidate = text[fa_match.end():].strip()
            if candidate:
                text = candidate

        # Pass 2 — strip lines that start with known ReAct prefixes
        react_prefixes = (
            "thought:", "action:", "observation:", "reflection:",
            "self-critique:", "system interrupt:", "[system hint]",
        )
        clean_lines = []
        skip = False
        for line in text.splitlines():
            low = line.strip().lower()
            if any(low.startswith(p) for p in react_prefixes):
                skip = True
                continue
            if skip and low == "":
                skip = False
                continue
            if not skip:
                clean_lines.append(line)
        result = "\n".join(clean_lines).strip()
        return result if result else text.strip()

    def _generate_answer_with_history(self, question: str, context_docs: List[Document]) -> str:
        """Standard non-streaming generation"""
        return self._generate_answer(question, context_docs)
    def clear_conversation(self):
        self.conversation_manager.clear()

    def get_conversation_stats(self) -> Dict:
        return self.conversation_manager.get_stats()
    
    def get_stats(self) -> Dict:
        """Get index statistics"""
        if not self.documents:
            return {"total_chunks": 0}
        
        stats = {
            "total_chunks": len(self.documents),
            "content_types": {}
        }
        
        for doc in self.documents:
            doc_type = doc.metadata.get("type", "unknown")
            stats["content_types"][doc_type] = stats["content_types"].get(doc_type, 0) + 1
        
        return stats

    ##Chunk checker. Only for debugging purposes.
    def debug_print_chunks_for_source(self, source_name: str, max_chunks: int = 20):
        """Print all (or first N) chunks for a given PDF source."""
        matched = [d for d in self.documents if d.metadata.get("source") == source_name]
        print(f"[DEBUG] Found {len(matched)} chunks for source='{source_name}'")
        for i, doc in enumerate(matched[:max_chunks], 1):
            print(f"\n--- Chunk {i} ---")
            print("metadata:", doc.metadata)
            preview = doc.page_content[:800].replace("\n", " ")
            print("text    :", preview, "...")

# SemanticChunker is defined above (single definition only)

if __name__ == "__main__":
    """
    Manual index builder / inspector.
    Run from terminal:
        python rag_pipeline.py
    """

    PDF_FOLDER = r"C:\Users\A.Kumarasiri\OneDrive - CGIAR\WETLAND CHATBOT DOCUMENT\ALL"
    INDEX_FILE = "pdf_index_enhanced1.pkl"

    # Minimal model params – just enough to construct RAGPipeline.
    # We won't call .query() here, so the tokens/URL don't actually get used.
    model_params = {
        "llm_type": "deepseek",
        "hf_token": f'st.secrets["hf_backup_token_2"]',
        "deepseek_url": "https://router.huggingface.co/v1/chat/completions",
        "deepseek_model": "deepseek-ai/DeepSeek-R1:novita",
    }

    print("[MAIN] Initializing RAGPipeline1...")
    pipeline = RAGPipeline1(
        pdf_folder=PDF_FOLDER,
        index_file=INDEX_FILE,
        model_params=model_params,
    )

        
    # 1) Try to load existing index
    if pipeline.load_index():
        print("[MAIN] Existing index loaded.")
    else:
        print("[MAIN] No index found or failed to load. Building a new one...")
        try:
            total_chunks = pipeline.build_index()
            print(f"[MAIN] Index built successfully. Total chunks: {total_chunks}")
        except Exception as e:
            print("[MAIN] Index build failed:", e)
            raise
