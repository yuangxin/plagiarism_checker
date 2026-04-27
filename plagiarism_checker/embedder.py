"""
文本向量化与索引构建：支持 CPU/GPU、并行批量处理与多语言模型。
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(".cache/embeddings")


def _cache_key(texts: list[str], model_name: str) -> str:
    """Generate a content-hash key for the embedding cache."""
    content = json.dumps(texts, ensure_ascii=False)
    h = hashlib.sha256(f"{model_name}:{content}".encode()).hexdigest()[:16]
    return h


def build_embeddings(
    texts: List[str],
    model_name: str = "all-MiniLM-L6-v2",
    device: str | None = None,
    batch_size: int = 64,
    use_cache: bool = True,
) -> np.ndarray:
    """
    将文本批量编码为向量。

    Args:
        texts: 文本列表。
        model_name: 使用的 SentenceTransformer 模型名。
        device: 设备选择（None 自动，'cuda' 使用GPU，'cpu' 使用CPU）。
        batch_size: 批量大小。
        use_cache: 是否使用基于内容哈希的缓存。

    Returns:
        归一化的 float32 向量矩阵。
    """
    # Check cache
    if use_cache:
        key = _cache_key(texts, model_name)
        cache_file = _CACHE_DIR / f"{key}.npy"
        if cache_file.exists():
            try:
                embeddings = np.load(cache_file)
                logger.info("Cache hit for %s (%d texts)", model_name, len(texts))
                return embeddings
            except Exception as e:
                logger.warning("Cache read failed: %s", e)

    # 自动判断设备并构建模型
    if device is None:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(model_name, device=device)
    
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    embeddings = embeddings.astype("float32")

    # Save to cache
    if use_cache:
        try:
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            np.save(cache_file, embeddings)
            logger.info("Cached embeddings for %s (%d texts)", model_name, len(texts))
        except Exception as e:
            logger.warning("Cache write failed: %s", e)

    return embeddings


def build_embeddings_parallel(
    texts: List[str],
    model_name: str = "all-MiniLM-L6-v2",
    device: str | None = None,
    batch_size: int = 64,
    num_workers: int = 2,
) -> np.ndarray:
    """
    多线程并行编码（CPU场景友好）。

    Args:
        texts: 文本列表。
        model_name: 模型名。
        device: 设备选择。
        batch_size: 批量大小。
        num_workers: 并行线程数。

    Returns:
        归一化的 float32 向量矩阵。

    Note:
        GPU 下使用并行编码通常不如单线程高效。
    """
    if device and 'cuda' in device:
        # GPU下直接用单线程就行
        return build_embeddings(texts, model_name, device, batch_size)
    
    model = SentenceTransformer(model_name, device='cpu')
    
    # 将文本分块以供并行处理
    chunk_size = len(texts) // num_workers
    if chunk_size == 0:
        return build_embeddings(texts, model_name, device, batch_size)
    
    chunks = []
    for i in range(num_workers):
        start = i * chunk_size
        end = start + chunk_size if i < num_workers - 1 else len(texts)
        chunks.append(texts[start:end])
    
    def encode_chunk(chunk):
        return model.encode(
            chunk,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(encode_chunk, chunks))
    
    embeddings = np.vstack(results)
    return embeddings.astype("float32")


def build_index(embeddings: np.ndarray) -> faiss.Index:
    """
    使用 FAISS 构建索引（内积），在归一化向量上等价于余弦相似。

    Args:
        embeddings: 归一化向量矩阵。

    Returns:
        FAISS 索引对象。
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # IP = 内积，对归一化向量就是余弦
    index.add(embeddings)
    return index


def build_multilingual_embeddings(
    texts: List[str],
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    device: str | None = None,
    batch_size: int = 64,
) -> np.ndarray:
    """
    构建跨语言向量（支持中英文混合）。

    Args:
        texts: 文本列表。
        model_name: 多语言模型名。
        device: 设备选择。
        batch_size: 批量大小。

    Returns:
        归一化的 float32 向量矩阵。
    """
    return build_embeddings(texts, model_name, device, batch_size)
