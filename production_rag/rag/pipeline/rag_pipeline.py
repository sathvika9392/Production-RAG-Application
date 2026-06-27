import logging
from typing import List, Dict, Any, Optional
import yaml
import os

from llama_index.core import VectorStoreIndex
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import NodeWithScore

from production_rag.rag.retrievers.hybrid_retriever import HybridRetriever
from production_rag.rag.reranker.cross_encoder import CrossEncoderReranker
from production_rag.rag.utils.logger import logger

class RagPipeline:
    """Production RAG Pipeline with Hybrid Search, Reranking, and Citation Enforcement."""

    def __init__(self, index: VectorStoreIndex, llm: Any, config_path: str = "production_rag/rag/config/config.yaml"):
        self.index = index
        self.llm = llm
        self.config = self._load_config(config_path)
        
        # Initialize components from config
        rag_cfg = self.config.get("rag", {})
        self.hybrid_retriever = HybridRetriever(
            index=index,
            vector_weight=rag_cfg.get("vector_weight", 0.5),
            bm25_weight=rag_cfg.get("bm25_weight", 0.5),
            top_k=rag_cfg.get("top_k", 5)
        )
        
        self.reranker = None
        if rag_cfg.get("reranker_enabled", True):
            self.reranker = CrossEncoderReranker(
                model_name=rag_cfg.get("reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
                top_n=rag_cfg.get("reranker_top_n", 3)
            )

        # Production-grade prompt with strict citation enforcement
        self.system_prompt = PromptTemplate(
            "You are a production-grade AI assistant. Your task is to answer user queries NOT based on your internal knowledge, "
            "but ONLY using the provided text context.\n\n"
            "STRICT RULES:\n"
            "1. If the answer is not contained within the context, concisely state: 'I am sorry, but the provided documentation does not contain enough information to answer this question.'\n"
            "2. Do NOT hallucinate or use external information.\n"
            "3. Every answer MUST be followed by a 'Sources:' section.\n"
            "4. Format the sources as bullet points with the format: * <document_name> (Page/Chunk: <id>)\n\n"
            "CONTEXT:\n"
            "{context_str}\n\n"
            "USER QUERY: {query_str}\n\n"
            "FINAL ANSWER FORMAT:\n"
            "Answer: <your grounded answer>\n\n"
            "Sources:\n"
            "* <source 1>\n"
            "* <source 2>"
        )

    def _load_config(self, path: str) -> Dict:
        if os.path.exists(path):
            with open(path, "r") as f:
                return yaml.safe_load(f)
        logger.warning(f"Config file {path} not found, using defaults.")
        return {}

    def run(self, query: str) -> Dict[str, Any]:
        """Execute the full RAG pipeline synchronously."""
        try:
            nodes = self._get_nodes(query)
            if not nodes:
                return {
                    "answer": "Answer: I am sorry, but no relevant documents were found to answer your query.\n\nSources: None",
                    "sources": []
                }

            prompt = self._prepare_prompt(query, nodes)
            response = self.llm.complete(prompt)
            
            return {
                "answer": response.text,
                "sources": self._format_sources(nodes)
            }
        except Exception as e:
            logger.error(f"Pipeline failure: {e}")
            return {"answer": f"Error: {str(e)}", "sources": []}

    def stream_run(self, query: str) -> Dict[str, Any]:
        """Execute the full RAG pipeline with streaming response."""
        try:
            nodes = self._get_nodes(query)
            if not nodes:
                # Return a generator for the empty case to match expectations
                def empty_gen():
                    yield "Answer: I am sorry, but no relevant documents were found."
                return {"answer": empty_gen(), "sources": []}

            prompt = self._prepare_prompt(query, nodes)
            response_gen = self.llm.stream_complete(prompt)
            
            return {
                "answer": (resp.delta for resp in response_gen),
                "sources": self._format_sources(nodes)
            }
        except Exception as e:
            logger.error(f"Streaming pipeline failure: {e}")
            def error_gen():
                yield f"Error: {str(e)}"
            return {"answer": error_gen(), "sources": []}

    def _get_nodes(self, query: str) -> List[NodeWithScore]:
        """Internal helper to get retrieved and reranked nodes."""
        nodes = self.hybrid_retriever.retrieve(query)
        if self.reranker and nodes:
            nodes = self.reranker.rerank(query, nodes)
        return nodes

    def _prepare_prompt(self, query: str, nodes: List[NodeWithScore]) -> str:
        """Internal helper to prepare the prompt with context."""
        context_str = ""
        for i, node in enumerate(nodes):
            content = node.node.get_content()
            metadata = node.node.metadata
            file_name = metadata.get("file_name", "Unknown Document")
            page = metadata.get("page_label", "N/A")
            context_str += f"[Doc {i+1}] Source: {file_name} (Page: {page})\nContent: {content}\n\n"
        return self.system_prompt.format(context_str=context_str, query_str=query)

    def _format_sources(self, nodes: List[NodeWithScore]) -> List[Dict]:
        """Internal helper to format source metadata."""
        return [
            {
                "file_name": node.node.metadata.get("file_name"),
                "page": node.node.metadata.get("page_label"),
                "score": node.score
            } for node in nodes
        ]

