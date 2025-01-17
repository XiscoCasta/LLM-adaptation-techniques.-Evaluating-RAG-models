from transformers import AutoModel, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
from langchain.embeddings.base import Embeddings
import torch
from langchain.vectorstores import FAISS
import re
from typing import Optional, List
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm.notebook import tqdm

class CustomHuggingFaceEmbeddings(Embeddings):
    def __init__(self, model_name: str, normalize = False ,device: str = "cpu"):
        """
        Initialize a custom wrapper for Hugginface embeddings.
        Args:
            model_name: Hugging Face model name for the generator (e.g., "google/flan-t5-small").
            normalize: boolean to return normalized.
            device: Device to load the model on ("cpu" or "cuda").
        """
        #Initialize the model and the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.device = device
        self.normalize = normalize
        self.model.to(self.device)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embeds a given list of documents.
        Args:
            texts: List of documents to embed.
        Returns:
            The generated embeddings.
        """
        self.model.eval()
        embeddings = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state[:, 0, :].squeeze(0)
                if self.normalize:
                    embedding = embedding / torch.norm(embedding, p=2)
            embeddings.append(embedding.cpu().numpy().tolist())
        return embeddings

    def embed_query(self, query: str) -> list[float]:
        """
        Embeds a given query.
        Args:
            query: Query to embed.
        Returns:
            The generated embedding.
        """
        self.model.eval()
        inputs = self.tokenizer(query, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].squeeze(0)
            if self.normalize:
                embedding = embedding / torch.norm(embedding, p=2)
        return embedding.cpu().numpy().tolist()
    

def evaluate_vector_databases(vector_databases, subset, k_values):
    """
    Evaluates multiple vector databases by checking if the actual context appears
    in the retrieved documents for each question in the subset.

    Args:
        vector_databases (dict): A dictionary mapping database names to vector database instances.
        subset (list): A list of question dictionaries, each containing 'id', 'question', and 'context'.
        k_values (list): A list of k values to test for similarity search.

    Returns:
        list: A list of dictionaries containing evaluation results.
    """
    results = []
  
    # Iterate through the subset of questions
    for row in tqdm(subset, desc="Evaluating questions"):
        question_id = row['id']
        question_text = row['question']
        actual_context = row['context']

        # Iterate through each vector database
        for db_name, vector_db in vector_databases.items():

            # Evaluate for each k value
            for k in k_values:
                retrieved_docs = vector_db.similarity_search(question_text, k=k)
                found = any(doc.page_content == actual_context for doc in retrieved_docs)
                results.append({
                    "question_id": question_id,
                    "question": question_text,
                    "actual_context": actual_context,
                    "db_name": db_name,
                    "k": k,
                    "retrieved_docs": [doc.page_content for doc in retrieved_docs],
                    "actual_context_found": found,
                })

    return results


def evaluate_answers(baseline_answers,tokenizer,return_errors = False):
    """
    Evaluate the exact match rate of generated answers against ground truths.

    Args:
        baseline_answers (list): A list of dictionaries containing 'answer' and 'ground_truths'.
        return_errors (bool): Variable to return the error indices.

    Returns:
        tuple: A tuple containing the number of exact matches and a list of indices where errors occurred.
    """
    exact_matches = 0
    errors = []

    # Iterate through the baseline answers
    for i in tqdm(range(len(baseline_answers)), desc="Generating answers", unit="question"):
        generated_answer = baseline_answers[i]['answer']
        ground_truths = baseline_answers[i]['ground_truths']

        # Use tokenized comparison instead of direct string comparison
        if tokenize_compare(generated_answer, ground_truths,tokenizer):
            exact_matches += 1
        else:
            errors.append(i)
    if return_errors:
        return exact_matches, errors
    else:
        return exact_matches


class GenerativePipeline:
    def __init__(self, model_name: str = "google/flan-t5-small", device: str = "cpu"):
        """
        Initialize the Flan-T5 model and tokenizer for question answering.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = device
        self.model.to(self.device)

    def generate_answer(self, question: str, context: str, max_length: int = 128) -> str:
        """
        Generate an answer given a question and context using Flan-T5.
        """
        input_text = f"question: {question} context: {context}"
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=max_length,
            num_beams=5,
            early_stopping=True
        )
        
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer

def tokenize_compare(generated: str, ground_truths: list[str], tokenizer ,threshold: float = 0.8) -> bool:
    """
    Compare generated and ground truths based on token overlap using a proper tokenizer.
    Args:
        generated: The generated answer.
        ground_truths: A list of ground truth answers.
        tokenizer: Tokenizer used to tokenized the answers in order to compare them.
        threshold: The minimum token overlap ratio to consider as a match.
    Returns:
        bool: True if there's sufficient token overlap, False otherwise.
    """
    generated_tokens = set(tokenizer.tokenize(re.sub(r'[^\w\s]', '', generated.lower())))
    
    for gt in ground_truths:
        gt_tokens = set(tokenizer.tokenize(re.sub(r'[^\w\s]', '', gt.lower())))
        # Calculate token overlap ratio
        if len(gt_tokens)>0:
            overlap = len(generated_tokens & gt_tokens) / len(gt_tokens)
        else:
            overlap =0
        if overlap >= threshold:
            return True
    return False

class RAGPipeline:
    def __init__(self, model_name: str, retriever_k1: FAISS, retriever_kgt1: FAISS, device: str = "cpu"):
        """
        Initialize the RAG pipeline.
        Args:
            model_name: Hugging Face model name for the generator (e.g., "google/flan-t5-small").
            retriever_k1: Retriever to use when k=1.
            retriever_kgt1: Retriever to use when k>1.
            device: Device to load the model on ("cpu" or "cuda").
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
        self.retriever_k1 = retriever_k1
        self.retriever_kgt1 = retriever_kgt1
        self.device = device
        self.model.to(self.device)

    def retrieve_context(self, question: str, k: int) -> str:
        """
        Retrieve context from the appropriate retriever based on k.
        Args:
            question: The query/question for the retriever.
            k: Number of documents to retrieve.
        Returns:
            A concatenated string of retrieved documents.
        """
        retriever = self.retriever_k1 if k == 1 else self.retriever_kgt1
        results = retriever.similarity_search(question, k=k)
        return "\n\n".join([doc.page_content for doc in results])

    def generate_answer(self, question: str, k: int, return_context = False) -> str:
        """
        Generate an answer using Flan-T5 with retrieved context.
        Args:
            question: The question to answer.
            k: Number of documents to retrieve.
        Returns:
            The generated answer.
        """
        context = self.retrieve_context(question, k)
        input_text = f"question: {question} context: {context}"
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True,truncation = True).to(self.device)
        
        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=128,
            num_beams=5,
            early_stopping=True
        )
        if return_context: 
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True), context
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

def split_documents(
    chunk_size: int,
    knowledge_base: List[LangchainDocument],
    tokenizer_name: Optional[str] = "avsolatorio/NoInstruct-small-Embedding-v0",
    MARKDOWN_SEPARATORS = [
    "\n\n",  # Paragraph breaks
    "\n",    # Line breaks
    ". ",    # Sentence endings
    "? ",    # Question endings
    "! ",    # Exclamation endings
    " ",     # Fallback to spaces
    ""       # Catch-all
    ]
) -> List[LangchainDocument]:
    """
    Split documents into chunks of maximum size `chunk_size` tokens and return a list of documents.
    Args:
        chunk_size: Maximum size of the chunks once tokenized.
        knowledge_base: List of documents to split.
        tokenizer_name: Name of the tokenizer to use.
        MARKDOWN_SEPARATORS: Separators to split on them in order until chunks are small enough.
    Returns:
        A list of chunks .
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 10),
        add_start_index=True,
        strip_whitespace=True,
        separators=MARKDOWN_SEPARATORS,
    )

    docs_processed = []
    for doc in tqdm(knowledge_base, desc="Splitting documents", unit="doc"):
        chunks = text_splitter.split_documents([doc])
        docs_processed += chunks

    # Remove duplicates
    unique_texts = {}
    docs_processed_unique = []
    for doc in docs_processed:
        if doc.page_content not in unique_texts:
            unique_texts[doc.page_content] = True
            docs_processed_unique.append(doc)

    return docs_processed_unique


class RAGPipeline_with_rerank:
    def __init__(self, model_name: str, retriever: FAISS, cross_encoder_name: str, device: str = "cpu"):
        """
        Initialize the RAG pipeline.
        Args:
            model_name: Hugging Face model name for the generator (e.g., "google/flan-t5-small").
            retriever: Retriever to use to get the first documents from the database.
            cross_encoder_name: Cross-encoder to rerank the retrieved documents.
            device: Device to load the model on ("cpu" or "cuda").
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
        self.retriever = retriever
        self.tokenizer_cross = AutoTokenizer.from_pretrained(cross_encoder_name, trust_remote_code=True)
        self.model_cross = AutoModelForSequenceClassification.from_pretrained(cross_encoder_name,trust_remote_code = True)
        self.device = device
        self.model.to(self.device)
        self.model_cross.to(self.device)

    def retrieve_context(self, question: str, k: int) -> str:
        """
        Retrieve context from the appropriate retriever based on k.
        Args:
            question: The query/question for the retriever.
            k: Number of documents to retrieve.
        Returns:
            Retrieved documents.
        """
        results = self.retriever.similarity_search(question, k=k)
        return results
    def rerank_context(self, retrieved_context: List[LangchainDocument], k_reranked: int, question: str, return_scores = False) -> str:
        """
        Rerank retrieved documents and return the top k_reranked.
        Args:
            retrieved_context: List of documents retrieved.
            k_reranked: Number of reranked documents to return.
            question: The question for which contexts were retrieved.
        Returns:
            A concatenated string of the top k_reranked documents.
        """
        scores = []

        # Tokenize question and contexts for the cross-encoder
        for doc in retrieved_context:
            tokenized_input = self.tokenizer_cross(
                question,
                doc.page_content,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)

            # Compute relevance score using the cross-encoder model
            self.model_cross.eval()
            with torch.no_grad():
                logits = self.model_cross(**tokenized_input).logits
                score = logits.squeeze().item()
            scores.append((doc, score))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)

        # Retrieve the top k_reranked documents
        top_k_docs = [doc.page_content for doc, _ in scores[:k_reranked]]

        if return_scores:
            return top_k_docs, [score for _, score in scores]
        return top_k_docs

    def generate_answer(self, question: str, k_retriever: int,k_reranked: int, return_context = False) -> str:
        """
        Generate an answer using Flan-T5 with retrieved context.
        Args:
            question: The question to answer.
            k_retriever: Number of documents to retrieve.
            k_reranked: Number of documents to rerank and use.
            return_context: Whether to return the reranked context with the answer.
        Returns:
            The generated answer (and optionally the context).
        """
        retrieved_context = self.retrieve_context(question, k_retriever)
        top_rerank_docs = self.rerank_context(retrieved_context, k_reranked, question)
        concatenated_context = "\n".join(top_rerank_docs)
        input_text = f"question: {question} context: {reranked_context}"
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True,truncation = True).to(self.device)
        
        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=128,
            num_beams=5,
            early_stopping=True
        )
        if return_context: 
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True), reranked_context
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


