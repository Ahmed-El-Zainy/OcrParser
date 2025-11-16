# DotsOCR-Based Question-Answer Mapping Implementation

## Overview

This guide implements the three-stage approach (Extraction, Structuring, Mapping) for matching questions and answers from separate PDFs using the DotsOCR model from your `embedding_model.py` code.

---

## Architecture Overview

```
┌─────────────────┐         ┌─────────────────┐
│  Questions PDF  │         │   Answers PDF   │
└────────┬────────┘         └────────┬────────┘
         │                           │
         ▼                           ▼
┌─────────────────────────────────────────────┐
│     Stage 1: EXTRACTION (DotsOCR)          │
│  Convert PDFs to structured text via vLLM  │
└────────┬────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│     Stage 2: STRUCTURING (Regex/LLM)       │
│  Parse text into Question/Answer entities  │
└────────┬────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│     Stage 3: MAPPING                        │
│  Match questions to answers using:          │
│  - Case A: ID-based matching (simple)       │
│  - Case B: Embedding similarity (complex)   │
└─────────────────────────────────────────────┘
```

---

## Prerequisites

### 1. DotsOCR Setup (From Your Code)

```bash
# Clone the DotsOCR repository
git clone https://github.com/dots-ocr/dots.ocr.git
cd dots.ocr

# Set up model path
export hf_model_path="./weights/DotsOCR_model"
mkdir -p $hf_model_path

# Download model weights
python tools/download_model.py

# Install dependencies
pip install transformers>=4.52.4
pip install flash-attn>=2.7.4.post1
pip install vllm
pip install sentence-transformers  # For embedding-based mapping
pip install openai  # For vLLM client
```

### 2. Start vLLM Server

```bash
# Launch vLLM server (from your code, adapted)
CUDA_VISIBLE_DEVICES=0 vllm serve ./weights/DotsOCR_model \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.95 \
    --chat-template-content-format string \
    --served-model-name dotsocr \
    --trust-remote-code \
    --port 8000
```

Wait ~30 seconds for the server to initialize.

---

## Stage 1: Extraction Using DotsOCR

### Implementation

```python
import os
import json
import base64
from pathlib import Path
from openai import OpenAI

class DotsOCRExtractor:
    """Extract text from PDFs using DotsOCR via vLLM server"""
    
    def __init__(self, base_url="http://localhost:8000/v1", model_name="dotsocr"):
        """
        Initialize DotsOCR client
        
        Args:
            base_url: vLLM server URL
            model_name: Model name as registered in vLLM
        """
        self.client = OpenAI(
            api_key="EMPTY",  # vLLM doesn't require API key
            base_url=base_url
        )
        self.model_name = model_name
    
    def encode_image(self, image_path):
        """Encode image to base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def extract_from_pdf(self, pdf_path, prompt=None):
        """
        Extract text from PDF using DotsOCR
        
        Args:
            pdf_path: Path to PDF file
            prompt: Custom extraction prompt (optional)
        
        Returns:
            str: Extracted text
        """
        if prompt is None:
            prompt = (
                "Extract all text from this document page. "
                "Maintain the original structure and formatting. "
                "Include all question numbers, text, and content."
            )
        
        # Convert PDF to images (using pdf2image)
        from pdf2image import convert_from_path
        
        images = convert_from_path(pdf_path)
        extracted_pages = []
        
        for page_num, image in enumerate(images, 1):
            print(f"Processing page {page_num}/{len(images)}...")
            
            # Save temporary image
            temp_image_path = f"/tmp/temp_page_{page_num}.png"
            image.save(temp_image_path, 'PNG')
            
            # Encode image
            base64_image = self.encode_image(temp_image_path)
            
            # Call DotsOCR via vLLM
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{base64_image}"
                                    }
                                },
                                {
                                    "type": "text",
                                    "text": prompt
                                }
                            ]
                        }
                    ],
                    max_tokens=4096,
                    temperature=0.0  # Deterministic for OCR
                )
                
                page_text = response.choices[0].message.content
                extracted_pages.append({
                    'page': page_num,
                    'text': page_text
                })
                
            except Exception as e:
                print(f"Error processing page {page_num}: {e}")
                extracted_pages.append({
                    'page': page_num,
                    'text': '',
                    'error': str(e)
                })
            finally:
                # Clean up temp file
                if os.path.exists(temp_image_path):
                    os.remove(temp_image_path)
        
        # Combine all pages
        full_text = "\n\n--- PAGE BREAK ---\n\n".join(
            [page['text'] for page in extracted_pages if page['text']]
        )
        
        return full_text, extracted_pages


# Usage Example
extractor = DotsOCRExtractor()

# Extract questions
questions_text, questions_pages = extractor.extract_from_pdf(
    "questions.pdf",
    prompt="Extract all questions with their numbers. Preserve question numbering."
)

# Extract answers
answers_text, answers_pages = extractor.extract_from_pdf(
    "answers.pdf",
    prompt="Extract all answers with their question numbers. Preserve answer numbering."
)

# Save extracted text
with open("extracted_questions.txt", "w", encoding="utf-8") as f:
    f.write(questions_text)

with open("extracted_answers.txt", "w", encoding="utf-8") as f:
    f.write(answers_text)

print(f"✅ Extracted {len(questions_pages)} question pages")
print(f"✅ Extracted {len(answers_pages)} answer pages")
```

---

## Stage 2: Structuring

### Method 2A: Regex-Based Structuring (Fast, Reliable IDs)

```python
import re
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class Question:
    """Structured question entity"""
    id: str
    text: str
    raw_text: str
    page: int = None

@dataclass
class Answer:
    """Structured answer entity"""
    id: str
    text: str
    raw_text: str
    page: int = None

class RegexStructurer:
    """Parse extracted text into structured Question/Answer objects"""
    
    def __init__(self):
        # Common patterns for question/answer numbering
        self.patterns = {
            'numeric': r'^(\d+)\.?\s+(.+?)(?=^\d+\.?\s+|\Z)',  # 1. Question text
            'numeric_paren': r'^\((\d+)\)\s+(.+?)(?=^\(\d+\)|\Z)',  # (1) Question text
            'letter': r'^([A-Z])\.?\s+(.+?)(?=^[A-Z]\.?\s+|\Z)',  # A. Question text
            'question_prefix': r'^(?:Question|Q)\s*(\d+)\.?\s+(.+?)(?=^(?:Question|Q)\s*\d+|\Z)',  # Question 1. Text
        }
    
    def parse_questions(self, text: str, pattern_name='numeric') -> List[Question]:
        """
        Parse questions from extracted text
        
        Args:
            text: Extracted text from questions PDF
            pattern_name: Which regex pattern to use
        
        Returns:
            List of Question objects
        """
        pattern = self.patterns.get(pattern_name, self.patterns['numeric'])
        matches = re.findall(pattern, text, re.MULTILINE | re.DOTALL)
        
        questions = []
        for match in matches:
            q_id, q_text = match
            questions.append(Question(
                id=q_id.strip(),
                text=q_text.strip(),
                raw_text=f"{q_id}. {q_text}".strip()
            ))
        
        return questions
    
    def parse_answers(self, text: str, pattern_name='numeric') -> Dict[str, Answer]:
        """
        Parse answers from extracted text into a dictionary
        
        Args:
            text: Extracted text from answers PDF
            pattern_name: Which regex pattern to use
        
        Returns:
            Dictionary mapping question_id -> Answer object
        """
        pattern = self.patterns.get(pattern_name, self.patterns['numeric'])
        matches = re.findall(pattern, text, re.MULTILINE | re.DOTALL)
        
        answers = {}
        for match in matches:
            a_id, a_text = match
            answers[a_id.strip()] = Answer(
                id=a_id.strip(),
                text=a_text.strip(),
                raw_text=f"{a_id}. {a_text}".strip()
            )
        
        return answers
    
    def auto_detect_pattern(self, text: str) -> str:
        """
        Auto-detect the best pattern for the given text
        
        Returns:
            Pattern name with highest match count
        """
        max_matches = 0
        best_pattern = 'numeric'
        
        for pattern_name, pattern in self.patterns.items():
            matches = re.findall(pattern, text[:5000], re.MULTILINE | re.DOTALL)
            if len(matches) > max_matches:
                max_matches = len(matches)
                best_pattern = pattern_name
        
        print(f"Auto-detected pattern: {best_pattern} ({max_matches} matches)")
        return best_pattern


# Usage
structurer = RegexStructurer()

# Auto-detect and parse
q_pattern = structurer.auto_detect_pattern(questions_text)
a_pattern = structurer.auto_detect_pattern(answers_text)

questions = structurer.parse_questions(questions_text, q_pattern)
answers_dict = structurer.parse_answers(answers_text, a_pattern)

print(f"✅ Parsed {len(questions)} questions")
print(f"✅ Parsed {len(answers_dict)} answers")

# Preview
if questions:
    print(f"\nFirst Question: {questions[0].id}")
    print(f"Text: {questions[0].text[:100]}...")
```

### Method 2B: LLM-Based Structuring (Robust, Handles Complex Layouts)

```python
class LLMStructurer:
    """Use LLM to structure extracted text when regex fails"""
    
    def __init__(self, extractor: DotsOCRExtractor):
        self.client = extractor.client
        self.model = extractor.model_name
    
    def parse_questions_llm(self, text: str) -> List[Question]:
        """
        Use LLM to extract structured questions
        
        Args:
            text: Raw extracted text
        
        Returns:
            List of Question objects
        """
        prompt = f"""
Extract all questions from the following text and format them as JSON.

Text:
{text[:8000]}  # Limit context size

Return ONLY a JSON array with this structure:
[
  {{"id": "1", "text": "Question text here"}},
  {{"id": "2", "text": "Another question"}}
]

JSON:
"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0.0
        )
        
        # Parse JSON response
        try:
            json_str = response.choices[0].message.content
            # Extract JSON from markdown code blocks if present
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]
            
            data = json.loads(json_str.strip())
            
            questions = []
            for item in data:
                questions.append(Question(
                    id=str(item['id']),
                    text=item['text'],
                    raw_text=f"{item['id']}. {item['text']}"
                ))
            
            return questions
            
        except json.JSONDecodeError as e:
            print(f"Failed to parse LLM response: {e}")
            return []
    
    def parse_answers_llm(self, text: str) -> Dict[str, Answer]:
        """Use LLM to extract structured answers"""
        prompt = f"""
Extract all answers from the following text and format them as JSON.
Each answer should be linked to its question number.

Text:
{text[:8000]}

Return ONLY a JSON array with this structure:
[
  {{"question_id": "1", "answer_text": "Answer to question 1"}},
  {{"question_id": "2", "answer_text": "Answer to question 2"}}
]

JSON:
"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0.0
        )
        
        try:
            json_str = response.choices[0].message.content
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]
            
            data = json.loads(json_str.strip())
            
            answers = {}
            for item in data:
                q_id = str(item['question_id'])
                answers[q_id] = Answer(
                    id=q_id,
                    text=item['answer_text'],
                    raw_text=f"{q_id}. {item['answer_text']}"
                )
            
            return answers
            
        except json.JSONDecodeError as e:
            print(f"Failed to parse LLM response: {e}")
            return {}


# Usage
llm_structurer = LLMStructurer(extractor)

questions_llm = llm_structurer.parse_questions_llm(questions_text)
answers_llm = llm_structurer.parse_answers_llm(answers_text)
```

---

## Stage 3: Mapping

### Case A: ID-Based Mapping (Simple & Fast)

```python
from typing import List, Tuple

class IDBasedMapper:
    """Map questions to answers using their IDs"""
    
    @staticmethod
    def map_qa_pairs(
        questions: List[Question], 
        answers: Dict[str, Answer]
    ) -> List[Tuple[Question, Answer]]:
        """
        Map questions to answers by matching IDs
        
        Args:
            questions: List of Question objects
            answers: Dictionary of Answer objects keyed by ID
        
        Returns:
            List of (Question, Answer) tuples
        """
        qa_pairs = []
        unmatched_questions = []
        
        for question in questions:
            if question.id in answers:
                qa_pairs.append((question, answers[question.id]))
            else:
                unmatched_questions.append(question)
                print(f"⚠️  No answer found for Question {question.id}")
        
        print(f"\n✅ Matched {len(qa_pairs)} Q&A pairs")
        print(f"⚠️  {len(unmatched_questions)} unmatched questions")
        
        return qa_pairs
    
    @staticmethod
    def save_qa_pairs(qa_pairs: List[Tuple[Question, Answer]], output_path: str):
        """Save Q&A pairs to JSON file"""
        data = []
        for question, answer in qa_pairs:
            data.append({
                'question_id': question.id,
                'question': question.text,
                'answer': answer.text
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved Q&A pairs to {output_path}")


# Usage
mapper = IDBasedMapper()
qa_pairs = mapper.map_qa_pairs(questions, answers_dict)
mapper.save_qa_pairs(qa_pairs, "qa_pairs_mapped.json")

# Display preview
for i, (q, a) in enumerate(qa_pairs[:3], 1):
    print(f"\n{'='*60}")
    print(f"Q{q.id}: {q.text[:100]}...")
    print(f"A{a.id}: {a.text[:100]}...")
```

### Case B: Embedding-Based Mapping (Semantic Similarity)

```python
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np

class EmbeddingBasedMapper:
    """Map questions to answers using semantic similarity"""
    
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize embedding model
        
        Args:
            model_name: Sentence transformer model to use
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
    
    def map_qa_by_similarity(
        self,
        questions: List[Question],
        answers: List[Answer],
        threshold: float = 0.5
    ) -> List[Tuple[Question, Answer, float]]:
        """
        Map questions to answers using cosine similarity
        
        Args:
            questions: List of Question objects
            answers: List of Answer objects (from dict.values())
            threshold: Minimum similarity score to consider a match
        
        Returns:
            List of (Question, Answer, similarity_score) tuples
        """
        print("Generating question embeddings...")
        question_texts = [q.text for q in questions]
        question_embeddings = self.model.encode(
            question_texts, 
            convert_to_tensor=True,
            show_progress_bar=True
        )
        
        print("Generating answer embeddings...")
        answer_texts = [a.text for a in answers]
        answer_embeddings = self.model.encode(
            answer_texts,
            convert_to_tensor=True,
            show_progress_bar=True
        )
        
        print("Computing similarity matrix...")
        # Compute cosine similarity between all questions and answers
        similarity_matrix = util.cos_sim(question_embeddings, answer_embeddings)
        
        qa_pairs = []
        unmatched_questions = []
        
        for i, question in enumerate(questions):
            # Find best matching answer
            similarities = similarity_matrix[i]
            best_match_idx = torch.argmax(similarities).item()
            best_score = similarities[best_match_idx].item()
            
            if best_score >= threshold:
                qa_pairs.append((
                    question,
                    answers[best_match_idx],
                    best_score
                ))
            else:
                unmatched_questions.append(question)
                print(f"⚠️  Low confidence match for Q{question.id}: {best_score:.3f}")
        
        print(f"\n✅ Matched {len(qa_pairs)} Q&A pairs (threshold={threshold})")
        print(f"⚠️  {len(unmatched_questions)} unmatched questions")
        
        return qa_pairs
    
    @staticmethod
    def save_qa_pairs_with_scores(
        qa_pairs: List[Tuple[Question, Answer, float]], 
        output_path: str
    ):
        """Save Q&A pairs with similarity scores"""
        data = []
        for question, answer, score in qa_pairs:
            data.append({
                'question_id': question.id,
                'question': question.text,
                'answer_id': answer.id,
                'answer': answer.text,
                'similarity_score': float(score)
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved Q&A pairs to {output_path}")
    
    def visualize_similarity_matrix(
        self,
        questions: List[Question],
        answers: List[Answer],
        output_path: str = "similarity_matrix.png"
    ):
        """Create heatmap visualization of similarity matrix"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        question_texts = [f"Q{q.id}" for q in questions[:20]]  # Limit for readability
        answer_texts = [f"A{a.id}" for a in answers[:20]]
        
        question_embeddings = self.model.encode([q.text for q in questions[:20]])
        answer_embeddings = self.model.encode([a.text for a in answers[:20]])
        
        similarity_matrix = util.cos_sim(question_embeddings, answer_embeddings).numpy()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            similarity_matrix,
            xticklabels=answer_texts,
            yticklabels=question_texts,
            annot=True,
            fmt='.2f',
            cmap='YlOrRd',
            cbar_kws={'label': 'Cosine Similarity'}
        )
        plt.title('Question-Answer Similarity Matrix')
        plt.xlabel('Answers')
        plt.ylabel('Questions')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📊 Saved similarity matrix to {output_path}")


# Usage
embedding_mapper = EmbeddingBasedMapper()

# Convert answers dict to list
answers_list = list(answers_dict.values())

# Map using embeddings
qa_pairs_semantic = embedding_mapper.map_qa_by_similarity(
    questions, 
    answers_list,
    threshold=0.6  # Adjust based on your needs
)

# Save results
embedding_mapper.save_qa_pairs_with_scores(
    qa_pairs_semantic, 
    "qa_pairs_semantic.json"
)

# Optional: Visualize similarity matrix
embedding_mapper.visualize_similarity_matrix(
    questions,
    answers_list,
    "similarity_heatmap.png"
)
```

---

## Complete Pipeline: End-to-End

```python
class QAMappingPipeline:
    """Complete pipeline for Q&A extraction and mapping"""
    
    def __init__(
        self,
        dotsocr_url="http://localhost:8000/v1",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.extractor = DotsOCRExtractor(base_url=dotsocr_url)
        self.regex_structurer = RegexStructurer()
        self.llm_structurer = LLMStructurer(self.extractor)
        self.id_mapper = IDBasedMapper()
        self.embedding_mapper = EmbeddingBasedMapper(model_name=embedding_model)
    
    def process(
        self,
        questions_pdf: str,
        answers_pdf: str,
        output_dir: str = "output",
        use_llm_structuring: bool = False,
        use_semantic_mapping: bool = False,
        similarity_threshold: float = 0.6
    ):
        """
        Complete Q&A mapping pipeline
        
        Args:
            questions_pdf: Path to questions PDF
            answers_pdf: Path to answers PDF
            output_dir: Output directory for results
            use_llm_structuring: Use LLM instead of regex for structuring
            use_semantic_mapping: Use embeddings instead of ID matching
            similarity_threshold: Threshold for semantic mapping
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print("="*60)
        print("STAGE 1: EXTRACTION")
        print("="*60)
        
        # Extract text from PDFs
        questions_text, _ = self.extractor.extract_from_pdf(
            questions_pdf,
            prompt="Extract all questions with their numbers."
        )
        
        answers_text, _ = self.extractor.extract_from_pdf(
            answers_pdf,
            prompt="Extract all answers with their question numbers."
        )
        
        # Save extracted text
        with open(f"{output_dir}/extracted_questions.txt", "w", encoding="utf-8") as f:
            f.write(questions_text)
        with open(f"{output_dir}/extracted_answers.txt", "w", encoding="utf-8") as f:
            f.write(answers_text)
        
        print("\n" + "="*60)
        print("STAGE 2: STRUCTURING")
        print("="*60)
        
        # Structure the data
        if use_llm_structuring:
            print("Using LLM-based structuring...")
            questions = self.llm_structurer.parse_questions_llm(questions_text)
            answers_dict = self.llm_structurer.parse_answers_llm(answers_text)
        else:
            print("Using regex-based structuring...")
            q_pattern = self.regex_structurer.auto_detect_pattern(questions_text)
            a_pattern = self.regex_structurer.auto_detect_pattern(answers_text)
            questions = self.regex_structurer.parse_questions(questions_text, q_pattern)
            answers_dict = self.regex_structurer.parse_answers(answers_text, a_pattern)
        
        print(f"Structured {len(questions)} questions and {len(answers_dict)} answers")
        
        print("\n" + "="*60)
        print("STAGE 3: MAPPING")
        print("="*60)
        
        # Map questions to answers
        if use_semantic_mapping:
            print("Using semantic similarity mapping...")
            answers_list = list(answers_dict.values())
            qa_pairs = self.embedding_mapper.map_qa_by_similarity(
                questions,
                answers_list,
                threshold=similarity_threshold
            )
            
            # Save with scores
            self.embedding_mapper.save_qa_pairs_with_scores(
                qa_pairs,
                f"{output_dir}/qa_pairs_semantic.json"
            )
            
            # Visualize
            self.embedding_mapper.visualize_similarity_matrix(
                questions,
                answers_list,
                f"{output_dir}/similarity_matrix.png"
            )
        else:
            print("Using ID-based mapping...")
            qa_pairs = self.id_mapper.map_qa_pairs(questions, answers_dict)
            self.id_mapper.save_qa_pairs(
                qa_pairs,
                f"{output_dir}/qa_pairs_id_based.json"
            )
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETE")
        print("="*60)
        print(f"Results saved to: {output_dir}/")
        
        return qa_pairs


# Usage Examples

# Example 1: Simple ID-based mapping (fast, reliable IDs)
pipeline = QAMappingPipeline()
qa_pairs = pipeline.process(
    questions_pdf="questions.pdf",
    answers_pdf="answers.pdf",
    output_dir="output_simple",
    use_llm_structuring=False,
    use_semantic_mapping=False
)

# Example 2: Semantic mapping (no reliable IDs)
pipeline = QAMappingPipeline()
qa_pairs = pipeline.process(
    questions_pdf="questions_unstructured.pdf",
    answers_pdf="answers_unstructured.pdf",
    output_dir="output_semantic",
    use_llm_structuring=True,
    use_semantic_mapping=True,
    similarity_threshold=0.7
)

# Example 3: Hybrid approach (regex structuring + semantic fallback)
structurer = RegexStructurer()
questions = structurer.parse_questions(questions_text)
answers_dict = structurer.parse_answers(answers_text)

# Try ID-based first
id_mapper = IDBasedMapper()
qa_pairs_id = id_mapper.map_qa_pairs(questions, answers_dict)

# For unmatched questions, use semantic matching
if len(qa_pairs_id) < len(questions):
    print("\n🔄 Using semantic matching for remaining questions...")
    matched_q_ids = {q.id for q, _ in qa_pairs_id}
    unmatched_questions = [q for q in questions if q.id not in matched_q_ids]
    
    embedding_mapper = EmbeddingBasedMapper()
    qa_pairs_semantic = embedding_mapper.map_qa_by_similarity(
        unmatched_questions,
        list(answers_dict.values()),
        threshold=0.6
    )
    
    # Combine results
    all_qa_pairs = qa_pairs_id + [(q, a, 1.0) for q, a in qa_pairs_id]
    all_qa_pairs.extend(qa_pairs_semantic)
```

---

## Performance Optimization

### 1. Batch Processing for Large Documents

```python
def batch_extract_pdf(pdf_path: str, batch_size: int = 5):
    """Process PDF in batches to avoid memory issues"""
    from pdf2image import convert_from_path
    
    images = convert_from_path(pdf_path)
    total_pages = len(images)
    all_text = []
    
    for i in range(0, total_pages, batch_size):
        batch = images[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1} ({i+1}-{min(i+batch_size, total_pages)})")
        
        # Process batch...
        for page in batch:
            # Extract text
            pass
        
        # Clear memory
        torch.cuda.empty_cache()
    
    return "\n\n".join(all_text)
```

### 2. Caching Embeddings

```python
import pickle

def cache_embeddings(texts: List[str], cache_path: str, model):
    """Cache embeddings to avoid recomputation"""
    if os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    print("Computing embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True)
    
    with open(cache_path, 'wb') as f:
        pickle.dump(embeddings, f)
    
    return embeddings
```

---

## Troubleshooting

### Issue: vLLM Server Not Starting

```bash
# Check GPU availability
nvidia-smi

# Check logs
tail -f vllm_server.log

# Reduce memory usage
vllm serve ./weights/DotsOCR_model \
    --gpu-memory-utilization 0.7 \
    --max-model-len 2048
```

### Issue: Poor Regex Matching

Try different patterns or use LLM-based structuring:
```python
# Enable LLM structuring
qa_pairs = pipeline.process(
    questions_pdf="questions.pdf",
    answers_pdf="answers.pdf",
    use_llm_structuring=True  # <-- Enable this
)
```

### Issue: Low Similarity Scores

- Lower the threshold: `similarity_threshold=0.5`
- Try a different embedding model:
  - `sentence-transformers/all-mpnet-base-v2` (better quality)
  - `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (multilingual)

---

## Summary

This implementation provides:

✅ **Stage 1: Extraction** using DotsOCR via vLLM  
✅ **Stage 2: Structuring** with both Regex and LLM approaches  
✅ **Stage 3: Mapping** with ID-based and embedding-based methods  
✅ **Complete Pipeline** with flexible configuration  
✅ **Visualization** of similarity matrices  
✅ **Error Handling** and fallback strategies  

Choose the approach based on your data:
- **Well-structured PDFs with consistent numbering** → Use regex + ID-based mapping
- **Inconsistent or missing IDs** → Use LLM structuring + semantic mapping
- **Mixed scenarios** → Use hybrid approach with ID-based first, semantic fallback