# Embedding Model Documentation
## Overview
This document provides comprehensive documentation for the `embedding_model.py` script, which implements multiple OCR (Optical Character Recognition) models and embedding techniques for document processing. The script is designed to run in Google Colab and includes functionality for text extraction, question-answer mapping, and semantic embedding generation.

---

## Table of Contents
1. [System Setup](#system-setup)
2. [OCR Models](#ocr-models)
   - [Qari Model](#qari-model)
   - [Nanonets OCR](#nanonets-ocr)
3. [Mapping Functionality (Detailed)](#mapping-functionality-detailed)
   - [Question-Answer-Choice Mapping](#question-answer-choice-mapping)
   - [Mapping Logic Overview](#mapping-logic-overview)
   - [Entity Extraction and Mapping](#entity-extraction-and-mapping)
   - [Attribute-Based Mapping](#attribute-based-mapping)
4. [Embedding Models](#embedding-models)
5. [Code Structure](#code-structure)
6. [Usage Examples](#usage-examples)

---
## System Setup

### Google Drive Integration

```python
from google.colab import drive
drive.mount('/content/drive')
```

**Purpose**: Mounts Google Drive to access input documents and save output files.

**Mount Point**: `/content/drive/MyDrive/`

---

## OCR Models

### Qari Model

**Model**: `NAMAA-Space/Qari-OCR-0.1-VL-2B-Instruct`

#### Dependencies
```bash
pip install transformers qwen_vl_utils accelerate>=0.26.0 PEFT -U
pip install -U bitsandbytes
```

#### Key Components

```python
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_name)
```

#### Configuration Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `max_tokens` | 2000 | Maximum tokens for OCR generation |
| `input_folder` | `/content/drive/MyDrive/test_data` | Source directory for images |
| `output_folder` | Same as input | Destination for OCR text files |

#### Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff, .tif)
- WebP (.webp)

#### Processing Pipeline

1. **Image Discovery**: Scans folder for all supported image formats (case-insensitive)
2. **Message Preparation**: Creates vision-language messages with image and prompt
3. **Processing**: 
   - Applies chat template
   - Processes vision information
   - Tokenizes and moves to CUDA
4. **Generation**: Generates OCR text with beam search
5. **Output**: Saves to text file with naming pattern: `{original_name}_{model_name}_ocr.txt`

#### Error Handling

```python
try:
    # Processing logic
except Exception as e:
    print(f"✗ Error processing {os.path.basename(image_path)}: {str(e)}")
    continue  # Continues with next image
```

### Nanonets OCR

**Model**: `nanonets/Nanonets-OCR-s`

#### Installation Methods

**Method 1: Using docext package**
```bash
uv pip install docext
python -m docext.app.app --model_name hosted_vllm/nanonets/Nanonets-OCR-s
```

**Method 2: HuggingFace Installation**
```python
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2",
)
```

#### API Server Configuration

```python
# Start server
python -m docext.app.app --max_model_len 15000 \
    --gpu_memory_utilization 0.9 \
    --max_num_imgs 1 \
    --max_img_size 1024 \
    --concurrency_limit 1
```

#### PDF to Markdown Conversion

```python
def convert_pdf_to_markdown(
    client_url: str,
    username: str,
    password: str,
    file_paths: list[str],
    model_name: str = "hosted_vllm/nanonets/Nanonets-OCR-s"
):
    """
    Convert PDF/images to markdown using the API
    
    Args:
        client_url: URL of the docext server
        username: Authentication username
        password: Authentication password
        file_paths: List of file paths to convert
        model_name: Model to use for conversion
    
    Returns:
        str: Converted markdown content
    """
    client = Client(client_url, auth=(username, password))
    file_inputs = [{"image": handle_file(file_path)} for file_path in file_paths]
    
    result = client.predict(
        images=file_inputs,
        api_name="/process_markdown_streaming"
    )
    
    return result
```

---

## Mapping Functionality (Detailed)

### Question-Answer-Choice Mapping

This is the core mapping functionality that extracts and links questions, their multiple-choice options, and correct answers from educational materials.

#### Overview

The mapping system uses **LangExtract** (`lx`) library to perform structured extraction and establish relationships between three entity types:

1. **Questions**: The question text
2. **Choices**: Multiple choice options (A, B, C, D)
3. **Answers**: The correct answer with explanation

#### Mapping Schema Definition

```python
from langextract import BaseEntity
from typing import Optional

class QuestionEntity(BaseEntity):
    """Represents a question in the exam"""
    entity_name: str = "question"
    extraction_hint: str = "The main question text being asked"
    
    class Attributes(BaseEntity.Attributes):
        question_number: Optional[str] = Field(
            None,
            description="The question number (e.g., 'Q1', '1', 'Question 1')"
        )
        topic: Optional[str] = Field(
            None,
            description="The topic or subject area of the question"
        )

class ChoiceEntity(BaseEntity):
    """Represents answer choices for a question"""
    entity_name: str = "choice"
    extraction_hint: str = "The multiple choice options (A, B, C, D) for the question"
    
    class Attributes(BaseEntity.Attributes):
        question_number: Optional[str] = Field(
            None,
            description="Links this choice to its question number"
        )

class AnswerEntity(BaseEntity):
    """Represents the correct answer for a question"""
    entity_name: str = "answer"
    extraction_hint: str = "The correct answer and explanation"
    
    class Attributes(BaseEntity.Attributes):
        question_number: Optional[str] = Field(
            None,
            description="Links this answer to its question number"
        )
        correct_choice: Optional[str] = Field(
            None,
            description="The letter of the correct choice (A, B, C, or D)"
        )
```

### Mapping Logic Overview

The mapping system works through **attribute-based linking** rather than explicit foreign keys:

```
Question (Q1) ←→ Choices (Q1) ←→ Answer (Q1)
     ↓                ↓                ↓
question_number  question_number  question_number
                                      + correct_choice
```

#### How Mapping Works

1. **Extraction Phase**:
```python
extraction_context = lx.ExtractionContext(
    entities=[QuestionEntity, ChoiceEntity, AnswerEntity],
    extraction_provider=lx.ExtractionProvider.GOOGLE_GENAI,
    api_key=api_key
)

result = lx.extract(input_file_path, extraction_context)
```

2. **Entity Separation**:
```python
questions = [e for e in result.entities if isinstance(e, QuestionEntity)]
choices = [e for e in result.entities if isinstance(e, ChoiceEntity)]
answers = [e for e in result.entities if isinstance(e, AnswerEntity)]
```

3. **Mapping by Attributes**:
```python
for i, question in enumerate(questions, 1):
    # Get question number from attributes
    q_num = question.attributes.get("question_number", str(i))
    
    # Find matching choices
    q_choices = [c for c in choices 
                 if c.attributes and 
                 c.attributes.get("question_number") == q_num]
    
    # Find matching answer
    q_answers = [a for a in answers 
                 if a.attributes and 
                 a.attributes.get("question_number") == q_num]
```

### Entity Extraction and Mapping

#### Extraction Context Setup

```python
extraction_context = lx.ExtractionContext(
    entities=[QuestionEntity, ChoiceEntity, AnswerEntity],
    extraction_provider=lx.ExtractionProvider.GOOGLE_GENAI,
    api_key=api_key
)
```

**Parameters**:
- `entities`: List of entity classes to extract
- `extraction_provider`: AI provider for extraction (Google Gemini)
- `api_key`: Authentication key for the provider

#### File Processing

```python
input_file_path = lx.io.load_file_from_path(file_path)
result = lx.extract(input_file_path, extraction_context)
```

**What happens**:
1. File is loaded into LangExtract's internal format
2. LLM analyzes the document structure
3. Entities are extracted with their attributes
4. Results are stored in `result.entities`

### Attribute-Based Mapping

#### Data Structure

Each entity has:
- `extraction_text`: The actual text content
- `attributes`: A dictionary of metadata

```python
# Example entity structure
question = QuestionEntity(
    extraction_text="What is the capital of France?",
    attributes={
        "question_number": "Q1",
        "topic": "Geography"
    }
)

choice = ChoiceEntity(
    extraction_text="A) Paris\nB) London\nC) Berlin\nD) Madrid",
    attributes={
        "question_number": "Q1"
    }
)

answer = AnswerEntity(
    extraction_text="Paris is the capital of France...",
    attributes={
        "question_number": "Q1",
        "correct_choice": "A"
    }
)
```

#### Mapping Algorithm

```python
def map_question_to_components(question, choices, answers):
    """
    Maps a question to its choices and answer using question_number attribute
    
    Args:
        question: QuestionEntity object
        choices: List of all ChoiceEntity objects
        answers: List of all AnswerEntity objects
    
    Returns:
        dict: Mapped components
    """
    q_num = question.attributes.get("question_number", "unknown")
    
    # Filter choices for this question
    related_choices = [
        c for c in choices 
        if c.attributes and c.attributes.get("question_number") == q_num
    ]
    
    # Filter answers for this question
    related_answers = [
        a for a in answers 
        if a.attributes and a.attributes.get("question_number") == q_num
    ]
    
    return {
        "question": question,
        "choices": related_choices,
        "answers": related_answers,
        "question_number": q_num
    }
```

#### Why This Mapping Approach?

**Advantages**:
1. **Flexibility**: Can handle varying document formats
2. **Robustness**: Doesn't rely on strict positioning
3. **LLM-Friendly**: Natural language understanding identifies relationships
4. **Extensible**: Easy to add new attributes

**How LLM Determines Mappings**:
- Analyzes document structure
- Identifies question numbers in various formats (Q1, 1., Question 1, etc.)
- Understands contextual relationships
- Extracts attributes automatically

### Display and Analysis

#### Formatted Output

```python
def analyze_results(result):
    """Display mapped Q&A with proper formatting"""
    questions = [e for e in result.entities if isinstance(e, QuestionEntity)]
    choices = [e for e in result.entities if isinstance(e, ChoiceEntity)]
    answers = [e for e in result.entities if isinstance(e, AnswerEntity)]
    
    for i, question in enumerate(questions, 1):
        q_num = question.attributes.get("question_number", str(i))
        
        print(f"\n🔸 Question {q_num}:")
        print(f"   {question.extraction_text.strip()}")
        
        # Map and display choices
        q_choices = [c for c in choices 
                     if c.attributes and 
                     c.attributes.get("question_number") == q_num]
        if q_choices:
            print(f"\n   📝 Choices:")
            for choice_line in q_choices[0].extraction_text.split('\n'):
                if choice_line.strip():
                    print(f"      {choice_line.strip()}")
        
        # Map and display answer
        q_answers = [a for a in answers 
                     if a.attributes and 
                     a.attributes.get("question_number") == q_num]
        if q_answers:
            correct_choice = q_answers[0].attributes.get("correct_choice", "")
            print(f"\n   ✅ Correct Answer: {correct_choice}")
            print(f"      {q_answers[0].extraction_text.strip()}")
        
        print("-" * 30)
```

#### Output Format Example

```
🔸 Question Q1:
   What is the capital of France?

   📝 Choices:
      A) Paris
      B) London
      C) Berlin
      D) Madrid

   ✅ Correct Answer: A
      Paris is the capital and largest city of France.
------------------------------
```

### Mapping Persistence

#### JSONL Export

```python
output_filename = f"{os.path.basename(file_path).split('.')[0]}_qa_extracted.jsonl"
lx.io.save_annotated_documents([result], 
                               output_name=output_filename, 
                               output_dir=".")
```

**Stored Information**:
- All entities with their text
- All attributes (including mapping keys)
- Relationships implicitly preserved through attributes

#### Visualization

```python
html_content = lx.visualize(output_filename)
html_filename = f"{os.path.basename(file_path).split('.')[0]}_qa_extracted_visualization.html"

with open(html_filename, "w", encoding='utf-8') as f:
    if hasattr(html_content, 'data'):
        f.write(html_content.data)
    else:
        f.write(str(html_content))
```

**Visualization Features**:
- Interactive HTML display
- Entity highlighting
- Relationship visualization
- Attribute inspection

---

## Embedding Models

### Sentence Transformers

**Model**: `sentence-transformers/all-MiniLM-L6-v2`

#### Basic Usage

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

sentences = [
    "That is a happy person",
    "That is a happy dog",
    "That is a very happy person",
    "Today is a sunny day"
]

# Generate embeddings
embeddings = model.encode(sentences)

# Calculate similarities
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)  # [4, 4]
```

#### Similarity Matrix

The `similarity` function returns a matrix where `similarities[i][j]` represents the cosine similarity between sentence `i` and sentence `j`.

**Matrix Interpretation**:
- Values range from -1 to 1
- Higher values indicate greater similarity
- Diagonal values are always 1 (self-similarity)

### HuggingFace Inference Client

```python
from huggingface_hub import InferenceClient

client = InferenceClient(
    provider="auto",
    api_key=os.environ["HF_TOKEN"],
)

result = client.sentence_similarity(
    sentences,
    sentences,
    model="sentence-transformers/all-MiniLM-L6-v2",
)
```

**Advantages**:
- No local model loading
- Cloud-based inference
- Supports multiple providers

### 3D Visualization of Embeddings

```python
from sklearn.decomposition import PCA
import plotly.express as px

# Reduce dimensions to 3 using PCA
pca = PCA(n_components=3)
reduced_embeddings = pca.fit_transform(embeddings)

# Create interactive 3D plot
fig = px.scatter_3d(
    x=reduced_embeddings[:, 0],
    y=reduced_embeddings[:, 1],
    z=reduced_embeddings[:, 2],
    text=sentences,
    title="3D Visualization of Sentence Embeddings"
)

fig.update_layout(
    scene=dict(
        xaxis_title='PCA Component 1',
        yaxis_title='PCA Component 2',
        zaxis_title='PCA Component 3'
    ),
    margin=dict(l=0, r=0, b=0, t=40)
)

fig.show()
```

**Purpose**: Visualizes high-dimensional embeddings in 3D space to understand semantic relationships.

---

## Code Structure

### Main Sections

1. **Setup and Installation** (Lines 1-20)
   - Google Drive mounting
   - Package installations

2. **Qari OCR Model** (Lines 21-122)
   - Model initialization
   - Batch image processing
   - Output generation

3. **Nanonets OCR** (Lines 124-382)
   - API server setup
   - PDF to Markdown conversion
   - Client-server architecture

4. **Question-Answer Extraction** (Lines 383-2200)
   - Entity definitions
   - Mapping logic
   - Visualization

5. **Embedding Models** (Lines 2227-2310)
   - Sentence transformers
   - Similarity computation
   - Visualization

6. **DotsOCR Setup** (Lines 2311-2382)
   - vLLM server configuration
   - Model deployment

---

## Usage Examples

### Complete Q&A Extraction Pipeline

```python
def main(file_path, api_key=None):
    try:
        # Extract Q&A data
        print("🚀 Starting extraction...")
        result = extract_qa_from_file(file_path, api_key)
        
        # Analyze results (performs mapping)
        analyze_results(result)
        
        # Save results
        output_filename = f"{os.path.basename(file_path).split('.')[0]}_qa_extracted.jsonl"
        lx.io.save_annotated_documents([result], 
                                       output_name=output_filename, 
                                       output_dir=".")
        print(f"\n💾 Results saved to: {output_filename}")
        
        # Generate visualization
        html_content = lx.visualize(output_filename)
        html_filename = f"{os.path.basename(file_path).split('.')[0]}_qa_extracted_visualization.html"
        
        with open(html_filename, "w", encoding='utf-8') as f:
            if hasattr(html_content, 'data'):
                f.write(html_content.data)
            else:
                f.write(str(html_content))
        print(f"📊 Visualization saved to: {html_filename}")
        
        return result
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return None
    except ValueError as e:
        print(f"❌ Error: {e}")
        return None
    except Exception as e:
        print(f"❌ Error during extraction: {str(e)}")
        return None

# Usage
if __name__=="__main__":
    file_path = "/content/drive/MyDrive/Extracted Text from Embedding Model (3).txt"
    api_key = os.environ.get("GEMINI_API_KEY")
    main(file_path, api_key)
```

### Batch Image OCR

```python
# Configure paths
input_folder = "/content/drive/MyDrive/test_data"
output_folder = input_folder

# Process all images
image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif', '*.webp']
image_files = []

for extension in image_extensions:
    image_files.extend(glob.glob(os.path.join(input_folder, extension)))
    image_files.extend(glob.glob(os.path.join(input_folder, extension.upper())))

# Process each image
for i, image_path in enumerate(image_files):
    try:
        # OCR processing logic here
        pass
    except Exception as e:
        print(f"Error: {e}")
        continue
```

---

## Key Takeaways

### Mapping System Summary

1. **Attribute-Based**: Uses `question_number` as the primary linking key
2. **LLM-Powered**: Leverages AI to identify and extract relationships
3. **Flexible**: Handles various document formats and numbering schemes
4. **Three-Entity Model**: Questions, Choices, and Answers
5. **Implicit Relationships**: Links established through shared attributes rather than explicit references

### Best Practices

1. **API Keys**: Store sensitive keys in environment variables
2. **Error Handling**: Always use try-except blocks for file operations
3. **Memory Management**: Clear CUDA cache after processing batches
4. **Output Organization**: Use descriptive filenames with model names
5. **Visualization**: Generate HTML visualizations for quality checking

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or `gpu_memory_utilization`
2. **Missing Entities**: Ensure document structure is clear
3. **Incorrect Mappings**: Verify question numbering is consistent
4. **API Rate Limits**: Implement delays between requests

---

## Dependencies

### Required Packages

```
transformers>=4.52.4
qwen_vl_utils
accelerate>=0.26.0
PEFT
bitsandbytes
flash-attn>=2.7.4.post1
sentence-transformers
huggingface-hub
langextract
plotly
scikit-learn
Pillow
rich
gradio-client
```

### Hardware Requirements

- **GPU**: CUDA-capable GPU (8GB+ VRAM recommended)
- **RAM**: 16GB+ system RAM
- **Storage**: 10GB+ for model weights

---

## Error Handling

### File Not Found

```python
try:
    result = extract_qa_from_file(file_path, api_key)
except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    print("Please check if the file path is correct.")
```

### Empty Content

```python
except ValueError as e:
    print(f"❌ Error: {e}")
    print("The input file seems to be empty or contains no readable content.")
```

### General Exceptions

```python
except Exception as e:
    print(f"❌ Error during extraction: {str(e)}")
    return None
```

---

## Conclusion

This embedding model system provides a comprehensive solution for:
- OCR from images and PDFs
- Structured extraction of educational content
- Intelligent mapping of questions to answers
- Semantic embedding generation
- Interactive visualization

The mapping functionality is particularly powerful for creating educational datasets, exam analysis, and automated grading systems.