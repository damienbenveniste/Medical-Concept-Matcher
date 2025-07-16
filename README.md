# Medical Concept Matcher

A sophisticated AI-powered system for matching medical concepts to standardized medical codes across multiple vocabularies including ICD-10, CPT, LOINC, and ATC classifications.

## 🎯 Overview

The Medical Concept Matcher is an intelligent system that takes free-text medical concepts (like "chest pain" or "blood pressure measurement") and matches them to appropriate standardized medical codes. It uses a multi-stage AI pipeline with retrieval-augmented generation (RAG) to ensure accurate and clinically relevant code assignments.

## 🏗️ Architecture

### Core Components

1. **Concept Router**: Determines the type of medical concept (diagnosis, procedure, measurement, drug)
2. **Retrieval System**: Finds candidate codes using semantic search with embeddings
3. **Code Selector**: Uses GPT-4 to select the most appropriate codes from candidates
4. **Validator**: Performs final validation to ensure clinical accuracy

### Technology Stack

- **Backend**: FastAPI with Python 3.13
- **AI Models**: OpenAI GPT-4.1 for concept classification and code selection
- **Embeddings**: OpenAI text-embedding-3-small for semantic search
- **Vector Database**: ChromaDB for storing and searching code embeddings
- **Workflow**: LangGraph for orchestrating the multi-stage pipeline

## 📁 Project Structure

```
Medical-concept-matcher/
├── backend/                    # Main application
│   ├── app/
│   │   ├── concept/           # Concept matching logic
│   │   │   ├── api.py         # REST API endpoints
│   │   │   └── matcher_agent/ # AI pipeline components
│   │   │       ├── agent.py   # LangGraph workflow
│   │   │       ├── state.py   # State management
│   │   │       └── nodes/     # Pipeline stages
│   │   │           ├── concept_router.py  # Concept type classification
│   │   │           ├── retrieval.py       # Semantic search
│   │   │           ├── selector.py        # Code selection
│   │   │           └── validation.py      # Final validation
│   │   ├── indexing/          # Data indexing system
│   │   │   ├── api.py         # Indexing endpoints
│   │   │   └── indexer.py     # Vector database operations
│   │   ├── clients.py         # OpenAI client configuration
│   │   └── main.py           # FastAPI application
│   ├── data/                  # Medical vocabulary files
│   │   ├── icd10cm_codes_2026.txt         # ICD-10 diagnosis codes
│   │   ├── 2025_DHS_Code_List_Addendum_11_26_2024.txt  # CPT procedure codes
│   │   ├── LoincTableCore.csv             # LOINC measurement codes
│   │   └── WHO ATC-DDD 2024-07-31.csv     # ATC drug codes
│   └── chroma/                # ChromaDB storage
└── evaluation/                # Evaluation framework
    ├── src/
    │   ├── data.py            # Test data preparation
    │   └── evaluate.py        # Performance evaluation
    ├── test_data/             # Sampled test datasets
    └── results/               # Evaluation results
```

## 🚀 Quick Start

### Prerequisites

- Python 3.13+
- OpenAI API key
- UV package manager (recommended) or pip

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd Medical-concept-matcher
```

2. **Set up environment variables**
```bash
# Create a .env file in the backend directory
echo "OPENAI_API_KEY=your_openai_api_key_here" > backend/.env
```

3. **Install backend dependencies**
```bash
cd backend
uv sync  # or pip install -e .
```

4. **Install evaluation dependencies**
```bash
cd ../evaluation
uv sync  # or pip install -e .
```

### Running the Application

1. **Start the API server**
```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

2. **Index the medical vocabularies** (first time only)
```bash
curl -X POST "http://localhost:8000/indexing/index"
```

3. **Test the API**
```bash
curl -X POST "http://localhost:8000/concept/match" \
     -H "Content-Type: application/json" \
     -d '{"concept": "chest pain"}'
```

## 📊 Medical Vocabularies Supported

### ICD-10-CM (International Classification of Diseases)
- **Purpose**: Diagnosis codes
- **Example**: `I20.9` - Angina pectoris, unspecified
- **Count**: ~70,000 codes

### CPT (Current Procedural Terminology)
- **Purpose**: Medical procedures and services
- **Example**: `99213` - Office visit, established patient
- **Count**: ~10,000 codes

### LOINC (Logical Observation Identifiers Names and Codes)
- **Purpose**: Laboratory tests and measurements
- **Example**: `33747-0` - Cholesterol [Mass/volume] in Serum or Plasma
- **Count**: ~90,000 codes

### ATC (Anatomical Therapeutic Chemical)
- **Purpose**: Drug classifications
- **Example**: `C09AA01` - Captopril
- **Count**: ~6,000 codes

## 🔄 Workflow Pipeline

### 1. Concept Classification
```
Input: "chest pain"
↓
Concept Router (GPT-4.1)
↓
Output: concept_type = "diagnose"
```

### 2. Semantic Retrieval
```
Input: "chest pain" + concept_type
↓
Embedding Generation (OpenAI)
↓
Vector Search (ChromaDB)
↓
Output: Top 20 candidate ICD codes
```

### 3. Code Selection
```
Input: candidates + concept
↓
Code Selector (GPT-4.1)
↓
Output: Best matching codes with confidence scores
```

### 4. Validation
```
Input: selected codes + concept
↓
Validator (GPT-4.1)
↓
Output: Clinically validated final codes
```

## 🧪 Evaluation Framework

### Evaluation Design

The evaluation system is designed to comprehensively assess the Medical Concept Matcher's performance across different medical vocabularies using a robust sampling and testing methodology.

#### Test Data Generation
1. **Random Sampling**: 100 codes are randomly selected from each vocabulary (ICD, CPT, LOINC, ATC)
2. **Ground Truth**: Each test item uses the official code description as the input concept
3. **Expected Output**: The system should return the original code among its predictions
4. **Cross-Validation**: Tests cover diverse medical domains and complexity levels

#### Evaluation Methodology
```
For each test item:
1. Input: Official code description (e.g., "Angina pectoris, unspecified")
2. Expected: System should return the original code (e.g., "I20.9")
3. Measure: Whether the correct code appears in the predictions
4. Metrics: Calculate precision, recall, and F1 scores
```

#### Multi-Dataset Testing
- **ICD-10**: 100 diagnosis codes testing symptom-to-diagnosis matching
- **CPT**: 100 procedure codes testing procedure description matching  
- **LOINC**: 100 laboratory codes testing measurement concept matching
- **ATC**: 100 drug codes testing medication concept matching

#### Parallel Processing
- **Concurrent Evaluation**: All test items processed simultaneously using asyncio
- **Exception Handling**: Failed requests don't halt the entire evaluation
- **Performance Tracking**: Response times and success rates monitored

### Running Evaluations

1. **Prepare test data**
```bash
cd evaluation
python src/data.py
```

2. **Run evaluation**
```bash
python src/evaluate.py
```

### Metrics Explained

#### Core Metrics
- **Accuracy**: Percentage of test cases where the correct code was found
  - Formula: `correct_matches / total_test_cases`
  - Interpretation: Overall system effectiveness

- **Precision**: Quality of predictions (how many predictions were correct)
  - Formula: `true_positives / (true_positives + false_positives)`
  - Interpretation: Reduces over-prediction

- **Recall**: Coverage of correct answers (how many correct codes were found)
  - Formula: `true_positives / (true_positives + false_negatives)`
  - Interpretation: Reduces under-prediction

- **F1 Score**: Harmonic mean balancing precision and recall
  - Formula: `2 * (precision * recall) / (precision + recall)`
  - Interpretation: Overall balanced performance

#### Calculation Details
- **True Positives**: Correct code found in predictions
- **False Positives**: Incorrect codes returned as predictions
- **False Negatives**: Correct codes not found in predictions
- **True Negatives**: Not applicable (infinite set of non-matching codes)

### Sample Results

```
Dataset: ICD
Total items: 100
Accuracy: 85.00% (85/100)
Precision: 92.31% (72/78)
Recall: 85.00% (85/100)
F1 Score: 88.52%

Overall Performance:
- High precision indicates few false positives
- Good recall shows most correct codes are found
- Balanced F1 score demonstrates robust performance
```

### Evaluation Outputs

#### Detailed Results File
- **Per-Dataset Metrics**: Individual performance for each vocabulary
- **Overall Summary**: Aggregated metrics across all vocabularies
- **Statistical Analysis**: Confidence intervals and significance testing
- **Error Analysis**: Common failure patterns and improvement opportunities

#### Result Interpretation
- **>90% Accuracy**: Excellent performance, production-ready
- **80-90% Accuracy**: Good performance, minor improvements needed
- **70-80% Accuracy**: Moderate performance, significant improvements required
- **<70% Accuracy**: Poor performance, major system revision needed

## 🔧 API Reference

### Endpoints

#### POST `/concept/match`
Match a medical concept to standardized codes.

**Request Body:**
```json
{
  "concept": "chest pain"
}
```

**Response:**
```json
{
  "concept": [
    {
      "code": "I20.9",
      "text": "Angina pectoris, unspecified",
      "confidence": 95,
      "concept_type": "diagnose"
    }
  ]
}
```

#### POST `/indexing/index`
Index medical vocabularies (admin endpoint).

**Response:**
```json
{
  "status": "success",
  "message": "Documents have been successfully indexed and are ready for search"
}
```

### Interactive API Documentation

Visit `http://localhost:8000/docs` for interactive API documentation.

## 🏥 Clinical Use Cases

### Primary Care
- **Symptom Documentation**: Convert patient-reported symptoms to ICD codes
- **Procedure Coding**: Match procedures to CPT codes for billing
- **Lab Orders**: Find appropriate LOINC codes for laboratory tests

### Hospital Systems
- **EHR Integration**: Automated code suggestion in electronic health records
- **Clinical Decision Support**: Assist clinicians with accurate coding
- **Quality Assurance**: Validate existing code assignments

### Research
- **Clinical Trial Matching**: Identify patients using standardized codes
- **Epidemiological Studies**: Consistent coding across datasets
- **Healthcare Analytics**: Standardized data analysis

## 🔒 Security & Privacy

- **API Keys**: Secure OpenAI API key management via environment variables
- **Data Processing**: No patient data stored; only processes concept text
- **Validation**: Multi-stage validation ensures clinical accuracy
- **Logging**: Comprehensive error logging for debugging

## 🚀 Performance Optimization

### Batch Processing
- **Embeddings**: Generated in batches of 1000 for efficiency
- **Database Operations**: Chunked inserts to respect ChromaDB limits
- **Parallel Evaluation**: Concurrent processing for faster evaluations

### Caching
- **Vector Storage**: Persistent ChromaDB storage for fast retrieval
- **Embeddings**: Pre-computed embeddings for all medical codes
- **API Responses**: Structured response caching (implementation ready)

## 🛠️ Development

### Code Quality
- **Type Hints**: Comprehensive type annotations throughout
- **Documentation**: Google-style docstrings for all functions
- **Error Handling**: Robust exception handling and logging
- **Testing**: Evaluation framework with multiple test datasets

### Contributing
1. Fork the repository
2. Create a feature branch
3. Add comprehensive tests
4. Ensure type hints and docstrings
5. Submit a pull request

## 📈 Future Enhancements

### Planned Features
- **SNOMED CT Integration**: Support for SNOMED Clinical Terms
- **ICD-11 Support**: Next-generation ICD codes
- **Multi-language Support**: Non-English medical concepts
- **Confidence Thresholds**: Configurable confidence levels
- **Batch API**: Process multiple concepts simultaneously

### Performance Improvements
- **Model Fine-tuning**: Domain-specific model training
- **Hybrid Search**: Combine semantic and keyword search
- **Caching Layer**: Response caching for common queries
- **Load Balancing**: Distributed processing capabilities

## 🤝 Support

For questions, issues, or contributions:
- **Documentation**: Check inline code documentation
- **Issues**: Use the evaluation framework to test specific cases
- **Performance**: Monitor evaluation metrics for system health

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **OpenAI**: For providing the GPT-4.1 and embedding models
- **ChromaDB**: For the vector database infrastructure
- **FastAPI**: For the high-performance web framework
- **Medical Vocabularies**: ICD-10, CPT, LOINC, and ATC organizations
- **LangGraph**: For workflow orchestration capabilities

---

*Built with ❤️ for the healthcare community*