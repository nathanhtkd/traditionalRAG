# RAG Benchmarking Tool

A comprehensive tool for benchmarking different RAG (Retrieval-Augmented Generation) implementations, focusing on document ingestion, processing, and query performance.

## Features

- Document ingestion and processing
- Multiple embedding model support
- Parallel processing capabilities
- Performance benchmarking
- Query interface with multiple LLM options
- Visualization of benchmark results

## Installation

1. Clone the repository:

```bash
git clone https://github.com/nathanhtkd/traditionalRAG.git
```

```bash
cd traditionalRAG
```

2. Create a virtual environment and activate it:

```bash
python -m venv venv
source venv/bin/activate
```

3. Install the dependencies:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory and add the following:

```bash
GROQ_API_KEY=<your-groq-api-key>
```

## Project Structure

- `main.py`: Core RAG implementation and benchmarking logic
- `download.py`: Utility for downloading test documents
- `graph.py`: Visualization tools for benchmark results
- `docs/`: Directory for downloaded documents
- `cache/`: Cached embeddings and intermediate results

## Implementation Details

### Document Processing
- Chunks documents using RecursiveCharacterTextSplitter
- Default chunk size: 400 characters
- Chunk overlap: 100 characters

### Vector Storage
- Uses Chroma for vector storage
- Implements cosine similarity for document retrieval
- Temporary storage with secure permissions

### Query Processing
- Supports multiple LLM models
- Implements relevance scoring
- Returns source documents with responses

To run the tool, use the following command:

```bash
python main.py
```

## Benchmark Results

The tool generates benchmark results comparing:
- Document ingestion time
- Processing efficiency
- Query response time

Results are saved in:
- CSV format (`ingestion_benchmarks.csv`)
- Visualization (`ingestion_time_comparison.png`)

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
