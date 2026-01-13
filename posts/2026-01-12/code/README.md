# LangGraph Error Handling Examples

Companion code for the blog post demonstrating four error handling patterns with LangGraph workflows.

## Quick Start

### 1. Install Ollama

```bash
# Install from https://ollama.ai
ollama pull llama3.2
ollama serve
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
cp .env.example .env
```

Optional:
- `LANGCHAIN_API_KEY` for LangSmith Studio tracing

### 3. Run Error Pattern Tests

Test all 4 patterns:
```bash
python test_error_patterns.py
```

Test specific pattern:
```bash
python test_error_patterns.py --test 1  # Retry with backoff
python test_error_patterns.py --test 2  # LLM-guided recovery
python test_error_patterns.py --test 3  # Human-in-the-loop
python test_error_patterns.py --test 4  # Unexpected failures
```

## Files

- **test_error_patterns.py** - Demonstrates all 4 error handling patterns
- **email_agent.py** - Simple email agent workflow used for examples
- **circular_recovery.py** - LLM-guided circular routing pattern (Test 2)
- **error_simulation.py** - Error simulation utilities for testing
- **workflow_builders.py** - Workflow construction helpers
