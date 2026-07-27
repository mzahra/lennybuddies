import os
import sys
from pathlib import Path

# Make src/ importable as top-level modules (document_processor,
# knowledge_base, prompt_templates, llm_integration) without needing an
# installed package.
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

# document_processor.py and llm_integration.py both validate OPENAI_API_KEY
# at import time. Set a dummy value so importing them during test
# collection never fails, even in CI where a real .env isn't present.
# Individual tests still mock out the actual API calls.
os.environ.setdefault("OPENAI_API_KEY", "test-key-for-unit-tests")
