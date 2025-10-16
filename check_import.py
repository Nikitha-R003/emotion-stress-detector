try:
    from transformers import pipeline
    print("Import successful")
except ImportError as e:
    print(f"Import error: {e}")

import transformers
print(f"Version: {transformers.__version__}")
