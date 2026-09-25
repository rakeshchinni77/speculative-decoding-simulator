"""
Pytest configuration and environment setup.
"""

import os

# Disable TensorFlow and Flax discovery in transformers
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["USE_TORCH"] = "1"
