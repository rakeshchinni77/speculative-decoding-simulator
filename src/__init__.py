"""
Speculative Decoding Simulator and Latency Analysis Package.
"""

import os

os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["USE_TORCH"] = "1"

__version__ = "1.0.0"
