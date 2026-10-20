"""Evaluation suite. Set DeepEval's switches BEFORE it's imported anywhere:

- DEEPEVAL_TELEMETRY_OPT_OUT: DeepEval reports usage telemetry by default.
- DEEPEVAL_DISABLE_DOTENV: DeepEval copies .env into os.environ at import time. Our settings already read
  .env; copying it into the environment would hand every secret to every child process (the Docker MCP
  containers are deliberately given only ONE token each).
"""

import os

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
os.environ.setdefault("DEEPEVAL_DISABLE_DOTENV", "1")
