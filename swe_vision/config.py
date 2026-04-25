"""
Configuration constants, logging setup, tool definitions, and system prompt.
"""

import datetime
import logging
import os

from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("vlm_agent")

# Load .env from project root so direct Python/IDE runs also get env vars.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DOTENV_PATH = os.path.join(_PROJECT_ROOT, ".env")

load_dotenv(_DOTENV_PATH, override=False)

# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────
DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")
MAX_ITERATIONS = 100
DEFAULT_MAX_HISTORY = 5
CELL_TIMEOUT = 120.0
MAX_OUTPUT_CHARS = 50000

# ─────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────
SUMMARY_SYSTEM_PROMPT = """\
You are a conversation summarizer for an AI coding assistant session.
Summarize the following conversation into a concise summary that captures:
1. User's questions and intent
2. Code that was executed and key results
3. Important variables/data currently in the Jupyter kernel
4. Key findings, decisions, or errors encountered

If a previous summary is provided, incorporate its information into the new summary.
Keep the summary concise but comprehensive enough to continue the conversation without losing critical context.
Respond with the summary only, no preamble.\
"""

# Container-side working directory (visible to the kernel)
CONTAINER_WORK_DIR = "/mnt/data"

# Host-side directory that is volume-mounted into the container.
_HOST_WORK_BASE = os.environ.get(
    "VLM_HOST_WORK_DIR",
    os.path.join(os.path.expanduser("~"), "tmp", "vlm_docker_workdir"),
)
HOST_WORK_DIR = os.path.join(
    _HOST_WORK_BASE,
    datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
)

# Docker image / build settings
DOCKER_IMAGE_NAME = os.environ.get("VLM_DOCKER_IMAGE", "swe-vision:latest")

DOCKERFILE_DIR = os.environ.get(
    "VLM_DOCKERFILE_DIR",
    os.path.join(_PROJECT_ROOT, "env"),
)

# Pre-assigned ZMQ ports for the Jupyter kernel inside the container
_KERNEL_BASE_PORT = 65500
KERNEL_PORTS = {
    "shell_port":   _KERNEL_BASE_PORT,
    "iopub_port":   _KERNEL_BASE_PORT + 1,
    "stdin_port":   _KERNEL_BASE_PORT + 2,
    "control_port": _KERNEL_BASE_PORT + 3,
    "hb_port":      _KERNEL_BASE_PORT + 4,
}

# ─────────────────────────────────────────────────────────────────────
# Tool Definitions (OpenAI function calling format)
# ─────────────────────────────────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_code",
            "description": (
                "Execute Python code in a **stateful** Jupyter notebook environment. "
                "The kernel persists across calls, so variables, imports, and state are retained. "
                "Use this to process images, perform calculations, create visualizations, "
                "or run any Python code. "
                "Any images generated (e.g. via matplotlib plt.show() or PIL Image.save()) "
                "will be captured and returned as base64-encoded images."
                "Print statements and expression results are captured as text output. "
                "All uploaded files are available under /mnt/data/. "
                "The kernel's working directory is /mnt/data/."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": (
                            "The Python code to execute. The code runs in a Jupyter kernel "
                            "so you can use magics, display(), etc. "
                            "Use print() for text output. "
                            "Images from matplotlib will be auto-captured."
                        ),
                    },
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": (
                "Call this tool when you have determined the final answer. "
                "This ends the agentic workflow and returns the answer to the user."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "The final answer to the user's question.",
                    },
                },
                "required": ["answer"],
            },
        },
    },
]

# ─────────────────────────────────────────────────────────────────────
# System Prompt
# ─────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are an expert AI assistant with access to a **stateful Jupyter notebook** environment. \
You can execute Python code to help answer the user's questions.

## Available Tools

1. **execute_code**: Run Python code in a persistent Jupyter notebook. The kernel state \
(variables, imports, loaded data) is preserved between calls. Use this for:
   - Image processing and analysis (PIL/Pillow, OpenCV, skimage, etc.)
   - Data analysis and computation (numpy, pandas, scipy, etc.)
   - Visualization (matplotlib, seaborn, plotly, etc.)
   - Any Python computation

2. **finish**: Call this when you have the final answer. This ends the workflow.

## File System

- All uploaded files (images, data files, etc.) are placed in `/mnt/data/`.
- The Jupyter kernel's working directory is `/mnt/data/`, so you can reference files \
by their filename directly (e.g. `open('image.png')`) or by absolute path \
(e.g. `open('/mnt/data/image.png')`).
- Any files you create or save will also go into `/mnt/data/`.

## Guidelines

- When given an image, you can load it in the notebook using PIL or OpenCV. \
The image file will be available at `/mnt/data/<filename>`.
- You can call execute_code **multiple times** to iteratively explore and process data.
- Always use print() to output results you want to see.
- When you generate plots with matplotlib, use plt.show() — the plot image will be \
captured and returned to you.
- Think step by step. Examine intermediate results before giving a final answer.
- When you're confident in your answer, call the **finish** tool with your final response.
- If code produces an error, analyze the error and try a different approach.
"""
