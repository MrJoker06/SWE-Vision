"""
VLM Tool Call Agent — agentic VLM framework with Docker Jupyter notebook tool.

The agent loop:
1. Send user message (with optional images) to the VLM
2. If the model calls ``execute_code``, run the code in the Docker kernel
3. Feed results (text + images) back to the model
4. Repeat until the model calls ``finish`` or max iterations reached
"""

import datetime
import json
import os
import traceback
from typing import Any, Dict, List, Optional

from openai import OpenAI

from swe_vision.config import (
    DEFAULT_MAX_HISTORY,
    DEFAULT_MODEL,
    MAX_ITERATIONS,
    SUMMARY_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    TOOLS,
    logger,
)
from swe_vision.file_manager import NotebookFileManager
from swe_vision.image_utils import make_base64_image_content_part, make_image_content_part
from swe_vision.kernel import JupyterNotebookKernel
from swe_vision.trajectory import TrajectoryRecorder


class VLMToolCallAgent:
    """
    An agentic VLM framework that uses OpenAI's function calling to
    give a vision-language model access to a stateful Jupyter notebook
    running inside a Docker container.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        system_prompt: str = SYSTEM_PROMPT,
        max_iterations: int = MAX_ITERATIONS,
        verbose: bool = True,
        save_trajectory: Optional[str] = None,
        reasoning: bool = True,
        max_history: int = DEFAULT_MAX_HISTORY,
        summary_model: Optional[str] = None,
    ):
        self.model = model
        self.max_iterations = max_iterations
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.reasoning = reasoning
        self.max_history = max_history
        self.summary_model = summary_model

        self._save_trajectory_dir = save_trajectory

        client_kwargs = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url
        elif os.environ.get("OPENAI_BASE_URL"):
            client_kwargs["base_url"] = os.environ["OPENAI_BASE_URL"]

        self.client = OpenAI(**client_kwargs)

        effective_api_key = api_key or os.environ.get("OPENAI_API_KEY")
        effective_base_url = base_url or client_kwargs.get("base_url")

        print(f"Using model: {self.model}")
        print(f"Using API key: {'set' if effective_api_key else 'None'}")
        print(f"Using base URL: {effective_base_url or 'OpenAI default'}")

        self.kernel: Optional[JupyterNotebookKernel] = None
        self.file_manager = NotebookFileManager()

        self.messages: List[Dict[str, Any]] = []

        self.trajectory: Optional[TrajectoryRecorder] = None

    async def _ensure_kernel(self):
        if self.kernel is None:
            self.kernel = JupyterNotebookKernel()
        if not self.kernel._started:
            await self.kernel.start()
            self.file_manager.setup_work_dir(
                host_work_dir=self.kernel.host_work_dir,
                container_work_dir=self.kernel.container_work_dir,
            )

    def _log(self, msg: str, *args, level: str = "info"):
        getattr(logger, level)(msg, *args)
        if self.verbose:
            formatted = msg % args if args else msg
            print(f"  [{level.upper()}] {formatted}", flush=True)

    def _build_user_message(
        self,
        query: str,
        image_paths: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        content = []

        file_hints = []
        if image_paths:
            basenames = [os.path.basename(os.path.abspath(p)) for p in image_paths]
            has_collision = len(basenames) != len(set(basenames))

            for idx, img_path in enumerate(image_paths):
                img_path = os.path.abspath(img_path)
                if not os.path.exists(img_path):
                    self._log("Warning: image not found: %s", img_path, level="warning")
                    continue
                # 将图片转换成字节码传输给model(包含大于20MB的压缩操作)
                # 但是容器拿到的图片是挂载过去的原图(/mnt/data/...)
                content.append(make_image_content_part(img_path))
                dest_name = None
                if has_collision or len(image_paths) > 1:
                    base = os.path.basename(img_path)
                    name, ext = os.path.splitext(base)
                    dest_name = f"{idx}_{name}{ext}"
                container_path = self.file_manager.copy_file_to_workdir(
                    img_path, dest_name=dest_name,
                )
                file_hints.append(container_path)

        text = query
        if file_hints:
            paths_str = ", ".join(f"`{p}`" for p in file_hints)
            text += f"\n\n[Uploaded file(s) available at: {paths_str}]"
        content.insert(0, {"type": "text", "text": text})

        return {"role": "user", "content": content}

    # ── Interactive memory / summary helpers ───────────────────────

    def _count_history_messages(self) -> int:
        """Count messages excluding system prompt and summary."""
        return sum(
            1 for m in self.messages
            if m.get("role") != "system" and not m.get("_is_summary")
        )

    def _extract_existing_summary(self) -> Optional[str]:
        """Extract the text content of the existing summary message, if any."""
        for msg in self.messages:
            if msg.get("_is_summary"):
                content = msg.get("content", "")
                # Strip the prefix marker
                prefix = "[Conversation Summary]\n"
                if content.startswith(prefix):
                    return content[len(prefix):]
                return content
        return None

    def _get_history_messages(self) -> List[Dict[str, Any]]:
        """Return all non-system, non-summary messages."""
        return [
            m for m in self.messages
            if m.get("role") != "system" and not m.get("_is_summary")
        ]

    @staticmethod
    def _strip_images_from_content(content) -> str:
        """Replace image content parts with text placeholders."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "image_url":
                        parts.append("[Image]")
                    elif item.get("type") == "text":
                        parts.append(item.get("text", ""))
                    else:
                        parts.append(str(item))
                else:
                    parts.append(str(item))
            return "\n".join(parts)
        return str(content)

    def _format_messages_for_summary(
        self, messages: List[Dict[str, Any]],
    ) -> str:
        """Format a list of messages into a readable text block for the summary LLM."""
        lines = []
        for msg in messages:
            role = msg.get("role", "unknown").upper()
            content = self._strip_images_from_content(msg.get("content", ""))
            # Truncate very long tool outputs to keep summary request manageable
            if role == "TOOL" and len(content) > 2000:
                content = content[:2000] + "\n... [truncated]"
            lines.append(f"[{role}] {content}")
        return "\n\n".join(lines)

    async def _maybe_summarize(self):
        """
        Check if history exceeds max_history; if so, summarize and compact.

        After summarization, self.messages becomes:
            [system_prompt, summary_message]
        """
        if self.max_history <= 0:
            return
        if self._count_history_messages() < self.max_history:
            return

        self._log("History reached %d messages, triggering summarization...",
                   self._count_history_messages())

        # 1. Extract old summary (if any)
        old_summary = self._extract_existing_summary()

        # 2. Build summary input
        history_msgs = self._get_history_messages()
        formatted = self._format_messages_for_summary(history_msgs)

        summary_input = ""
        if old_summary:
            summary_input += f"Previous summary:\n{old_summary}\n\n"
        summary_input += f"Conversation to summarize:\n{formatted}"

        # 3. Call LLM to generate summary
        model = self.summary_model or self.model
        self._log("Generating summary with model: %s", model)
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                    {"role": "user", "content": summary_input},
                ],
            )
            new_summary = response.choices[0].message.content
        except Exception as e:
            self._log("Summary generation failed: %s. Keeping history as-is.", e,
                       level="warning")
            return

        # 4. Rebuild messages: [system] + [summary]
        self.messages = [
            self.messages[0],  # system prompt
            {
                "role": "user",
                "content": f"[Conversation Summary]\n{new_summary}",
                "_is_summary": True,
            },
        ]
        self._log("Summarization complete. History compacted.")

    def _call_llm(self) -> Any:
        # Strip internal metadata fields (e.g. _is_summary) before sending
        # to the API — OpenAI rejects unknown keys.
        clean_messages = []
        for msg in self.messages:
            if msg.get("_is_summary"):
                clean = {k: v for k, v in msg.items() if k != "_is_summary"}
                clean_messages.append(clean)
            else:
                clean_messages.append(msg)

        kwargs = dict(
            model=self.model,
            messages=clean_messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        if self.reasoning:
            # kwargs["extra_body"] = {"reasoning": {"enabled": True, 'effort': 'xhigh'}}
            kwargs["reasoning_effort"] = 'xhigh'
        else:
            kwargs["extra_body"] = {"reasoning": {"enabled": False, 'effort': 'minimal'}}

        response = self.client.chat.completions.create(**kwargs)
        return response

    async def _handle_execute_code(self, code: str) -> Dict[str, Any]:
        await self._ensure_kernel()

        self._log("Executing code in Docker Jupyter notebook:\n%s",
                   code[:200] + ("..." if len(code) > 200 else ""))

        result = await self.kernel.execute(code)

        text = result["text_output"]
        if result["status"] == "error":
            text = f"[Execution Error]\n{text}"

        image_parts = []
        for img_b64 in result["images"]:
            image_parts.append(make_base64_image_content_part(img_b64))

        return {
            "text_output": text,
            "image_parts": image_parts,
            "base64_images": result["images"],
        }

    def _init_trajectory(self, query: str, image_paths: Optional[List[str]]) -> TrajectoryRecorder:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if self._save_trajectory_dir:
            save_dir = f"{self._save_trajectory_dir}_{ts}"
        else:
            save_dir = os.path.join("trajectories", f"run_{ts}")
        recorder = TrajectoryRecorder(save_dir)
        recorder.set_metadata(
            model=self.model,
            start_time=TrajectoryRecorder._now_iso(),
            query=query,
            image_paths=image_paths or [],
            max_iterations=self.max_iterations,
            system_prompt=self.system_prompt,
        )
        return recorder

    async def run(
        self,
        query: str,
        image_paths: Optional[List[str]] = None,
    ) -> str:
        """
        Run the agentic loop for a single user query.

        Returns the final answer string.
        """
        self.trajectory = self._init_trajectory(query, image_paths)

        self.messages = [
            {"role": "system", "content": self.system_prompt},
        ]

        user_msg = self._build_user_message(query, image_paths)
        self.messages.append(user_msg)

        self.trajectory.record_user_step(query, image_paths)

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"User Query: {query}")
            if image_paths:
                print(f"Images: {image_paths}")
            print(f"{'='*60}\n")

        final_answer = None
        try:
            final_answer = await self._run_loop()
        finally:
            if final_answer is not None:
                self.trajectory.record_finish(final_answer)
            self.trajectory.save()
            self.trajectory.save_messages_raw(self.messages)

        return final_answer

    async def _run_loop(self) -> str:
        """Core agentic loop."""
        for iteration in range(1, self.max_iterations + 1):
            if self.verbose:
                print(f"\n--- Iteration {iteration}/{self.max_iterations} ---")

            MAX_RETRIES = 10
            for retry in range(MAX_RETRIES):
                try:
                    response = self._call_llm()
                    break
                except Exception as e:
                    self._log("OpenAI API error: %s, retry %d/%d", str(e), retry, MAX_RETRIES, level="error")

            if retry == MAX_RETRIES - 1:
                return f"[Error] Failed to call LLM: {e}"

            choice = response.choices[0]
            message = choice.message

            if hasattr(message, "to_dict"):
                assistant_msg = message.to_dict()
            elif hasattr(message, "model_dump"):
                assistant_msg = message.model_dump()
            else:
                assistant_msg = {"role": "assistant", "content": message.content}
            assistant_msg.setdefault("role", "assistant")
            self.messages.append(assistant_msg)

            tool_call_dicts = assistant_msg.get("tool_calls")
            reasoning_details = assistant_msg.get("reasoning_details")

            self.trajectory.record_assistant_step(
                message.content, tool_call_dicts, reasoning_details=reasoning_details,
            )

            try:
                if message.reasoning and self.verbose:
                    summary = message.reasoning if isinstance(message.reasoning, str) else ""
                    preview = summary[:300] + ("..." if len(summary) > 300 else "")
                    print(f"\n[Reasoning] {preview}")
            except Exception:
                try:
                    summary = message.reasoning_content[:300]
                except Exception:
                    pass

            if message.content:
                if self.verbose:
                    print(f"\n[Assistant] {message.content[:500]}")

            if not message.tool_calls:
                if choice.finish_reason == "stop":
                    self._log("Model stopped without calling finish tool.")
                    return message.content or "[No response]"
                continue

            for tool_call in message.tool_calls:
                fn_name = tool_call.function.name
                try:
                    fn_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError as e:
                    self._log("Failed to parse tool arguments: %s", e, level="error")
                    err_text = f"[Error] Invalid JSON arguments: {e}"
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": err_text,
                    })
                    self.trajectory.record_tool_step(
                        tool_call_id=tool_call.id,
                        tool_name=fn_name,
                        code=None,
                        text_output=err_text,
                    )
                    continue

                if fn_name == "finish":
                    answer = fn_args.get("answer", "")
                    if self.verbose:
                        print(f"\n{'='*60}")
                        print(f"[FINISH] Final Answer:")
                        print(answer)
                        print(f"{'='*60}\n")
                    return answer

                elif fn_name == "execute_code":
                    code = fn_args.get("code", "")
                    text_output = ""
                    image_parts: List[Dict[str, Any]] = []
                    base64_images: List[str] = []
                    try:
                        exec_result = await self._handle_execute_code(code)
                        text_output = exec_result["text_output"]
                        image_parts = exec_result["image_parts"]
                        base64_images = exec_result["base64_images"]
                    except Exception as e:
                        tb = traceback.format_exc()
                        self._log("Code execution failed: %s", e, level="error")
                        text_output = f"[Execution Error] {e}\n{tb}"

                    if image_parts:
                        tool_content: Any = [
                            {"type": "text", "text": text_output},
                        ] + image_parts
                    else:
                        tool_content = text_output

                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_content,
                    })

                    self.trajectory.record_tool_step(
                        tool_call_id=tool_call.id,
                        tool_name=fn_name,
                        code=code,
                        text_output=text_output,
                        base64_images=base64_images,
                    )

                    if self.verbose:
                        print(f"\n[Code Output] {text_output[:500]}")
                        if image_parts:
                            print(f"  [{len(image_parts)} image(s) returned to model in tool message]")

                else:
                    self._log("Unknown tool: %s", fn_name, level="warning")
                    err_text = f"[Error] Unknown tool: {fn_name}"
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": err_text,
                    })
                    self.trajectory.record_tool_step(
                        tool_call_id=tool_call.id,
                        tool_name=fn_name,
                        code=None,
                        text_output=err_text,
                    )

        self._log("Max iterations reached (%d)", self.max_iterations, level="warning")
        return "[Error] Max iterations reached without a final answer."

    async def run_interactive(self, image_paths: Optional[List[str]] = None):
        """
        Run in interactive mode — the user can keep asking questions
        and both the kernel state and conversation history are preserved.

        When the message count (excluding system prompt and summary)
        reaches ``max_history``, the history is compressed into a single
        summary message before the next user turn.
        """
        print("\n" + "="*60)
        print("VLM Tool Call Agent - Interactive Mode (Docker Runtime)")
        print(f"  Memory: last {self.max_history} messages kept"
              if self.max_history > 0 else "  Memory: unlimited history")
        if self.summary_model:
            print(f"  Summary model: {self.summary_model}")
        print("Type 'quit' or 'exit' to stop.")
        print("Type 'image:<path>' to add an image to the next query.")
        print("="*60 + "\n")

        # Session-level state: messages persist across turns
        self.messages = [
            {"role": "system", "content": self.system_prompt},
        ]
        self.trajectory = self._init_trajectory("interactive_session", image_paths)

        session_images = list(image_paths or [])

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit"):
                print("Goodbye!")
                break

            if user_input.lower().startswith("image:"):
                img_path = user_input[6:].strip()
                if os.path.exists(img_path):
                    session_images.append(img_path)
                    print(f"  Added image: {img_path}")
                else:
                    print(f"  Image not found: {img_path}")
                continue

            # Summarize if history has grown too large
            await self._maybe_summarize()

            # Build and append user message (without resetting messages)
            user_msg = self._build_user_message(
                user_input, session_images if session_images else None,
            )
            self.messages.append(user_msg)
            self.trajectory.record_user_step(user_input, session_images or None)

            if self.verbose:
                print(f"\n{'='*60}")
                print(f"User Query: {user_input}")
                if session_images:
                    print(f"Images: {session_images}")
                hist_count = self._count_history_messages()
                print(f"History: {hist_count} messages"
                      + (" (has summary)" if self._extract_existing_summary() else ""))
                print(f"{'='*60}\n")

            # Run the agentic loop (appends to self.messages in-place)
            answer = await self._run_loop()
            print(f"\nAnswer: {answer}\n")

            session_images = []

        # Save trajectory on exit
        if self.trajectory:
            self.trajectory.save()
            self.trajectory.save_messages_raw(self.messages)

    async def cleanup(self):
        """Shut down the Docker kernel and clean up resources."""
        if self.kernel:
            await self.kernel.shutdown()
