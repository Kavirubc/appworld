
import os
import json
import time
from datetime import datetime
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_compensation import create_comp_agent, CompensationLog
from langchain_core.messages import HumanMessage
from langchain.tools import tool, ToolRuntime
from appworld import AppWorld

load_dotenv()


# --- Example tool Definitions for AppWorld Venmo ---
def get_world():
    # Helper to get the current AppWorld instance from the agent
    # This is a hack; in production, use ToolRuntime/context injection
    import inspect
    frame = inspect.currentframe()
    while frame:
        if "self" in frame.f_locals and hasattr(frame.f_locals["self"], "world"):
            return frame.f_locals["self"].world
        frame = frame.f_back
    raise RuntimeError("AppWorld instance not found in call stack.")

@tool("send_payment", description="Send money to a recipient using Venmo. Args: amount, recipient. Returns payment_id.")
def send_payment(amount: float, recipient: str) -> str:
    """Send money to a recipient using Venmo API."""
    world = get_world()
    return world.execute(f"apis.venmo.send_payment(amount={amount}, recipient='{recipient}')")

@tool("request_refund", description="Request a refund for a payment. Args: payment_id.")
def request_refund(payment_id: str) -> str:
    """Request a refund for a payment using Venmo API."""
    world = get_world()
    return world.execute(f"apis.venmo.request_refund(payment_id={payment_id})")


class LangChainCompAgent:
    def __init__(self, task, model_name="gemini-2.5-pro", google_api_key=None, dry_run: bool = False):
        """Create the comp agent.

        Args:
            task: Task object
            model_name: LLM model name
            google_api_key: optional API key for Google models
            dry_run: if True, skip LLM calls and return a fake response (useful for local testing)
        """
        self.task = task
        self.dry_run = bool(dry_run)
        # Avoid printing full API keys to logs; show masked presence only
        if google_api_key:
            masked = google_api_key[:4] + "..." + google_api_key[-4:]
        else:
            masked = None
        print(f"[DEBUG] GOOGLE_API_KEY present: {bool(google_api_key)}, masked: {masked}")

        # Set up AppWorld environment for this task
        self.world = AppWorld(task_id=task.id, experiment_name="compensation_demo")

        # Define tools (decorated functions)
        self.tools = [send_payment, request_refund]

        # Compensation mapping
        self.compensation_mapping = {
            "send_payment": "request_refund"
        }

        # State mappers (minimal example)
        self.state_mappers = {
            "send_payment": lambda r, p: {"payment_id": r}  # r = result (payment_id)
        }

        self.comp_log = CompensationLog(records={})

        if not self.dry_run:
            # Only create the real LLM and agent when not in dry-run mode
            # Allow overriding timeout and internal retries with env vars for tuning
            llm_timeout = int(os.getenv("LLM_TIMEOUT", "600"))
            llm_max_retries = int(os.getenv("LLM_MAX_RETRIES", "5"))
            self.llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=google_api_key,
                timeout=llm_timeout,
                max_retries=llm_max_retries,
            )
            # streaming flag (useful for long responses)
            self.llm_stream = bool(int(os.getenv("LLM_STREAM", "0")))
            self.agent = create_comp_agent(
                self.llm,
                tools=self.tools,
                compensation_mapping=self.compensation_mapping,
                state_mappers=self.state_mappers,
                comp_log_ref=self.comp_log,
                system_prompt="You are a financial assistant. Use Venmo to send payments and handle refunds if needed."
            )
        else:
            # Placeholders for dry-run
            self.llm = None
            self.agent = None


    def run(self, instruction=None):
        instruction = instruction or self.task.instruction
        payload = {"messages": [HumanMessage(content=instruction)]}
        # Logging: print instruction and payload size
        print("\n[DEBUG] Instruction to Gemini:")
        print(instruction)
        payload_json = json.dumps({"messages": [{"content": instruction}]})
        print(f"[DEBUG] Payload size (bytes): {len(payload_json.encode('utf-8'))}")
        if self.dry_run:
            print("[DEBUG] dry_run=True — skipping LLM call and returning fake response")
            fake_result = {"dry_run": True, "instruction": instruction}
            # still print compensation log for consistency
            print("\n=== COMPENSATION LOG (dry_run) ===")
            try:
                print(json.dumps(self.comp_log.to_dict(), indent=2))
            except Exception:
                print(repr(self.comp_log))
            return fake_result

        # Controlled invoke with configurable retry/backoff independent of underlying LLM retries
        invoke_max_attempts = int(os.getenv("LLM_INVOKE_MAX_ATTEMPTS", "3"))
        backoff_base = float(os.getenv("LLM_INVOKE_BACKOFF_BASE", "2.0"))
        backoff_max = float(os.getenv("LLM_INVOKE_BACKOFF_MAX", "30.0"))

        last_exc = None
        for attempt in range(1, invoke_max_attempts + 1):
            ts = datetime.utcnow().isoformat() + "Z"
            print(f"[DEBUG] Invoke attempt {attempt}/{invoke_max_attempts} at {ts}")
            # If streaming is enabled, try streaming first (safer for long outputs)
            if getattr(self, "llm_stream", False):
                print("[DEBUG] LLM_STREAM enabled — attempting streaming mode")
                stream_iter = None
                try:
                    if self.agent is not None and hasattr(self.agent, "stream"):
                        stream_iter = self.agent.stream(payload)
                    elif self.llm is not None and hasattr(self.llm, "stream"):
                        stream_iter = self.llm.stream(payload)
                    else:
                        print("[DEBUG] Streaming not supported by agent/llm; falling back to invoke")
                        stream_iter = None

                    if stream_iter is not None:
                        collected = []
                        try:
                            for chunk in stream_iter:
                                text = None
                                if isinstance(chunk, str):
                                    text = chunk
                                else:
                                    text = getattr(chunk, "content", None) or getattr(chunk, "text", None)
                                    if text is None and isinstance(chunk, dict):
                                        text = chunk.get("content") or chunk.get("text")
                                if text:
                                    print(text, end="", flush=True)
                                    collected.append(text)
                        except Exception as se:
                            print("[WARN] Streaming failed during iteration, will fallback to invoke:", se)
                            try:
                                import traceback
                                traceback.print_exc()
                            except Exception:
                                pass
                            stream_iter = None
                        else:
                            # successful streaming
                            result = {"streamed": True, "content": "".join(collected)}
                            print("\n[DEBUG] Streaming completed successfully")
                            break
                except Exception as e:
                    print("[WARN] Streaming attempt raised; falling back to invoke:", e)
                    try:
                        import traceback
                        traceback.print_exc()
                    except Exception:
                        pass
                    stream_iter = None

            try:
                result = self.agent.invoke(payload)
                print(f"[DEBUG] Invoke succeeded on attempt {attempt}")
                break
            except KeyboardInterrupt:
                raise
            except Exception as e:
                last_exc = e
                print(f"[WARN] Invoke attempt {attempt} failed: {type(e).__name__}: {e}")
                try:
                    import traceback
                    traceback.print_exc()
                except Exception:
                    print(repr(e))
                if attempt == invoke_max_attempts:
                    print("[ERROR] Exhausted invoke retries")
                    error_info = {
                        "error": str(e),
                        "type": type(e).__name__,
                        "attempts": attempt,
                    }
                    print("\n=== COMPENSATION LOG (on error) ===")
                    try:
                        print(json.dumps(self.comp_log.to_dict(), indent=2))
                    except Exception:
                        print(repr(self.comp_log))
                    return error_info
                # compute exponential backoff
                backoff = min(backoff_base * (2 ** (attempt - 1)), backoff_max)
                print(f"[DEBUG] Sleeping for {backoff} seconds before next attempt")
                time.sleep(backoff)
        # Print compensation log for inspection
        print("\n=== COMPENSATION LOG ===")
        try:
            print(json.dumps(self.comp_log.to_dict(), indent=2))
        except Exception:
            print(repr(self.comp_log))
        return result

    def close(self):
        self.world.close()
