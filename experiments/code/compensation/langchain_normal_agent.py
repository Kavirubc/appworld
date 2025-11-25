# Baseline LangChain ReAct Agent for AppWorld
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.tools import tool
from appworld import AppWorld

load_dotenv()


# --- Example tool for AppWorld Venmo ---
def get_world():
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

class LangChainNormalAgent:
    def __init__(self, task, model_name="gemini-2.5-flash-lite", google_api_key=None, dry_run: bool = False):
        """Simple LangChain agent wrapper that supports dry-run and configurable LLM params.

        Args:
            task: Task object
            model_name: model name to use
            google_api_key: optional API key passed to `ChatGoogleGenerativeAI`
            dry_run: if True, skip creating the real LLM/agent (for local testing)
        """
        self.task = task
        self.dry_run = bool(dry_run)
        # Avoid printing full API keys
        if google_api_key:
            masked = google_api_key[:4] + "..." + google_api_key[-4:]
        else:
            masked = None
        print(f"[DEBUG] GOOGLE_API_KEY present: {bool(google_api_key)}, masked: {masked}")

        # Set up AppWorld environment for this task
        self.world = AppWorld(task_id=task.id, experiment_name="normal_agent_demo")
        # Define tools as plain callables (avoid decorator return-type incompatibilities)
        # Skip tool wiring for the minimal normal-agent test to avoid decorator
        # conversion issues across langchain/langgraph versions. If you need
        # tools, add them here as `Tool` objects or plain callables compatible
        # with your installed langchain version.
        self.tools = []

        if not self.dry_run:
            llm_timeout = int(os.getenv("LLM_TIMEOUT", "600"))
            llm_max_retries = int(os.getenv("LLM_MAX_RETRIES", "5"))
            self.llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=google_api_key,
                timeout=llm_timeout,
                max_retries=llm_max_retries,
            )
            # Create the agent with the llm and tools
            self.agent = create_agent(self.tools, self.llm)
        else:
            self.llm = None
            self.agent = None

    def run(self, instruction=None):
        instruction = instruction or self.task.instruction
        if self.dry_run:
            print("[DEBUG] dry_run=True — skipping LLM call and returning fake response")
            return {"dry_run": True, "instruction": instruction}
        return self.agent.run(instruction)

    def close(self):
        try:
            self.world.close()
        except Exception:
            pass
