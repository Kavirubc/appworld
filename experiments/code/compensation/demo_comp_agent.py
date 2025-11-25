import os
import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_compensation import create_comp_agent, CompensationLog
from langchain_core.messages import HumanMessage
from experiments.code.compensation.langchain_comp_agent import send_payment, request_refund

load_dotenv()


def build_agent(dry_run: bool = False):
    """Build a compensation-aware agent. If dry_run=True, do not create the LLM/agent.

    Returns (agent, comp_log)
    """
    comp_log = CompensationLog(records={})

    ALL_TOOLS = [send_payment, request_refund]

    compensation_mapping = {"send_payment": "request_refund"}
    state_mappers = {"send_payment": lambda r, p: {"payment_id": r}}

    if dry_run:
        return None, comp_log

    model_name = os.getenv("MODEL_NAME", "gemini-2.5-flash-lite")
    google_api_key = os.getenv("GOOGLE_API_KEY")
    llm_timeout = int(os.getenv("LLM_TIMEOUT", "600"))
    llm_max_retries = int(os.getenv("LLM_MAX_RETRIES", "1"))

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=google_api_key,
        timeout=llm_timeout,
        max_retries=llm_max_retries,
    )

    agent = create_comp_agent(
        llm,
        tools=ALL_TOOLS,
        compensation_mapping=compensation_mapping,
        state_mappers=state_mappers,
        comp_log_ref=comp_log,
        system_prompt="You are an assistant that may call tools and should record compensation actions.",
    )

    return agent, comp_log


def run_demo(instruction: str = None):
    DRY_RUN = bool(int(os.getenv("DRY_RUN", "0")))
    agent, comp_log = build_agent(dry_run=DRY_RUN)

    instruction = instruction or "Perform payment flow for demo"
    payload = {"messages": [HumanMessage(content=instruction)]}

    if DRY_RUN:
        print("[DEBUG] DRY_RUN=1 — skipping LLM/agent creation. Would invoke with payload:")
        print(json.dumps(payload, indent=2))
        print("\n=== COMPENSATION LOG (dry_run) ===")
        try:
            print(json.dumps(comp_log.to_dict(), indent=2))
        except Exception:
            print(repr(comp_log))
        return {"dry_run": True, "instruction": instruction}

    try:
        print("[DEBUG] Invoking compensation agent...")
        result = agent.invoke(payload)
    except Exception as e:
        print("[ERROR] Agent invocation failed:", type(e).__name__, e)
        try:
            import traceback

            traceback.print_exc()
        except Exception:
            pass
        result = {"error": str(e), "type": type(e).__name__}
    finally:
        print("\n=== COMPENSATION LOG ===")
        try:
            print(json.dumps(comp_log.to_dict(), indent=2))
        except Exception:
            try:
                print(comp_log)
            except Exception:
                pass

    return result


if __name__ == "__main__":
    # Example usage: set DRY_RUN=1 for local testing without LLM calls
    res = run_demo()
    print("\nAgent result:", res)
