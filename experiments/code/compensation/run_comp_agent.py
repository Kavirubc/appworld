#!/usr/bin/env python3
"""
Compensation Agent for AppWorld Venmo Tasks

This script demonstrates a LangChain-based agent with compensation capabilities
for Venmo payment operations in AppWorld.

Usage:
    python run_comp_agent.py [--task-id TASK_ID] [--dry-run]

Environment Variables:
    GOOGLE_API_KEY or GEMINI_API_KEY: Required for Gemini API access
    MODEL_NAME: Model to use (default: gemini-2.5-flash)
    DRY_RUN: Set to 1 to skip LLM calls
"""

import os
import sys
import json
import argparse
from typing import Optional, Any
from datetime import datetime
from functools import partial

# Add parent paths for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from dotenv import load_dotenv

# Load .env from multiple locations
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '..', '.env'))

from langchain_compensation import create_comp_agent, CompensationLog
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from appworld import AppWorld, load_task_ids

# Try multiple options for LLM - litellm is more reliable
USE_LITELLM = False
ChatLiteLLM = None
ChatGoogleGenerativeAI = None

# Option 1: Try langchain-litellm (newer package)
try:
    from langchain_litellm import ChatLiteLLM
    USE_LITELLM = True
except ImportError:
    pass

# Option 2: Try langchain_community ChatLiteLLM
if not USE_LITELLM:
    try:
        from langchain_community.chat_models import ChatLiteLLM
        USE_LITELLM = True
    except ImportError:
        pass

# Option 3: Fall back to langchain_google_genai
if not USE_LITELLM:
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError:
        raise ImportError("No LLM backend available. Install: pip install langchain-litellm or langchain-google-genai")


# Global reference to the active agent instance (needed for tools)
_ACTIVE_AGENT = None


def _get_agent():
    """Get the currently active agent instance."""
    if _ACTIVE_AGENT is None:
        raise RuntimeError("No active agent. Ensure VenmoCompensationAgent context is active.")
    return _ACTIVE_AGENT


# Define tools at module level with simple signatures
@tool
def send_payment(receiver_email: str, amount: float, description: str = "") -> str:
    """
    Send money to a user via Venmo.

    Args:
        receiver_email: Email address of the person to send money to
        amount: Amount of money to send (must be > 0)
        description: Optional note for the payment

    Returns:
        JSON with transaction_id on success, or error message
    """
    agent = _get_agent()
    try:
        token = agent._get_venmo_token()
        desc = description.replace("'", "\\'") if description else ""
        code = f"""
result = apis.venmo.create_transaction(
    receiver_email='{receiver_email}',
    amount={amount},
    access_token='{token}',
    description='{desc}'
)
import json
print(json.dumps(result))
"""
        output = agent.world.execute(code)
        return output.strip() if output else '{"error": "No response"}'
    except Exception as e:
        return json.dumps({"error": str(e), "success": False})


@tool
def request_refund(receiver_email: str, amount: float, description: str = "Refund request") -> str:
    """
    Request a refund/payment from a user via Venmo (compensation action).

    This is used to compensate for a previous send_payment action.

    Args:
        receiver_email: Email of the person to request money from
        amount: Amount to request back
        description: Note explaining the refund request

    Returns:
        JSON with payment_request_id on success, or error message
    """
    agent = _get_agent()
    try:
        token = agent._get_venmo_token()
        desc = description.replace("'", "\\'").replace('"', '\\"') if description else "Refund request"
        code = f'''
import json
try:
    result = apis.venmo.create_payment_request(
        user_email="{receiver_email}",
        amount={amount},
        access_token="{token}",
        description="{desc}"
    )
    print(json.dumps(result))
except Exception as e:
    print(json.dumps({{"error": str(e), "success": False}}))
'''
        output = agent.world.execute(code)
        return output.strip() if output else '{"error": "No response"}'
    except Exception as e:
        return json.dumps({"error": str(e), "success": False})


@tool
def check_balance() -> str:
    """
    Check the current Venmo balance.

    Returns:
        Current balance amount
    """
    agent = _get_agent()
    try:
        token = agent._get_venmo_token()
        # Check if cached balance is available
        if hasattr(agent, '_cached_balance'):
            return json.dumps(agent._cached_balance)
        code = f"""
result = apis.venmo.show_venmo_balance(access_token='{token}')
import json
print(json.dumps(result))
"""
        output = agent.world.execute(code)
        return output.strip() if output else '{"error": "No response"}'
    except Exception as e:
        return json.dumps({"error": str(e), "success": False})


@tool
def show_transactions(limit: int = 10) -> str:
    """
    Show recent Venmo transactions.

    Args:
        limit: Maximum number of transactions to return

    Returns:
        JSON list of recent transactions
    """
    agent = _get_agent()
    try:
        token = agent._get_venmo_token()
        code = f"""
result = apis.venmo.show_transactions(access_token='{token}', page_limit={limit})
import json
print(json.dumps(result))
"""
        output = agent.world.execute(code)
        return output.strip()
    except Exception as e:
        return json.dumps({"error": str(e), "success": False})


@tool
def search_user(query: str) -> str:
    """
    Search for a Venmo user by name or email.

    Args:
        query: Name or email to search for

    Returns:
        JSON list of matching users
    """
    agent = _get_agent()
    try:
        token = agent._get_venmo_token()
        q = query.replace("'", "\\'")
        code = f"""
result = apis.venmo.search_users(query='{q}', access_token='{token}')
import json
print(json.dumps(result))
"""
        output = agent.world.execute(code)
        return output.strip()
    except Exception as e:
        return json.dumps({"error": str(e), "success": False})


@tool
def complete_task(answer: str = "") -> str:
    """
    Mark the task as complete. Call this when you have finished the task.

    Args:
        answer: Optional answer if the task requires one

    Returns:
        Confirmation message
    """
    agent = _get_agent()
    try:
        if answer:
            ans = answer.replace("'", "\\'")
            code = f"apis.supervisor.complete_task(answer='{ans}')"
        else:
            code = "apis.supervisor.complete_task()"
        output = agent.world.execute(code)
        return f"Task completed. {output}"
    except Exception as e:
        return f"Error completing task: {e}"


class VenmoCompensationAgent:
    """
    A compensation-aware agent for Venmo operations in AppWorld.

    This agent can:
    - Send payments via Venmo
    - Request payment refunds as compensation for failed/incorrect payments
    - Track all compensatable actions in a compensation log
    """

    def __init__(
        self,
        task_id: str,
        experiment_name: str = "compensation_agent",
        model_name: Optional[str] = None,
        google_api_key: Optional[str] = None,
        dry_run: bool = False,
    ):
        global _ACTIVE_AGENT

        self.task_id = task_id
        self.experiment_name = experiment_name
        self.dry_run = dry_run
        self.model_name = model_name or os.getenv("MODEL_NAME", "gemini-2.0-flash")
        self.google_api_key = google_api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

        if not self.google_api_key and not dry_run:
            raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY environment variable required")

        # Initialize AppWorld with timeout disabled (None) to allow tool calls from threads
        print(f"[INFO] Initializing AppWorld for task: {task_id}")
        self.world = AppWorld(task_id=task_id, experiment_name=experiment_name, timeout_seconds=None)
        self.task = self.world.task

        # Set global agent reference so tools can access this instance
        _ACTIVE_AGENT = self

        # Pre-authenticate to Venmo in main thread (avoids signal issues in tool threads)
        self._venmo_token = None
        self._pre_authenticate()

        # Initialize compensation log
        self.comp_log = CompensationLog(records={})

        # Use module-level tools
        self.tools = [send_payment, request_refund, check_balance, show_transactions, search_user, complete_task]

        # Compensation mapping: action -> compensation action
        self.compensation_mapping = {
            "send_payment": "request_refund"
        }

        # State mappers: extract state needed for compensation from action result
        self.state_mappers = {
            "send_payment": self._map_payment_state
        }

        # Initialize LLM and agent
        if not dry_run:
            self._init_llm_agent()
        else:
            self.llm = None
            self.agent = None
            print("[INFO] Dry run mode - LLM/agent not initialized")

    def _pre_authenticate(self):
        """Pre-authenticate to Venmo in the main thread to avoid signal issues."""
        try:
            print("[INFO] Pre-authenticating to Venmo...")
            self._get_venmo_token()
            # Pre-cache balance
            self._cache_balance()
        except Exception as e:
            print(f"[WARN] Pre-authentication failed: {e}")

    def _cache_balance(self):
        """Cache balance in main thread to avoid signal issues when tool runs in thread."""
        try:
            result = self.world.execute(f"""
result = apis.venmo.show_venmo_balance(access_token='{self._venmo_token}')
import json
print(json.dumps(result))
""")
            if result:
                self._cached_balance = json.loads(result.strip())
                print(f"[INFO] Cached balance: {self._cached_balance}")
        except Exception as e:
            print(f"[WARN] Could not cache balance: {e}")

    def _map_payment_state(self, result: Any, params: dict) -> dict:
        """Extract state from payment result for potential compensation."""
        # Parse result if it's a string
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except:
                pass

        state = {
            "receiver_email": params.get("receiver_email"),
            "amount": params.get("amount"),
        }

        if isinstance(result, dict):
            state["transaction_id"] = result.get("transaction_id")

        return state

    def _get_venmo_token(self) -> str:
        """Get or cache Venmo access token for the supervisor."""
        if self._venmo_token:
            return self._venmo_token

        try:
            # Get supervisor's credentials
            result = self.world.execute("""
passwords = apis.supervisor.show_account_passwords()
venmo_password = next((p['password'] for p in passwords if p['account_name'] == 'venmo'), None)
print(venmo_password)
""")
            venmo_password = result.strip() if result else None

            if not venmo_password:
                raise ValueError("Could not retrieve Venmo password")

            # Login to Venmo
            supervisor = self.task.supervisor
            email = supervisor.get('email')

            result = self.world.execute(f"""
login_result = apis.venmo.login(username='{email}', password='{venmo_password}')
print(login_result['access_token'])
""")
            self._venmo_token = result.strip() if result else None

            if not self._venmo_token:
                raise ValueError("Could not login to Venmo")

            print(f"[INFO] Logged into Venmo as {email}")
            return self._venmo_token
        except Exception as e:
            print(f"[WARN] Token retrieval error: {e}")
            raise

    def _init_llm_agent(self):
        """Initialize the LLM and compensation agent."""
        print(f"[INFO] Initializing LLM: {self.model_name} (litellm={USE_LITELLM})")

        timeout = int(os.getenv("LLM_TIMEOUT", "120"))
        max_retries = int(os.getenv("LLM_MAX_RETRIES", "5"))

        if USE_LITELLM and ChatLiteLLM is not None:
            # Use litellm for better reliability
            model_id = f"gemini/{self.model_name}" if not self.model_name.startswith("gemini/") else self.model_name
            try:
                self.llm = ChatLiteLLM(
                    model=model_id,
                    api_key=self.google_api_key,
                    timeout=timeout,
                    max_retries=max_retries,
                    temperature=0,
                )
            except Exception as e:
                print(f"[WARN] ChatLiteLLM failed: {e}, falling back to ChatGoogleGenerativeAI")
                self.llm = None
        else:
            self.llm = None

        # Fallback to ChatGoogleGenerativeAI
        if self.llm is None and ChatGoogleGenerativeAI is not None:
            # Use request_timeout instead of timeout for better compatibility
            self.llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=self.google_api_key,
                timeout=timeout,
                max_retries=max_retries,
                temperature=0,
                transport="rest",  # Use REST instead of gRPC (more reliable)
            )

        if self.llm is None:
            raise RuntimeError("Could not initialize any LLM backend")

        system_prompt = """You are a Venmo assistant that ensures payment integrity.

IMPORTANT COMPENSATION RULES:
1. When executing multiple payments as part of a single request (e.g., splitting a bill), treat them as a transaction.
2. If ANY payment in a multi-payment request fails, you MUST request refunds for ALL previously successful payments in that request.
3. To request a refund, use the request_refund tool with the same email and amount as the original payment.
4. Always report what happened: which payments succeeded, which failed, and what compensations were made.

Example: If asked to pay A, B, and C, and payment to C fails:
- Payment to A: Success
- Payment to B: Success
- Payment to C: Failed (user not found)
- Action: Request refund from A, request refund from B
- Result: All payments rolled back due to failure

Use the provided tools to complete tasks. Call complete_task() when done."""

        self.agent = create_comp_agent(
            self.llm,
            tools=self.tools,
            compensation_mapping=self.compensation_mapping,
            state_mappers=self.state_mappers,
            comp_log_ref=self.comp_log,
            system_prompt=system_prompt,
        )

    def run(self, instruction: Optional[str] = None) -> dict:
        """
        Run the agent on the given instruction.

        Args:
            instruction: Task instruction (defaults to task.instruction)

        Returns:
            Agent result dictionary
        """
        instruction = instruction or self.task.instruction

        print("\n" + "="*60)
        print("TASK INSTRUCTION")
        print("="*60)
        print(instruction)
        print("="*60 + "\n")

        if self.dry_run:
            print("[DRY RUN] Skipping LLM invocation")
            result = {
                "dry_run": True,
                "instruction": instruction,
                "task_id": self.task_id,
            }
        else:
            payload = {"messages": [HumanMessage(content=instruction)]}

            try:
                print("[INFO] Invoking compensation agent...")
                result = self.agent.invoke(payload)
                print("[INFO] Agent completed successfully")
            except Exception as e:
                print(f"[ERROR] Agent failed: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                result = {"error": str(e), "type": type(e).__name__}

        # Print compensation log with detailed analysis
        print("\n" + "="*60)
        print("COMPENSATION LOG")
        print("="*60)
        try:
            log_dict = self.comp_log.to_dict()
            print(json.dumps(log_dict, indent=2, default=str))
            if log_dict:
                print(f"\n{'─'*40}")
                print("COMPENSATION SUMMARY")
                print(f"{'─'*40}")

                successful_payments = []
                failed_payments = []
                compensated_payments = []

                for entry_id, entry in log_dict.items():
                    tool_name = entry.get('tool_name', 'unknown')
                    params = entry.get('params', {})
                    result = entry.get('result', {})
                    compensated = entry.get('compensated', False)

                    # Check if it was a successful payment
                    if tool_name == 'send_payment':
                        is_success = isinstance(result, dict) and result.get('transaction_id')
                        if is_success:
                            if compensated:
                                compensated_payments.append(params)
                            else:
                                successful_payments.append(params)
                        else:
                            failed_payments.append(params)

                print(f"Successful payments (not compensated): {len(successful_payments)}")
                for p in successful_payments:
                    print(f"  → ${p.get('amount', '?')} to {p.get('receiver_email', '?')}")

                print(f"Failed payments: {len(failed_payments)}")
                for p in failed_payments:
                    print(f"  ✗ ${p.get('amount', '?')} to {p.get('receiver_email', '?')}")

                print(f"Compensated (refund requested): {len(compensated_payments)}")
                for p in compensated_payments:
                    print(f"  ↩ ${p.get('amount', '?')} from {p.get('receiver_email', '?')}")

                print(f"{'─'*40}")
                if successful_payments and failed_payments:
                    print("⚠️  WARNING: Some payments succeeded but others failed!")
                    print("    You may want to request refunds for consistency.")
                elif not successful_payments and not failed_payments:
                    print("ℹ️  No payment transactions in log")
                elif compensated_payments:
                    print("✓ Compensation completed - payments were rolled back")
        except Exception as e:
            print(f"Error printing log: {e}")
        print("="*60 + "\n")

        return result

    def evaluate(self) -> dict:
        """Evaluate the task completion."""
        try:
            eval_result = self.world.evaluate()
            print("\n" + "="*60)
            print("EVALUATION RESULT")
            print("="*60)
            eval_result.report()
            return eval_result.to_dict()
        except Exception as e:
            print(f"[ERROR] Evaluation failed: {e}")
            return {"error": str(e)}

    def close(self):
        """Clean up resources."""
        global _ACTIVE_AGENT
        _ACTIVE_AGENT = None

        if hasattr(self, 'world') and self.world:
            try:
                self.world.close()
                print("[INFO] AppWorld closed")
            except PermissionError:
                # Ignore safety guard errors during cleanup
                print("[INFO] AppWorld cleanup completed (with warnings)")


def main():
    parser = argparse.ArgumentParser(description="Run Venmo Compensation Agent")
    parser.add_argument("--task-id", type=str, default=None,
                        help="AppWorld task ID (default: first train task)")
    parser.add_argument("--instruction", type=str, default=None,
                        help="Custom instruction (overrides task instruction)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip LLM calls for testing")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (default: gemini-2.5-flash)")
    parser.add_argument("--evaluate", action="store_true",
                        help="Run evaluation after task completion")
    parser.add_argument("--list-tasks", action="store_true",
                        help="List available task IDs and exit")

    args = parser.parse_args()

    if args.list_tasks:
        print("Available task IDs:")
        for dataset in ["train", "dev", "test_normal", "test_challenge"]:
            try:
                ids = load_task_ids(dataset)
                print(f"\n{dataset} ({len(ids)} tasks):")
                print(f"  First 5: {ids[:5]}")
            except:
                pass
        return

    # Get task ID
    task_id = args.task_id
    if not task_id:
        task_ids = load_task_ids("train")
        task_id = task_ids[0]
        print(f"[INFO] Using default task: {task_id}")

    # Create and run agent
    agent = None
    try:
        agent = VenmoCompensationAgent(
            task_id=task_id,
            model_name=args.model,
            dry_run=args.dry_run,
        )

        result = agent.run(instruction=args.instruction)

        print("\n" + "="*60)
        print("AGENT RESULT")
        print("="*60)
        if isinstance(result, dict):
            # Extract key info from result
            if "messages" in result:
                for msg in result["messages"]:
                    if hasattr(msg, "content") and msg.content:
                        print(f"[{type(msg).__name__}] {msg.content[:500]}...")
            else:
                print(json.dumps(result, indent=2, default=str)[:1000])
        else:
            print(str(result)[:1000])
        print("="*60)

        if args.evaluate:
            agent.evaluate()

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if agent:
            agent.close()


if __name__ == "__main__":
    main()
