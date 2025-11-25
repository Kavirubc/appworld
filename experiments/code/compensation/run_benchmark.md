# Compensation Agent Benchmark

This guide explains how to run the Venmo compensation agent benchmark using `langchain_compensation`.

## Quick Start

```bash
# 1. Create and activate virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -e .
pip install -e "experiments[simplified]"
pip install langchain langchain_google_genai langgraph langchain-community litellm

# 3. Setup AppWorld
appworld install --repo
appworld download data

# 4. Set environment variables
export GEMINI_API_KEY="your-api-key"
export GOOGLE_API_KEY="your-api-key"

# 5. Run the benchmark
python experiments/code/compensation/run_comp_agent.py --task-id 2a163ab_1 --instruction "Check my Venmo balance"
```

## Environment Setup

### Required Environment Variables

Create a `.env` file in `experiments/code/compensation/` or set these environment variables:

```bash
GEMINI_API_KEY=your-gemini-api-key
GOOGLE_API_KEY=your-gemini-api-key

# Optional tuning
LLM_TIMEOUT=120
LLM_MAX_RETRIES=5
MODEL_NAME=gemini-2.0-flash
```

### Python Version

Requires Python 3.11+ (tested with Python 3.12).

## Running the Agent

### Basic Usage

```bash
# Check Venmo balance
python experiments/code/compensation/run_comp_agent.py \
  --task-id 2a163ab_1 \
  --instruction "Check my Venmo balance"

# Send a payment (tracked in compensation log)
python experiments/code/compensation/run_comp_agent.py \
  --task-id 2a163ab_1 \
  --instruction "Send 5 dollars to ed_wilson@gmail.com for lunch"

# Send payment and request refund
python experiments/code/compensation/run_comp_agent.py \
  --task-id 2a163ab_1 \
  --instruction "Send 3 dollars to ed_wilson@gmail.com and then request a refund"
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `--task-id` | AppWorld task ID (default: first train task) |
| `--instruction` | Custom instruction (overrides task instruction) |
| `--model` | Model name (default: gemini-2.0-flash) |
| `--dry-run` | Skip LLM calls for testing |
| `--evaluate` | Run evaluation after task completion |
| `--list-tasks` | List available task IDs and exit |

### Examples

```bash
# Dry run (no LLM calls)
python experiments/code/compensation/run_comp_agent.py --task-id 2a163ab_1 --dry-run

# List available tasks
python experiments/code/compensation/run_comp_agent.py --list-tasks

# Run with evaluation
python experiments/code/compensation/run_comp_agent.py --task-id 2a163ab_1 --evaluate

# Use a different model
python experiments/code/compensation/run_comp_agent.py --task-id 2a163ab_1 --model gemini-2.5-flash
```

## Compensation Log

The agent tracks all compensatable actions in a `CompensationLog`. Example output:

```json
{
  "abc123": {
    "id": "abc123",
    "tool_name": "send_payment",
    "params": {
      "receiver_email": "ed_wilson@gmail.com",
      "amount": 5,
      "description": "lunch"
    },
    "result": {
      "message": "Sent money.",
      "transaction_id": 8234
    },
    "timestamp": 1684445161.0,
    "status": "COMPLETED",
    "compensated": false,
    "compensation_tool": "request_refund",
    "depends_on": []
  }
}
```

### Compensation Mapping

| Action | Compensation Action |
|--------|-------------------|
| `send_payment` | `request_refund` |

## Available Tools

| Tool | Description |
|------|-------------|
| `send_payment` | Send money to a user via Venmo |
| `request_refund` | Request a refund/payment from a user (compensation) |
| `check_balance` | Check current Venmo balance |
| `show_transactions` | Show recent Venmo transactions |
| `search_user` | Search for a Venmo user by name or email |
| `complete_task` | Mark the task as complete |

## Architecture

```
run_comp_agent.py
├── VenmoCompensationAgent (main class)
│   ├── Pre-authenticate to Venmo (main thread)
│   ├── Pre-cache balance (main thread)
│   ├── Initialize LLM (litellm + Gemini)
│   └── Create compensation agent (langchain_compensation)
├── Module-level tools (send_payment, request_refund, etc.)
└── Global agent reference (_ACTIVE_AGENT)
```

### Key Implementation Details

1. **Threading Workaround**: AppWorld uses UNIX signals for timeouts, which don't work in threads. Solution: `timeout_seconds=None` in AppWorld initialization.

2. **Pre-authentication**: Venmo token is obtained in the main thread before agent starts to avoid signal issues.

3. **LiteLLM**: Used instead of `langchain_google_genai` for more stable API calls.

4. **Global Agent Reference**: Tools access the agent instance via `_ACTIVE_AGENT` global variable.

## Troubleshooting

### "signal only works in main thread"

This error occurs when AppWorld's timeout mechanism (which uses signals) runs in a thread. Solution: Ensure `timeout_seconds=None` is set when creating AppWorld.

### API Timeout (504 Deadline Exceeded)

The Gemini API may occasionally timeout. Solutions:
- Retry the request
- Use a different model (e.g., `gemini-2.0-flash` instead of `gemini-2.5-flash`)
- Increase `LLM_TIMEOUT` environment variable

### "User does not exist"

The email address doesn't exist in the simulated AppWorld environment. Use valid emails from the task's friend list:
- `ed_wilson@gmail.com`
- `kri-powe@gmail.com`
- `les_ball@gmail.com`

### PermissionError during cleanup

This is a known issue with AppWorld's safety guard. The error is caught and ignored during cleanup.

## Running AppWorld Standard Agents

You can also run AppWorld's built-in agents:

```bash
# Run simplified agent with Gemini
appworld run auto \
  --agent-name simplified_react_code_agent \
  --model-name gemini-2.5-flash-no-reasoning \
  --dataset-name train \
  --task-id 82e2fac_1
```

## Files

| File | Description |
|------|-------------|
| `run_comp_agent.py` | Main compensation agent script |
| `langchain_comp_agent.py` | Original LangChain compensation agent (reference) |
| `langchain_normal_agent.py` | Normal LangChain agent without compensation |
| `demo_comp_agent.py` | Simple demo script |
| `.env` | Environment variables (API keys) |
