from appworld.task import Task, load_task_ids
import os
from experiments.code.compensation.langchain_comp_agent import LangChainCompAgent
from experiments.code.compensation.langchain_normal_agent import LangChainNormalAgent

# Choose a task_id from your dataset (replace with a real one)
task_ids = load_task_ids("test_challenge")  # or "dev", "train", etc.
task_id = task_ids[0]  # Just pick the first for demo
print(f"Running compensation agent on task: {task_id}")


# Optionally pass API key and model name from environment
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.5-flash-lite")
# Read DRY_RUN env var (0/1) so callers can run without editing this file
DRY_RUN = bool(int(os.getenv("DRY_RUN", "0")))


task = Task.load(task_id=task_id)
agent = LangChainNormalAgent(task, model_name=MODEL_NAME, google_api_key=GOOGLE_API_KEY, dry_run=DRY_RUN)
result = agent.run()  # Uses the task's instruction by default
print("\nAgent result:", result)
agent.close()
