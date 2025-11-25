import time, os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
model = ChatGoogleGenerativeAI(
    model=os.getenv("MODEL_NAME", "gemini-2.5-flash-lite"),
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    timeout=int(os.getenv("LLM_TIMEOUT", "1200")),
    max_retries=int(os.getenv("LLM_MAX_RETRIES", "1")),
)

for i in range(3):
    prompt = "Hi"
    t0 = time.time()
    try:
        resp = model.invoke([{"role": "user", "content": prompt}])
        print("Succeeded in", time.time()-t0, "s; resp summary:", str(resp)[:200])
        break
    except Exception as e:
        print("Attempt", i+1, "failed:", type(e).__name__, e)
        print("Elapsed:", time.time()-t0)