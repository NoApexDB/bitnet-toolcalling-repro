import time
import requests
from concurrent.futures import ThreadPoolExecutor

def test_worker(worker_id):
    t0 = time.perf_counter()
    session = requests.Session()
    session.trust_env = False
    payload = {
        "prompt": f"Human: Count from 1 to 10 for worker {worker_id}\n\nBITNETAssistant:",
        "n_predict": 32,
        "temperature": 0.0
    }
    resp = session.post("http://127.0.0.1:8088/completion", json=payload, timeout=30.0)
    dur = time.perf_counter() - t0
    data = resp.json()
    content = data.get("content", "").strip()
    timings = data.get("timings", {})
    t_gen = timings.get("predicted_per_second", 0)
    tokens = timings.get("predicted_n", 0)
    print(f"Worker {worker_id}: {dur:.2f}s | {t_gen:.2f} t/s | tokens: {tokens} | resp: {content[:30]}...")
    return {
        "worker_id": worker_id,
        "duration": dur,
        "t_gen": t_gen,
        "tokens": tokens
    }

def main():
    print("Testing 4 concurrent requests via ThreadPoolExecutor...")
    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(test_worker, range(4)))
    total_dur = time.perf_counter() - t_start
    total_tokens = sum(r["tokens"] for r in results)
    agg_throughput = total_tokens / total_dur
    print(f"\nAll 4 workers completed in {total_dur:.2f}s!")
    print(f"Total tokens generated: {total_tokens}")
    print(f"Aggregate swarm throughput: {agg_throughput:.2f} tokens/s")

if __name__ == "__main__":
    main()
