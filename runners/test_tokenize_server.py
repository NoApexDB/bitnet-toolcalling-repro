import subprocess
import time
import requests
import json
from transformers import AutoTokenizer

LLAMA_SERVER = r"C:\ai_models\BitNet\build\bin\llama-server.exe"
MODEL_GGUF = r"C:\ai_models\BitNet\models\BitNet-b1.58-2B-4T\ggml-model-i2_s.gguf"
MODEL_HF = r"C:\ai_models\BitNet-b1.58-2B-4T"
PORT = 8089

# Start server
proc = subprocess.Popen([
    LLAMA_SERVER,
    "-m", MODEL_GGUF,
    "--host", "127.0.0.1",
    "--port", str(PORT),
    "-t", "4",
    "-c", "2048"
], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

try:
    # Wait for ready
    ready = False
    for _ in range(40):
        try:
            r = requests.get(f"http://127.0.0.1:{PORT}/health", timeout=1)
            if r.status_code == 200:
                ready = True
                break
        except Exception:
            pass
        time.sleep(0.5)

    if not ready:
        print("Failed to start server!")
        exit(1)

    tok_hf = AutoTokenizer.from_pretrained(MODEL_HF)

    # Test prompt
    test_text = "System: Hello world<|eot_id|>"
    hf_tokens = tok_hf.encode(test_text, add_special_tokens=False)
    
    # Query llama-server tokenize
    resp = requests.post(f"http://127.0.0.1:{PORT}/tokenize", json={"content": test_text})
    print("Tokenize HTTP status:", resp.status_code)
    print("Tokenize response:", resp.text)
    server_tokens = resp.json().get("tokens", [])
    print("HF tokens:    ", hf_tokens)
    print("Server tokens:", server_tokens)
    print("Tokens match: ", hf_tokens == server_tokens)

finally:
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except Exception:
        proc.kill()
