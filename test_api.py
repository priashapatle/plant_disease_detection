# test_api.py (put in project root)
import requests, sys, os

url = "http://127.0.0.1:8000/predict"
file_path = "assets/sample_leaf.png"

print("FILE EXISTS:", os.path.exists(file_path))
if not os.path.exists(file_path):
    print("File not found:", file_path)
    sys.exit(1)

try:
    with open(file_path, "rb") as f:
        files = {"file": ("sample_leaf.png", f, "image/png")}
        print("Sending request to", url)
        r = requests.post(url, files=files, timeout=30)
        print("STATUS:", r.status_code)
        try:
            print("RESULT:", r.json())
        except Exception:
            print("Response text:", r.text[:1000])
except Exception as e:
    print("ERROR:", repr(e))
    sys.exit(1)
