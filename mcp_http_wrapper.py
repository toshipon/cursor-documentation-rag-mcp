import sys
import json
import requests

API_URL = "http://localhost:8000/query"

def main():
    for line in sys.stdin:
        try:
            req = json.loads(line)
            # MCPのdocument-searchツール呼び出しに対応
            if req.get("tool") == "document-search":
                query = req["arguments"]["query"]
                top_k = req["arguments"].get("topK", 5)
                filters = req["arguments"].get("filters", {})
                payload = {
                    "query": query,
                    "top_k": top_k,
                    "filters": filters
                }
                resp = requests.post(API_URL, json=payload)
                sys.stdout.write(json.dumps({"result": resp.json()}) + "\n")
                sys.stdout.flush()
            else:
                sys.stdout.write(json.dumps({"error": "unsupported tool"}) + "\n")
                sys.stdout.flush()
        except Exception as e:
            sys.stdout.write(json.dumps({"error": str(e)}) + "\n")
            sys.stdout.flush()

if __name__ == "__main__":
    main()