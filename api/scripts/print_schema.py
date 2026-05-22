import json, sys
d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "qwen_schema.json"))
s = d["latest_version"]["openapi_schema"]["components"]["schemas"]["Input"]
print(json.dumps(s, indent=2))
