import urllib.request, json

login_data = json.dumps({"username": "admin", "password": "predictiq2026"}).encode()
req = urllib.request.Request(
    "http://127.0.0.1:8000/api/auth/login",
    data=login_data,
    headers={"Content-Type": "application/json"},
    method="POST"
)
try:
    resp = urllib.request.urlopen(req)
    d = json.loads(resp.read())
    token = d.get("access_token", "")
    print("Token:", token[:30], "...")
except Exception as e:
    print("Login failed:", e)
    token = ""

for url in ["/api/dataset/stats", "/api/segments/overview", "/api/models/metrics", "/api/predict/history"]:
    try:
        r = urllib.request.Request(f"http://127.0.0.1:8000{url}", headers={"Authorization": f"Bearer {token}"})
        resp = urllib.request.urlopen(r)
        data = json.loads(resp.read())
        print(url, "-> success:", data.get("success"), "| error:", data.get("error"))
    except Exception as e:
        print(url, "-> FAILED:", e)
