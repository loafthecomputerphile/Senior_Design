"""Smoke test: run against a live server (uvicorn app.main:app). No frameworks."""
import io, json, sys, urllib.request, wave, struct

BASE = "http://127.0.0.1:8000"

def req(method, path, data=None, token=None, headers=None):
    h = dict(headers or {})
    if token: h["Authorization"] = f"Bearer {token}"
    r = urllib.request.Request(BASE + path, data=data, method=method, headers=h)
    with urllib.request.urlopen(r) as resp:
        return resp.status, resp.read()

# 1s sine wav, 44100Hz
buf = io.BytesIO()
w = wave.open(buf, "wb")
w.setnchannels(1); w.setsampwidth(2); w.setframerate(44100)
w.writeframes(b"".join(struct.pack("<h", int(30000 * ((i * 440 / 44100) % 1 < 0.5))) for i in range(44100)))
w.close()
wav_bytes = buf.getvalue()

# signup (may 409 on rerun), login
import urllib.error
try:
    req("POST", "/signup", json.dumps({"username": "smoketest", "password": "password123"}).encode(), headers={"Content-Type": "application/json"})
except urllib.error.HTTPError as e:
    assert e.code == 409, e
_, body = req("POST", "/login", json.dumps({"username": "smoketest", "password": "password123"}).encode(), headers={"Content-Type": "application/json"})
token = json.loads(body)["token"]
print("login OK")

# wrong password rejected
try:
    req("POST", "/login", json.dumps({"username": "smoketest", "password": "wrongpass"}).encode(), headers={"Content-Type": "application/json"})
    sys.exit("bad login accepted!")
except urllib.error.HTTPError as e:
    assert e.code == 401
print("bad password rejected OK")

# evaluate
boundary = "----smoke"
body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"test.wav\"\r\nContent-Type: audio/wav\r\n\r\n").encode() + wav_bytes + f"\r\n--{boundary}--\r\n".encode()
_, body = req("POST", "/evaluate", body, token=token, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
eval_res = json.loads(body)
print("evaluate OK:", eval_res)

# resampled to 16k? decode returned wav equivalent — check duration metadata instead via list
_, body = req("GET", "/evaluations", token=token)
evals = json.loads(body)
assert len(evals) >= 1 and evals[-1]["id"] == eval_res["id"]
print("list OK")

_, body = req("GET", "/export", token=token)
assert json.loads(body)["user"] == "smoketest"
print("export OK")

# no auth rejected
try:
    req("GET", "/evaluations")
    sys.exit("unauthenticated request accepted!")
except urllib.error.HTTPError as e:
    assert e.code in (401, 403)
print("auth enforced OK")
print("ALL SMOKE TESTS PASSED")
