from fastapi import FastAPI
from pydantic import BaseModel
from model import predict, update_lut, get_lut
import time

app = FastAPI(title="TinyNet Edge Inference API")

# Track stats across requests
stats = {"total_requests": 0, "skipped": 0, "computed": 0}

class InputData(BaseModel):
    a: float
    b: float

class LUTUpdate(BaseModel):
    key: int    # 0 to 255
    value: int  # 0 = compute, 1 = skip

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": round(time.time(), 2)}

@app.post("/predict")
def run_predict(data: InputData):
    result = predict(data.a, data.b)
    stats["total_requests"] += 1
    if result["skip"] == 1:
        stats["skipped"] += 1
    else:
        stats["computed"] += 1
    return result

@app.get("/stats")
def get_stats():
    total = stats["total_requests"]
    skip_rate = round((stats["skipped"] / total * 100), 1) if total > 0 else 0
    return {**stats, "skip_rate_pct": skip_rate}

@app.post("/update-lut")
def update(data: LUTUpdate):
    return update_lut(data.key, data.value)

@app.get("/lut")
def lut():
    return get_lut()