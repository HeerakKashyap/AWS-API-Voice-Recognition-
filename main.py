from model import preprocess_audio
from model import extract_features
from model import analyze_features


from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import librosa
import soundfile as sf
import io
import tempfile
import os


app = FastAPI(title="Cognitive Speech Analysis API")

# Enable CORS for local testing

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Modified report generator
def generate_report(features, anomalies, return_json=False):
    """Maintains your original thresholds and clinical metrics"""
    return {
        "clinical_metrics": {
            "pause_ratio": {
                "value": round(np.mean([f['pause_ratio'] for f in features]), 2),
                "threshold": 1.2,
                "unit": "seconds"
            },
            "lexical_diversity": {
                "value": round(np.mean([f['lexical_diversity'] for f in features]), 2),
                "threshold": 0.58,
                "unit": "ratio"
            },
            "pitch_variability": {
                "value": round(np.mean([f['pitch_cv'] for f in features]), 2),
                "threshold": 0.35,
                "unit": "coefficient"
            }
        },
        "risk_assessment": {
            "anomalous_samples": int(sum(anomalies)),
            "detected_clusters": len(set(anomalies))
        }
    }


@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI!"}


@app.post("/analyze/")
async def analyze_audio(file: UploadFile = File(...)):
    try:
        # Windows-compatible temp file handling
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Your exact processing pipeline
        audio, asr_result = preprocess_audio(tmp_path)
        features = extract_features(audio, asr_result)
        feature_matrix = np.array([list(features.values())])
        anomalies, clusters = analyze_features(feature_matrix)
        
        # Cleanup temp file
        os.unlink(tmp_path)

        # Generate JSON report
        report_data = generate_report([features], anomalies, return_json=True)
        
        print("Printing Report")
        print(report_data)
        
        return JSONResponse({
            "processing_chain": "preprocess → extract → analyze → report",
            "results": report_data
        })
    
    except Exception as e:
        raise HTTPException(500, f"Processing error: {str(e)}") from e


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
