import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from inference_service import inference_service

app = FastAPI(title="Ceramic Defect Scanner API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict")
async def predict_ceramic(file: UploadFile = File(...)):
    if inference_service.model is None:
        raise HTTPException(status_code=500, detail="Model gagal dimuat di server.")

    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Format file harus JPG atau PNG")

    try:
        file_contents = await file.read()
        prediction_result = inference_service.predict(file_contents)

        return {
            "status": "success",
            "file_name": file.filename,
            "prediction": prediction_result["prediction"],
            "confidence_score": prediction_result["confidence_score"]
        }
    except Exception as processing_error:
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan: {str(processing_error)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
