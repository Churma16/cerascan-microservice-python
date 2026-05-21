import io
import uvicorn
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# Import blok bangunan Keras
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout

app = FastAPI(title="Ceramic Defect Scanner API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "ceramic_tiles_4class_bs16_lr0.01_epoch50_do0.1.h5" 
CLASS_LABELS = ["stain", "scratch", "normal", "crack"]  

try:
    print("Membangun ulang arsitektur Sequential (Kotak dalam kotak)...")
    
    base_model = VGG19(input_shape=(224, 224, 3), include_top=False, weights=None)
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.1), 
        Dense(4, activation='softmax')
    ])
    
    print("Menyuntikkan bobot (otak) dari file .h5...")
    model.load_weights(MODEL_PATH)
    print("Model berhasil dimuat dengan sempurna!")
    
except Exception as e:
    print(f"Gagal memuat model: {e}")
    model = None


def prepare_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


@app.post("/predict")
async def predict_ceramic(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model gagal dimuat di server.")
        
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Format file harus JPG atau PNG")

    try:
        contents = await file.read()
        processed_image = prepare_image(contents)
        
        probs = model.predict(processed_image, verbose=0)[0]
        
        pred_id = int(np.argmax(probs))
        confidence = float(probs[pred_id])
        
        return {
            "status": "success",
            "file_name": file.filename,
            "prediction": CLASS_LABELS[pred_id],
            "confidence_score": round(confidence * 100, 2)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)