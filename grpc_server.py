import io
import grpc
import numpy as np
from concurrent import futures
from PIL import Image
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout

# Import file protobuf
import ceramic_pb2
import ceramic_pb2_grpc

# Load model Keras
MODEL_PATH = "ceramic_tiles_4class_bs16_lr0.01_epoch50_do0.1.h5" 
CLASS_LABELS = ["crack", "normal", "scratch", "stain"]

try:
    print("Membangun ulang arsitektur Sequential untuk gRPC...")
    base_model = VGG19(input_shape=(224, 224, 3), include_top=False, weights=None)
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.1), 
        Dense(4, activation='softmax')
    ])
    print("Menyuntikkan bobot model Keras...")
    model.load_weights(MODEL_PATH)
    print("Model berhasil dimuat untuk gRPC Server!")
except Exception as e:
    print(f"Gagal memuat model: {e}")
    model = None

# Processing image function
def prepare_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# GRPC Service Implementation
class CeramicServiceServicer(ceramic_pb2_grpc.CeramicServiceServicer):
    
    def Predict(self, request, context):
        if model is None:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Model AI gagal dimuat di server.")
            return ceramic_pb2.PredictResponse()

        try:
            image_bytes = request.image_data
            
            processed_image = prepare_image(image_bytes)
            probs = model.predict(processed_image, verbose=0)[0]
            
            pred_id = int(np.argmax(probs))
            confidence = float(probs[pred_id])
            
            # return response
            return ceramic_pb2.PredictResponse(
                status="success",
                file_name=request.file_name,
                prediction=CLASS_LABELS[pred_id],
                confidence_score=round(confidence * 100, 2)
            )
            
        except Exception as e:
            print(f"Error saat prediksi gRPC: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Terjadi kesalahan: {str(e)}")
            return ceramic_pb2.PredictResponse()

# Start gRPC server
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    ceramic_pb2_grpc.add_CeramicServiceServicer_to_server(CeramicServiceServicer(), server)
    
    server.add_insecure_port('[::]:50051')
    print("gRPC Server berjalan di port 50051...")
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()