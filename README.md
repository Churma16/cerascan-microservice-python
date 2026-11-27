# CeraScan: Python Machine Learning Service

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

This service classifies ceramic tile defects using a fine-tuned VGG19 convolutional neural network. It serves predictions to the Express backend over gRPC on port 50051 and also provides a standalone FastAPI endpoint for local HTTP testing.

---

## Defects Detected

The model sorts tile images into four surface conditions:

* Normal (defect-free surface)
* Crack
* Scratch
* Stain

---

## Setup and Installation

### Prerequisites

* Python 3.9 or higher
* Virtual environment (recommended)

### Steps

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows PowerShell:
   .\venv\Scripts\Activate.ps1
   # On Linux / macOS:
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the trained model file is in place:
   ```
   models/ceramic_defect_vgg19.h5
   ```
   *(You can override this path by setting the `MODEL_PATH` environment variable).*

---

## Running the Service

### Option A: gRPC Server (Default for CeraScan)

Start the gRPC server to accept inference requests from the Express backend:

```bash
python grpc_server.py
```

The server listens on port `50051`.

### Option B: FastAPI HTTP Server (Testing / Standalone)

Run the FastAPI server to test predictions directly via HTTP:

```bash
uvicorn app:app --reload --port 8000
```

Swagger UI documentation becomes available at `http://localhost:8000/docs`.

---

## File Structure

* `grpc_server.py`: Main gRPC server listening for RPC classification requests.
* `inference_service.py`: Loads the Keras model and preprocesses tile images.
* `app.py`: Standalone FastAPI server for direct HTTP testing (`POST /predict`).
* `ceramic.proto`: Protobuf contract defining the inference service and messages.
* `ceramic_pb2.py` / `ceramic_pb2_grpc.py`: Generated Protobuf and gRPC stubs.
* `models/`: Directory holding the saved `.h5` model file.

---

## License

This project is licensed under the [MIT License](LICENSE).
