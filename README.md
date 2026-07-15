# CeraScan ML Microservice

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://python.org) [![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://tensorflow.org) [![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

A dedicated Python machine learning microservice for the CeraScan ecosystem. It performs real-time image classification on ceramic tiles, categorizing them into normal surfaces or specific defect types utilizing a customized VGG19 Convolutional Neural Network.

---

## Key Features

*   **High-Accuracy Inference**: Leverages a robust, fine-tuned VGG19 CNN (`ceramic_tiles_4class_bs16_lr0.01_epoch50_do0.1.h5`) for 4-class categorization (Crack, Scratch, Stain, Normal).
*   **gRPC Communication**: Provides lightning-fast, highly efficient inference endpoints for the main Express API Gateway.
*   **Container-Ready**: Designed to operate seamlessly as a background worker within Dockerized environments.

---

## Tech Stack & Libraries

*   **Language**: Python
*   **Machine Learning**: TensorFlow / Keras
*   **RPC Framework**: gRPC
*   **Containerization**: Docker

---

## Installation & Setup

### Steps

#### Prerequisites
*   Python >= 3.9

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run the gRPC Server**:
    ```bash
    python grpc_server.py
    ```

*(Note: The required `.h5` model file must be present in the root of this directory for the server to successfully load and serve predictions.)*

---

## Folder Structure

*   `app.py` - Standard script (if applicable) for testing or alternate serving.
*   `grpc_server.py` - Main gRPC server initializing the model and listening for client requests.
*   `ceramic_pb2*.py` - Auto-generated Protobuf files defining the gRPC interface.

---

## Acknowledgements

Special thanks and appreciation to the researchers and institutions for the foundational work in industrial computer vision.

---

## Author

**Churma16**

---

## License

This project is licensed under the [MIT License](LICENSE).
