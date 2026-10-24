import grpc
from concurrent import futures
import ceramic_pb2
import ceramic_pb2_grpc
from inference_service import inference_service


class CeramicServiceServicer(ceramic_pb2_grpc.CeramicServiceServicer):
    def Predict(self, request, context):
        if inference_service.model is None:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Model AI gagal dimuat di server.")
            return ceramic_pb2.PredictResponse()

        try:
            image_bytes = request.image_data
            prediction_result = inference_service.predict(image_bytes)

            return ceramic_pb2.PredictResponse(
                status="success",
                file_name=request.file_name,
                prediction=prediction_result["prediction"],
                confidence_score=prediction_result["confidence_score"]
            )
        except Exception as prediction_error:
            print(f"[error] Error saat prediksi gRPC: {str(prediction_error)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Terjadi kesalahan: {str(prediction_error)}")
            return ceramic_pb2.PredictResponse()


def serve():
    grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    ceramic_pb2_grpc.add_CeramicServiceServicer_to_server(CeramicServiceServicer(), grpc_server)

    grpc_server.add_insecure_port('[::]:50051')
    print("[info] gRPC Server berjalan di port 50051...")
    grpc_server.start()
    grpc_server.wait_for_termination()


if __name__ == '__main__':
    serve()
