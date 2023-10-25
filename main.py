import click
from flask import Flask, request, jsonify
from flask_restful import Api, Resource
import logging
from langchain_tasks import initialize_qa, CustomException, run_qa_service

app = Flask(__name)
api = Api(app)

class QAService(Resource):
    def __init__(self):
        self.qa = None

    def initialize(self, data_ingestion_config):
        self.qa = initialize_qa(data_ingestion_config)

    def post(self):
        try:
            data = request.get_json()
            if not data or "query" not in data:
                return {"error": "Invalid request data"}, 400

            query = data["query"]
            result = self.qa(query)
            answer, docs = result["result"], result["source_documents"]

            response = {
                "question": query,
                "answer": answer,
                "source_documents": [doc.page_content for doc in docs]
            }

            return jsonify(response)

        except CustomException as ce:
            return {"error": ce.message}, 500
        except Exception as e:
            return {"error": "An unexpected error occurred"}, 500

if __name__ == '__main__':
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    
    data_ingestion_config = {
        "device_type": "cuda",
        "use_history": False,
        "model_type": "llama"
    }

    api.add_resource(QAService, '/qa')
    qa_service = QAService()
    qa_service.initialize(data_ingestion_config)
    app.run(host='0.0.0.0', port=5000)