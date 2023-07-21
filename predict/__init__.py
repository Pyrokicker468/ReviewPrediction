import logging
import random
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
import azure.functions as func
import importlib
import google.protobuf
import tensorflow as tf

logging.info(google.protobuf.__version__)
logging.info(google.protobuf.__file__)

importlib.reload(google.protobuf)

logging.info(google.protobuf.__version__)
logging.info(google.protobuf.__file__)

print('Code got to this point 1')

model_dir = "/home/site/wwwroot/predict/DistilBERT_Model"
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
model = TFDistilBertForSequenceClassification.from_pretrained(model_dir)

def main(req: func.HttpRequest) -> func.HttpResponse:
    try:
        logging.info('Python HTTP trigger function processed a request.')
        text = req.params.get('name')
    
        if not text:
            try:
                req_body = req.get_json()
            except ValueError:
                pass
            else:
                text = req_body.get('name')

        if text:
            # Tokenize the input text
            input_ids = tokenizer.encode(text, truncation=True, padding=True, return_tensors="tf")
            
            # Perform prediction
            outputs = model(input_ids)[0]
            predicted_label = tf.argmax(outputs, axis=1).numpy()[0]

            if predicted_label == 0:
                return func.HttpResponse(f"The text '{text}' is predicted as human-generated.")
            else:
                return func.HttpResponse(f"The text '{text}' is predicted as computer-generated.")
        else:
            return func.HttpResponse(
                "Please provide a 'text' parameter in the query string or in the request body.",
                status_code=400
            )
    
    except Exception as e:
        logging.exception("An error occurred:")
        return func.HttpResponse("Internal Server Error", status_code=500)
