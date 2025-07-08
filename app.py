import os
import ast
import json
from flask import Flask, jsonify, request
from application import run_chatbot
import config
app = Flask(__name__)

@app.route('/Chat_with_AI', methods=['POST'])
# @my_profiler
def chat():
    payload = {}
    payload['Status'] = 'Success'
    # try:
    query = request.form.get('query')
    history = request.form.get('chat_history')
    hist = ast.literal_eval(history)
    chat_history = run_chatbot(message=str(query), chat_history=hist)
    # if config.DEBUG:
    #     print(chat_history)
    payload['Chat_history'] = chat_history
    return jsonify(payload)
    # except Exception as e:
    #     payload['Status'] = 'Error'
    #     payload['Chat_history'] = None
    #     return jsonify(payload)


@app.route('/', methods=['GET'])
def health_check():
    return json.dumps({'success': True}), 200, {'ContentType': 'application/json'}


if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5005, threaded=True)