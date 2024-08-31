from flask import Flask, request, jsonify
import pandas as pd
from multicultureCCT import get_consensus
import logging

app = Flask(__name__)

@app.route('/assign_cultures', methods=['POST'])
def assign_cultures():
    try:
        if request.method == 'POST':
            data = request.get_json()
            logging.info(f"Received data: {data}")
            
            df = pd.DataFrame(data['data'])
            result = get_consensus(df)
            
            logging.info(f"Result: {result}")
            
            response = jsonify(result)
            response.headers['Content-Type'] = 'application/json'
            return response
        else:
            return 'Method not allowed', 405
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        return f"Internal Server Error: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)