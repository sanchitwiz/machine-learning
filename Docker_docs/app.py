from flask import Flask
import os

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return "Welcome to the Dockerized Flask Application!"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)