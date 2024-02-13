from flask import Flask, request
from main import raccomanda_film, scrivi_su_file

app = Flask(__name__)

@app.route('/', methods = ['POST'])
def hello_world():
    raccomandazioni_utente, numero_cluster = raccomanda_film(request.json)
    return raccomandazioni_utente

@app.route('/feedback', methods = ['POST'])
def feedback():
    return scrivi_su_file(request.json)

if __name__ == '__main__':
    app.run(debug=True)
