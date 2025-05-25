from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from Virtual_Assistant import Transcriptor
load_dotenv(".env")
class Pag_Web_Local:
    def __init__(self):
        self.app=Flask(__name__) #inicializacion de flask
        self.transcriptor=Transcriptor() #tambien el Transcriptor(ya que ahi esta declarado practicamente todo)
        self.app.add_url_rule('/','home', self.home)#la ruta principal
        self.app.add_url_rule('/transcribir','transcribir',self.transcribir,methods=['POST']) #ruta para la transcripcion
    def home(self):
        return render_template('index.html') #metodo que renderiza el html
    def transcribir(self): #con esto verifica si el audio fue recibido en la solicitud
        if 'audio' not in request.files:
            return jsonify({
                "transcripcion": "",
                "respuesta": "NO recibio archivo de audio"
            }), 400#es 400 ya que por default pone un 200 (ok [siempre lo admite incluso si no recibio nada]) y el 400 es para solicitud incorrecta
        audio=request.files['audio']#con request trae el audio desde la solicitud
        resultado=self.transcriptor.T_audio(audio) #transcribe el audio y obtiene la respuesta
        return jsonify(resultado) #devuelve el resultado como JSON
    def run(self):
        self.app.run(host='0.0.0.0', port=8000, debug=True)

if __name__ == '__main__':#meras inicializaciones
    app_instance = Pag_Web_Local()
    app_instance.run()
