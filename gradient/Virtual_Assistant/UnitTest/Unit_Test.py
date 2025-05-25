import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
from unittest.mock import patch, MagicMock as MM
#mock es para simular funciones y demas, con el fin de evitar llamar a los apis reales
#en este caso una buena opcion ya que los apis son de paga
from Virtual_Assistant import GPT, Transcriptor
from P_Web import Pag_Web_Local
import openai as OpenAI
import json #Pues JSON es para convertir las respuestas a diccionarios
#test para flask

class T_FlaskApp(unittest.TestCase):      
    def test_page(self):
        self.appinst=Pag_Web_Local() #se trae la clase de la pagina web
        self.client=self.appinst.app.test_client()#es un test que permite interactuar con lo de la pag
        response=self.client.get('/') #llama a la ruta principal
        self.assertEqual(response.status_code,200)  #verifica que carga (o sea que todo este bien)
    def test_transcribir(self): #test para transcribir
        self.appinst=Pag_Web_Local() #nuevamente se trae esto
        self.client=self.appinst.app.test_client()#lo mismo con esto
        response=self.client.post('/transcribir') #hace un post a transcribir
        self.assertEqual(response.status_code,400) #y espera que sea una solicitud incorrecta(Bad Request)
        jsonD=json.loads(response.data) #convierte la respuesta (string) en un JSON
        self.assertEqual(jsonD['respuesta'], "NO recibio archivo de audio")
        self.assertEqual(jsonD['transcripcion'], "")#la transcripcion sin nada

#Test que es en referencia a la clase GPT
class T_GPT(unittest.TestCase):
    @patch('Virtual_Assistant.OpenAI')#este es la simulacion de de OPENAI(la clase)
    def test_pedir_respuesta(self, mock):#mock es el objeto simulando OPENAI(mas como un remplazo)
        openAIC=MM()#objeto "falso" del api
        mock.return_value=openAIC #se le asigna el objeto falso al MOCK
        openAIC.chat.completions.create.return_value.choices=[MM(message=MM(content='Hola'))]
        #se hace pasar por el api de openai, simulando que te pone un simple HOLA (el asistente)
        gpt=GPT() #se crea la calse GPT
        respuesta=gpt.pedir_respuesta("Hola como estas?")#trae ese metodo texto incluido
        self.assertEqual(respuesta, 'Hola')#esto es para comparar (el nombre lo dice ) que la respuesta sea igual a Hola
#test para la transcripcion.
class T_Transcriptor(unittest.TestCase):
    @patch('Virtual_Assistant.OpenAI')
    @patch.object(GPT, 'pedir_respuesta', return_value="Probando Probando") #simula una respuesta(con pedir_respuesta) para que traiga un texto
    def test_transcriptor_audio(self, mock_gpt_response, mock_openai):
        audioC=MM() #crea el magicmock
        audioC.save=MM()#simulando un audio
        TransC=MM() #crea otro MM y se guarda en TransC, que hace referencia a la transcripcion
        TransC.text="Texto del audio"#crea una "transcripcion"
        mock_openai.return_value.audio.transcriptions.create.return_value=TransC
    #de la misma manera que arriba, se le asigna el contenido falso (recordando la nota de Virtual Assistant)
        transcriptor=Transcriptor() #manda a traer transcriptor 
        resultado=transcriptor.T_audio(audioC)#llama a T_audio para convertir el audio en texto
        self.assertEqual(resultado['transcripcion'], 'Texto del audio')
        self.assertEqual(resultado['respuesta'], 'Probando Probando')
    #que sea tanto la trasncripcion como la respuesta igual al texto/audio mandado, si si, todo OK

if __name__ == '__main__':
    unittest.main() #-> para ejecutar todos los Test que se hicieron
