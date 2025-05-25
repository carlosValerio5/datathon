class AudioRecorder{
    constructor(micSelector, boxSelector, outputSelector) {
        this.estado =0; //0 es "no grabando"
        this.blobs =[]; //array para almacenar los datos de audio
        this.stream =null; //stream de audio
        this.rec =null; //rec de media recorder
        this.mic =document.querySelector(micSelector); //selector del micro
        this.box =document.querySelector(boxSelector);//selector del cuadro
        this.output =document.getElementById(outputSelector); //selector del output en la cajita
        this.initEvents();
    }

    initEvents(){
        document.addEventListener('DOMContentLoaded',() => { //espera a que el DOM esté cargado (DOM es la estruct de la pagina)
            this.mic.addEventListener('click', () =>this.processMic()); //cuando se hace click en el micro, llama a la función processMic
        });
    }
    async sRec(){ //inicia la grabacion
        this.blobs=[];//reinicia el array de blobs (siendo blobs los datos de audio)
        this.stream =await navigator.mediaDevices.getUserMedia({audio: true,video: false}); //pide permiso para acceder al micro y le quite el derecho a vudeo
        this.rec=new MediaRecorder(this.stream); //crea un nuevo objeto MediaRec con el stream de audio (mediarecorder es la API de grabación de audio)
        this.rec.ondataavailable =e => this.blobs.push(e.data); //cuando se pueda grabar, añade los datos al array de blobs
        this.rec.onstop =async () => { //esto es para cuando se detiene la grabacion
            const blob =new Blob(this.blobs, {type: 'audio/webm; codecs=opus'}); //crea un nuevo blob con los datos de audio y el tipo de codec
            const formData =new FormData(); //crea un nuevo objeto FormData para enviar los datos al servidor (formdata es un objeto que permite construir un conjunto de pares clave/valor para enviar al servidor)
            formData.append('audio',blob,'audio.webm'); //append es para añadir un nuevo par clave/valor al objeto formdata 
            this.output.placeholder ="TRANSCRIBIENDO...";
            try{
                const response =await fetch('/transcribir',{ //envia los datos al servidor
                    method: 'POST', 
                    body: formData //el body es el objeto formdata que contiene los datos de audio
                });
                const data =await response.json();//espera la respuesta del servidor y la convierte a json (es json porque el servidor devuelve un objeto json con la transcripcion y la respuesta)
                this.output.value=`TU: ${data.transcripcion}\nASISTENTE: ${data.respuesta}`; //muestra la transcripcion y la respuesta en el output
                if (data.audio_path) {
                    const audio=new Audio(data.audio_path); //carga el audio y lo reproduce
                    audio.play(); //esto le da play
                }
            } catch (err){ //meros posibles errores
                console.error('Error transcribiendo el audio:', err);
                this.output.value = "Error al transcribir el audio";
            }
            //this.mic.classList.remove('active', 'moved'); 
            this.estado=0; //reinicia el estado a 0 (no grabando)
        };
        this.rec.start();
    }
    stopRec(){
        if (this.rec && this.rec.state ==='recording') { //stop recording es para detener la grabación(cuando el rec y el estado del rec son iguales a grabando)
            this.rec.stop();
            this.stream.getTracks().forEach(track => track.stop());
        }
    }
    processMic() {
        if (this.estado === 0) {
            this.mic.classList.add('active');
            this.sRec();
            this.estado = 1;//cambia el estado a 1 (grabando)
        } else if (this.estado === 1) {
            this.mic.classList.add('moved'); //añade la clase moved al micro (moved es para indicar que se ha movido el micro)
            this.output.placeholder ="PROCESANDO...";//simplemente se muestra eso mientras se procesa la grabacion
            this.box.style.display ='block';//muestra el cuadro de texto (box es el cuadro donde se muestra la transcripcion y la respuesta)
            this.output.value="";//reinicia el output (es el cuadro de texto donde se muestra la transcripcion y la respuesta)
            this.stopRec(); //detiene la grabacion
            this.estado =2; //cambia el estado a 2 (grabando y procesando)
        }
    }
}
new AudioRecorder('.microphone', '.box', 'output'); 
