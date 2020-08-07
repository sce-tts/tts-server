import requests
from flask import Flask, request, send_file, render_template, jsonify, Response
from io import BytesIO
import scipy.io.wavfile as swavfile

from synthesys import SAMPLING_RATE
from synthesys import generate_audio_glow_tts
# from synthesys import generate_audio_fastspeech2
from text_processer import normalize_text, process_text

app = Flask(__name__)

@app.route('/')
@app.route('/test')
def test():
    return render_template('test.html')

@app.route('/cc')
def cc():
    return render_template('cc.html')

@app.route('/text', methods=['POST'])
def text():
    text = request.json.get('text', '')
    texts = process_text(text)

    return jsonify(texts)

@app.route('/<path:path>')
def twip_proxy(path):
    new_url = request.url.replace(request.host, 'twip.kr')
    resp = requests.request(
        method=request.method,
        url=new_url,
        headers={key: value for (key, value) in request.headers if key != 'Host'},
        data=request.get_data(),
        cookies=request.cookies,
        allow_redirects=False
    )
    excluded_headers = ['content-encoding', 'content-length', 'transfer-encoding', 'connection']
    headers = [(name, value) for (name, value) in resp.raw.headers.items()
               if name.lower() not in excluded_headers]
    content = resp.content
    if new_url.startswith('http://twip.kr/assets/js/alertbox/lib-'):
        content = resp.text + f"""
        const original_function = Howl.prototype.init;
        Howl.prototype.init = function (o) {{
            if (o.src.startsWith('https://www.google.com/speech-api/v1/synthesize?text=')) {{
                o.src = o.src.replace(
                    'https://www.google.com/speech-api/v1/synthesize?text=',
                    '/infer/glowtts?text='
                );
                o.html5 = false;
                o.volume = o.volume * 2;
            }}
            return original_function.call(this, o);
        }}
        """
    response = Response(content, resp.status_code, headers)
    return response

@app.route('/infer/glowtts')
def infer_glowtts():
    text = request.args.get('text', '')

    wav = BytesIO()
    if text:
        text = normalize_text(text.strip())
        try:
            audio = generate_audio_glow_tts(text)
            swavfile.write(wav, rate=SAMPLING_RATE, data=audio.numpy())
        except:
            pass

    return send_file(
        wav,
        mimetype='audio/wave',
        attachment_filename='audio.wav'
    )

# @app.route('/infer/fastspeech2')
# def infer_fastspeech2():
#     text = request.args.get('text', '')

#     wav = BytesIO()
#     if text:
#         text = normalize_text(text.strip())
#         try:
#             audio = generate_audio_fastspeech2(text)
#             swavfile.write(wav, rate=SAMPLING_RATE, data=audio.numpy())
#         except:
#             pass

#     return send_file(
#         wav,
#         mimetype='audio/wave',
#         attachment_filename='audio.wav'
#     )

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)