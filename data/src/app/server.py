import requests
import io
from flask import (
    Flask,
    request,
    render_template,
    jsonify,
    Response,
    redirect,
    url_for,
)

from .synthesys import synthesize
from .text_processer import normalize_text, normalize_multiline_text


app = Flask(__name__)
with io.BytesIO() as wav_io:
    synthesize("테스트", wav_io)  # force load model


@app.after_request
def allow_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response


@app.route("/")
def index():
    return redirect(url_for("text_inference"))


@app.route("/tts-server/text-inference")
def text_inference():
    return render_template("text-inference.html")


@app.route("/tts-server/oc-overlay")
def open_captions_overlay():
    return render_template("oc-overlay.html")


@app.route("/tts-server/api/process-text", methods=["POST"])
def text():
    text = request.json.get("text", "")
    texts = normalize_multiline_text(text)

    return jsonify(texts)


@app.route("/tts-server/api/infer")
def infer():
    text = request.args.get("text", "")
    text = normalize_text(text).strip()

    if not text:
        return "text shouldn't be empty", 400

    try:
        with io.BytesIO() as wav_io:
            return synthesize(text, wav_io)
    except Exception as e:
        return f"Cannot generate audio: {str(e)}", 500


@app.route("/favicon.ico")
def favicon():
    return "I don't have favicon :p", 404


@app.route("/<path:path>")
def twip_proxy(path):
    new_url = request.url.replace(request.host, "twip.kr")
    new_url = new_url.replace("http://", "https://")
    resp = requests.request(
        method=request.method,
        url=new_url,
        headers={key: value for (key, value) in request.headers if key != "Host"},
        data=request.get_data(),
        cookies=request.cookies,
        allow_redirects=False,
    )
    excluded_headers = [
        "content-encoding",
        "content-length",
        "transfer-encoding",
        "connection",
    ]
    headers = [
        (name, value)
        for (name, value) in resp.raw.headers.items()
        if name.lower() not in excluded_headers
    ]
    content = resp.content
    if new_url.startswith("https://twip.kr/assets/js/alertbox/"):
        content = resp.text.replace(
            "https://www.google.com/speech-api/v1/synthesize?",
            "/tts-server/api/infer?",
        )
    response = Response(content, resp.status_code, headers)
    return response
