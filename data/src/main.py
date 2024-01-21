import os

from app.server import app as flask_app


def run():
    flask_app.run(host="0.0.0.0", debug=os.environ.get("TTS_DEBUG", "0") == "1")


if __name__ == "__main__":
    run()
