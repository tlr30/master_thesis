from flask import Flask, request, Response, stream_with_context
import subprocess
import sys

app = Flask(__name__)

@app.route("/run-script", methods=["POST"])
def run_script():
    data = request.get_json()
    command = data.get("command", "")

    def generate():
        process = subprocess.Popen(
            ["python", "speech/command_execution.py", command],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            bufsize=1
        )

        for line in iter(process.stdout.readline, ""):
            if line:
                yield line
                # Optionally flush stdout if necessary
                sys.stdout.flush()

        process.stdout.close()
        process.wait()


    return Response(stream_with_context(generate()), mimetype="text/plain")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
