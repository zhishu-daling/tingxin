"""
听心 · Flask Web UI (流式版)

支持流式处理，逐块返回结果
"""
import json
import os
import uuid
import time
import threading
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file

from core.stream_pipeline import StreamPipeline

app = Flask(__name__)

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "temp" / "uploads"
OUTPUT_DIR = BASE_DIR / "output"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_tasks = {}  # {task_id: {status, segments, ...}}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "没有文件"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "文件名为空"}), 400

    ext = os.path.splitext(file.filename)[1]
    task_id = str(uuid.uuid4())[:8]
    save_name = f"{task_id}{ext}"
    save_path = UPLOAD_DIR / save_name
    file.save(str(save_path))

    task = {
        "task_id": task_id,
        "filename": file.filename,
        "path": str(save_path),
        "status": "uploaded",
        "progress": 0,
        "segments": [],
        "result": None,
        "error": None,
    }
    _tasks[task_id] = task

    return jsonify({"task_id": task_id, "filename": file.filename})


@app.route("/api/transcribe/<task_id>", methods=["POST"])
def start_transcribe(task_id):
    """启动流式转写"""
    if task_id not in _tasks:
        return jsonify({"error": "任务不存在"}), 404

    task = _tasks[task_id]
    if task["status"] != "uploaded":
        return jsonify({"error": "任务已处理"}), 400

    language = request.json.get("language", "zh") if request.json else "zh"
    task["status"] = "processing"

    def _stream_callback(chunk_result):
        """每块处理完后的回调"""
        task_ = _tasks.get(task_id)
        if not task_:
            return
        task_["progress"] = chunk_result["progress"]
        task_["segments"].extend(chunk_result["new_segments"])
        task_["total_segments"] = len(task_["segments"])

    def _run():
        try:
            pipe = StreamPipeline(task["path"])
            result = pipe.run(
                language=language,
                use_diarization=True,
                callback=_stream_callback,
                output_dir=str(OUTPUT_DIR),
            )
            task_ = _tasks.get(task_id)
            if task_:
                task_["result"] = result
                task_["status"] = "completed"
                task_["progress"] = 1.0
        except Exception as e:
            import traceback
            task_ = _tasks.get(task_id)
            if task_:
                task_["status"] = "error"
                task_["error"] = str(e)
                task_["traceback"] = traceback.format_exc()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    return jsonify({"task_id": task_id, "status": "processing"})


@app.route("/api/status/<task_id>")
def get_status(task_id):
    """获取任务状态（含流式中间结果）"""
    if task_id not in _tasks:
        return jsonify({"error": "任务不存在"}), 404

    task = _tasks[task_id]

    resp = {
        "task_id": task_id,
        "filename": task["filename"],
        "status": task["status"],
        "progress": task.get("progress", 0),
        "total_segments": task.get("total_segments", 0),
        "error": task.get("error"),
    }

    # 总是返回当前所有片段（不管是否完成）
    resp["segments"] = task.get("segments", [])

    # 如果完成了，返回完整结果
    if task.get("result"):
        r = task["result"]
        resp["result"] = {
            "duration": r["duration"],
            "num_speakers": r["num_speakers"],
            "segments": r["segments"],
        }

    return jsonify(resp)


@app.route("/api/audio/<task_id>")
def serve_audio(task_id):
    if task_id not in _tasks:
        return jsonify({"error": "任务不存在"}), 404
    return send_file(_tasks[task_id]["path"])


@app.route("/api/results")
def list_results():
    results = []
    for task_id, task in _tasks.items():
        results.append({
            "task_id": task_id,
            "filename": task["filename"],
            "status": task["status"],
            "progress": task.get("progress", 0),
            "total_segments": task.get("total_segments", 0),
            "error": task.get("error"),
        })
    return jsonify({"results": results})


if __name__ == "__main__":
    import sys
    if sys.stdout.encoding.lower() == 'gbk':
        import io as _io
        sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    print(" 听心 · 听觉认知系统 (流式版)")
    print("   http://localhost:5001")
    app.run(host="0.0.0.0", port=5001, debug=True)
