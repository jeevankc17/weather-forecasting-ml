# ─────────────────────────────────────────────────────────────
# api.py — Vercel API for scheduled retraining
# Called by Vercel Cron Job
# ─────────────────────────────────────────────────────────────

from flask import Flask, jsonify
import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({
        "status": "ok",
        "message": "Weather Forecaster API"
    })

@app.route("/retrain")
def retrain():
    """Trigger model retraining - called by Vercel Cron"""
    try:
        from node1_forecaster import retrain as do_retrain
        do_retrain()
        return jsonify({"status": "success", "message": "Model retrained"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/health")
def health():
    return jsonify({"status": "healthy"})

# Vercel handler
def handler(request):
    return app(request)