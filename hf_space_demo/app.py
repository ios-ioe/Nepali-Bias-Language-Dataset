from __future__ import annotations

import json
import os
import subprocess
import threading
from pathlib import Path
from typing import List, Tuple

import gradio as gr


# Paths and config
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT.parent
MODEL_DIR = ROOT / "outputs" / "transformer_baseline"
LABELS_FILE = MODEL_DIR / "label_names.json"
TRAIN_SCRIPT = DATA_DIR / "scripts" / "train_transformer.py"
SAMPLE_PATH = ROOT / "assets" / "sample_examples.json"


def load_sample_examples() -> List[str]:
    if not SAMPLE_PATH.exists():
        return [
            "मधेसी पुरुषहरूले मुख्य रूपमा प्रदर्शन र आन्दोलनको नेतृत्व गरेका छन्।",
            "यादव, महतो, साह जस्ता मधेसी जातिका व्यक्तिहरूले सधैं अशान्ति बढाउने गर्छन्।",
            "युवा महिलाहरू राजनीति बुझ्दैनन् भन्ने गलत धारणा छ।",
        ]
    try:
        data = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
        examples = [item["text"] for item in data if isinstance(item, dict) and item.get("text")]
        return examples[:24]
    except Exception:
        return []


def model_available() -> bool:
    return (LABELS_FILE.exists() and ((MODEL_DIR / "pytorch_model.bin").exists() or (MODEL_DIR / "flax_model.msgpack").exists()))


class ModelWrapper:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.label_names: List[str] = []

    def load(self):
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            self.tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
            self.model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
            if LABELS_FILE.exists():
                self.label_names = json.loads(LABELS_FILE.read_text(encoding="utf-8"))
            else:
                self.label_names = []
            return True
        except Exception as e:
            print("Model load failed:", e)
            return False

    def predict(self, text: str) -> List[Tuple[str, float]]:
        if self.model is None or self.tokenizer is None:
            return []
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with __import__("torch").no_grad():
            logits = self.model(**inputs).logits
            import numpy as np

            probs = 1 / (1 + np.exp(-logits.detach().cpu().numpy()))
            probs = probs.ravel()
        if not self.label_names:
            labels = [f"label_{i}" for i in range(len(probs))]
        else:
            labels = self.label_names
        return sorted(list(zip(labels, probs.tolist())), key=lambda x: x[1], reverse=True)


model_wrapper = ModelWrapper()
training_thread = None
training_lock = threading.Lock()


def start_background_training() -> str:
    global training_thread
    with training_lock:
        if model_available():
            return "Model already available."
        if training_thread and training_thread.is_alive():
            return "Training already in progress."

        def run_training():
            cmd = ["python3", str(TRAIN_SCRIPT), "--output_dir", str(MODEL_DIR), "--epochs", "3"]
            env = os.environ.copy()
            # Run training; output will go to Space logs
            try:
                subprocess.run(cmd, check=True, cwd=str(DATA_DIR))
            except Exception as e:
                print("Background training failed:", e)
            # attempt to load model after training
            model_wrapper.load()

        training_thread = threading.Thread(target=run_training, daemon=True)
        training_thread.start()
        return "Training started in background. Check logs for progress."


def maybe_load_model():
    if model_available():
        ok = model_wrapper.load()
        return ok
    return False


def predict_with_model(text: str):
    if model_wrapper.model is None:
        return {"status": "model_missing"}, "Model not available. Click \"Start training\" to begin training on the Space."
    scores = model_wrapper.predict(text)
    # return top 5 as label:score mapping and a short explanation
    top = scores[:5]
    explanation = f"Top label: {top[0][0]} ({top[0][1]:.2f})" if top else "No prediction"
    return {label: float(score) for label, score in scores}, explanation


with gr.Blocks(theme=gr.themes.Soft(), title="Nepali Bias Detection Demo") as demo:
    gr.Markdown(
        """
        # Nepali Bias Detection Demo (Transformer-backed)
        
        This Space will train a transformer model on the dataset on first run if no pretrained model is present.
        Training runs in the background and the UI will load the trained model when available.
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(label="Enter Nepali text", lines=5, placeholder="Paste a sentence here...")
            with gr.Row():
                predict_btn = gr.Button("Predict", variant="primary")
                train_btn = gr.Button("Start training on Space")
                status_btn = gr.Button("Check model status")
        with gr.Column(scale=2):
            output = gr.Label(label="Bias scores", num_top_classes=5)
            summary = gr.Textbox(label="Explanation", lines=3, interactive=False)

    examples = load_sample_examples()
    sample_btn = gr.Button("Load sample")
    sample_btn.click(lambda: __import__("random").choice(examples), inputs=None, outputs=text_input)

    def start_training_and_report():
        return start_background_training()

    def check_status():
        if model_available():
            loaded = maybe_load_model()
            return "Model ready and loaded." if loaded else "Model files present but failed to load. Check logs."
        else:
            if training_thread and training_thread.is_alive():
                return "Training in progress. Check logs for details."
            return "No model and no training in progress."

    predict_btn.click(fn=predict_with_model, inputs=text_input, outputs=[output, summary])
    train_btn.click(fn=start_training_and_report, inputs=None, outputs=summary)
    status_btn.click(fn=check_status, inputs=None, outputs=summary)

    # Attempt to load model at startup
    maybe_load_model()


if __name__ == "__main__":
    demo.launch()
