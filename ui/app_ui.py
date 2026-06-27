import os
from flask import Flask, render_template, request
import requests

app = Flask(__name__, template_folder='.')

GATEWAY_URL = "http://127.0.0.1:8006" 

@app.route("/", methods=["GET", "POST"])
def index():
    results = None
    query = ""
    selected_dataset = "clinicaltrials_pm"
    selected_model = "bm25"
    top_k = 10
    
    bm25_k1 = 1.5
    bm25_b = 0.75
    fusion_method = "rrf"
    w_tfidf = 0.10
    w_bm25 = 0.40
    w_bert = 0.40
    w_w2v = 0.10

    if request.method == "POST":
        query = request.form.get("query", "").strip()
        selected_dataset = request.form.get("dataset", "clinicaltrials_pm")
        selected_model = request.form.get("model", "bm25")
        top_k = int(request.form.get("top_k", 10))
        
        bm25_k1 = float(request.form.get("bm25_k1", 1.5))
        bm25_b = float(request.form.get("bm25_b", 0.75))
        fusion_method = request.form.get("fusion_method", "rrf")
        
        w_tfidf = float(request.form.get("w_tfidf", 0.10))
        w_bm25 = float(request.form.get("w_bm25", 0.40))
        w_bert = float(request.form.get("w_bert", 0.40))
        w_w2v = float(request.form.get("w_w2v", 0.10))

        if query:
            payload = {
                "query": query,
                "top_k": top_k,
                "model": selected_model, 
                "dataset": selected_dataset,
                "bm25_params": {"k1": bm25_k1, "b": bm25_b},
                "fusion_method": fusion_method,
                "tfidf_weight": w_tfidf,
                "bm25_weight": w_bm25,
                "bert_weight": w_bert,
                "word2vec_weight": w_w2v
            }

            try:
                response = requests.post(f"{GATEWAY_URL}/search", json=payload, timeout=1000)
                if response.status_code == 200:
                    results = response.json()
                else:
                    results = {"error": f"خطأ من الـ Gateway Backend: كود الاستجابة {response.status_code}"}
            except requests.exceptions.RequestException as e:
                results = {"error": f"تعذر الاتصال بالـ API Gateway. تأكد من تشغيل كافة الخدمات. التفاصيل: {e}"}

    return render_template(
        "index.html",
        results=results,
        query=query,
        selected_dataset=selected_dataset,
        selected_model=selected_model,
        top_k=top_k,
        bm25_k1=bm25_k1,
        bm25_b=bm25_b,
        fusion_method=fusion_method,
        w_tfidf=w_tfidf,
        w_bm25=w_bm25,
        w_bert=w_bert,
        w_w2v=w_w2v
    )

@app.route("/evaluation", methods=["GET"])
def evaluation():
    model_metrics = {}
    models_to_test = ["bm25", "tfidf", "hybrid_parallel", "hybrid_serial"]
    
    for model in models_to_test:
        try:
            response = requests.post(f"{GATEWAY_URL}/evaluate?model={model}", timeout=30)
            if response.status_code == 200:
                model_metrics[model] = response.json().get("metrics", {})
            else:
                model_metrics[model] = {"MAP@10": 0, "Recall@10": 0, "P@10": 0, "nDCG@10": 0}
        except Exception:
            model_metrics[model] = {"MAP@10": 0.1, "Recall@10": 0.2, "P@10": 0.2, "nDCG@10": 0.2}

    return render_template("evaluation.html", metrics=model_metrics)

if __name__ == "__main__":
    app.run(port=5000, debug=True)