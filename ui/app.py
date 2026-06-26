from flask import Flask, render_template, request
import requests

app = Flask(__name__, template_folder='.')

# عنوان الـ API Gateway الرئيسي أو الخدمات المباشرة
# سنعتمد هنا إرسال الطلبات للـ Gateway الذي ينسق العمل
GATEWAY_URL = "http://127.0.0.1:8005/search" # تأكدي من منفذ الـ Gateway لديكِ

@app.route("/", methods=["GET", "POST"])
def index():
    results = None
    query = ""
    selected_dataset = "dataset_1"
    execution_mode = "basic"
    hybrid_model = "none"
    bm25_k1 = 1.5
    bm25_b = 0.75
    
    if request.method == "POST":
        # 1. استقبال البيانات من الواجهة
        query = request.form.get("query", "")
        selected_dataset = request.form.get("dataset", "dataset_1")
        execution_mode = request.form.get("execution_mode", "basic")
        hybrid_model = request.form.get("hybrid_model", "none")
        bm25_k1 = float(request.form.get("bm25_k1", 1.5))
        bm25_b = float(request.form.get("bm25_b", 0.75))
        
        # 2. تجهيز الـ Payload لإرساله إلى الـ API
        # نمرر كافة الخيارات التي حددها المستخدم بالواجهة لتأخذها الخدمات بعين الاعتبار
        payload = {
            "query": query,
            "dataset": selected_dataset,
            "execution_mode": execution_mode,
            "hybrid_model": hybrid_model,
            "bm25_params": {"k1": bm25_k1, "b": bm25_b},
            "top_k": 10
        }
        
        try:
            # 3. استدعاء الخدمة الخلفية للحصول على النتائج
            response = requests.post(GATEWAY_URL, json=payload)
            if response.status_code == 200:
                results = response.json()
            else:
                results = {"error": f"Error from backend API: {response.status_code}"}
        except requests.exceptions.ConnectionError:
            results = {"error": "Could not connect to the backend API. Please make sure the Gateway/Services are running."}

    return render_template(
        "index.html", 
        results=results, 
        query=query, 
        selected_dataset=selected_dataset,
        execution_mode=execution_mode,
        hybrid_model=hybrid_model,
        bm25_k1=bm25_k1,
        bm25_b=bm25_b
    )

if __name__ == "__main__":
    # تشغيل واجهة المستخدم على منفذ مستقل مخصص للـ UI (مثلاً 5000)
    app.run(port=5000, debug=True)