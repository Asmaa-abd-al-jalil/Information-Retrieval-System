# 🔍 Information Retrieval System

## 📋 وصف المشروع
نظام استرجاع معلومات مبني على مجموعة بيانات MS MARCO Passage.


## ⚙️ متطلبات التشغيل
```bash
conda activate ir_project
pip install ir-datasets nltk sentence-transformers gensim
```

## 🚀 طريقة التشغيل
```bash
# أول مرة فقط - بناء الـ Index والـ Database
$env:PYTHONUTF8 = "1"
python main.py
```

## ✅ الطلبات المنجزة
- [x] الطلب 1: Data Preprocessing
- [x] الطلب 2: تمثيل الوثائق (TF-IDF, BM25, BERT, Word2Vec, Hybrid)
- [x] الطلب 3: Indexing + Index Terms Selection
- [x] الطلب 4: Query Processing
- [ ] الطلب 5: Query Refinement
- [ ] الطلب 6: Matching & Ranking
- [ ] الطلب 7: SOA Architecture
- [ ] الطلب 8: Evaluation
- [ ] الطلب 9: واجهة المستخدم

## 📝 ملاحظات للفريق
- شغّل `$env:PYTHONUTF8 = "1"` قبل أي تشغيل على Windows
- الـ Index والـ Database بيتبنوا تلقائياً أول مرة
- Word2Vec بيتحفظ على الـ disk بعد التدريب ويتحمّل تلقائياً
- هلق الـ Index على 1,000 وثيقة للاختبار — قبل التسليم غيّر `max_docs=1000` لـ `max_docs=None`
