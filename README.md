# 🔍 Information Retrieval System

## 📋 وصف المشروع
نظام استرجاع معلومات مبني على مجموعة بيانات MS MARCO Passage.

## 👥 تقسيم العمل
| العضو | المهمة | الملف |
|---|---|---|
| عضو 1 | Data & Preprocessing | `services/preprocessing_service.py` |
| عضو 2 | Indexing Service | `services/index_service.py` |
| عضو 3 | Retrieval Service | `services/retrieval_service.py` |
| عضو 4 | Ranking & Evaluation | `services/rank_service.py` |
| عضو 5 | Query Refinement | `services/query_service.py` |
| عضو 6 | API Gateway & UI | `main.py` |

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
