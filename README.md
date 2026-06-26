# نظام استرجاع المعلومات المتقدم - Information Retrieval System

## 1. وصف المشروع (Project Overview)
نظام متطور لاسترجاع الوثائق الطبية مبني بلغة Python وفق معمارية البرمجيات القائمة على الخدمات (Service-Oriented Architecture - SOA)، حيث تعمل كل عملية كميكروسيرفيس (Microservice) مستقلة تخاطب بقية الخدمات عبر بروتوكول HTTP من خلال الـ API Gateway، مما يضمن كفاءة المنظومة وفصل المسؤوليات برمجياً.

---

## 2. مواصفات مجموعة البيانات المعتمدة (Dataset Specifications)
تم اختيار مجموعة بيانات **TREC Precision Medicine (ClinicalTrials PM)** لتلبية شروط النظام القياسية:
* **حجم البيانات:** **306,238 وثيقة طبية**، مما يحقق شرط اختبار كفاءة الفهارس وسرعة الاستجابة للبيانات الضخمة.
* **بيانات الاختبار (Queries):** تعتمد على ملف بنيوي قياسي (`queries.xml`) يضم حالات مرضية مقسمة إلى: المرض (`<disease>`)، الجين والمطفر (`<gene>`)، والعمر والجنس (`<demographic>`).
* **روابط الملاءمة (Qrels):** تشتمل على الأحكام القياسية للملاءمة (Relevance Judgments) كمعيار ذهبي لحساب دقة وموثوقية محرك البحث.

---

## 3. البنية الهيكلية للمشروع وتوصيف المجلدات (Directory Architecture)

```text
├── apis/
│   ├── api_indexing.py
│   ├── api_retrieval.py
│   ├── api_ranking.py
│   ├── api_refinement.py
│   ├── api_preprocessing.py
│   ├── api_clustering.py
│   ├── api_crawling.py
│   ├── api_evaluation.py
│   └── gateway.py
├── services/
│   ├── ClusteringService.py
│   ├── CrawlingService.py
│   ├── data_service.py
│   ├── database_service.py
│   ├── index_service.py
│   ├── preprocessing_service.py
│   └── retrieval_service.py
├── notebooks/
│   └── evaluation_results.ipynb
├── query/
│   ├── query_refinement_service.py
│   └── query_service.py
├── rank/
│   ├── matching_service.py
│   └── rank_service.py
├── evaluation/
│   ├── compare_models.py
│   └── evaluate_system.py
├── ui/
│   ├── app_ui.py
│   ├── evaluation.html
│   └── index.html
├── data/
│   ├── indexes/
│   ├── documents.db
│   └── word2vec.model
└── tests/
│   ├── test_matching.py
│   └── test_query.py
├── .gitignore
├── main.py
├── requirements.txt
├── run_all.bat
└── README.md
```
* **طبقة واجهات برمجة التطبيقات (apis):** تضم جميع نقاط النهاية (Endpoints) الخاصة بالخدمات المستقلة، بالإضافة إلى بوابة النظام (`gateway.py`) المسؤولة عن استقبال الطلبات وتوجيهها وربط الخدمات المختلفة.
* **طبقة الخدمات (services):** تمثل النواة البرمجية للنظام، وتتكفل بتشغيل الخدمات وإدارة منطق الأعمال وتنسيق عمليات معالجة الاستعلامات والفهرسة.
* **وحدات معالجة الاستعلامات (query):** تحتوي على أدوات معالجة اللغة الطبيعية، مثل تنظيف النصوص، إزالة الكلمات غير المهمة والرموز، إجراء التجذير (Stemming)، وتوسيع المصطلحات دلالياً.
* **نماذج الترتيب والاسترجاع (rank):** تضم خوارزميات استرجاع المعلومات المختلفة، مثل نموذج الفضاء المتجهي (TF-IDF)، ونموذج BM25 الاحتمالي، والاسترجاع الدلالي باستخدام BERT وWord2Vec، بالإضافة إلى خوارزميات الدمج الهجين (Fusion).
* **خدمة تجميع الوثائق (clustering):** مسؤولة عن تجميع الوثائق المسترجعة المتشابهة دلالياً في مجموعات متجانسة لتسهيل عملية استعراض النتائج.
* **خدمة الزحف (crawling):** تتولى جمع البيانات الطبية من المصادر المستهدفة على الويب وتحديث قاعدة بيانات النظام والفهارس بصورة دورية.
* **وحدة التقييم (evaluation):** مسؤولة عن تقييم جودة نتائج الاسترجاع بالاعتماد على ملفات الأحكام القياسية (Qrels)، وحساب المقاييس العلمية المختلفة.
* **واجهة المستخدم (ui):** توفر واجهة رسومية مبنية باستخدام Flask تتيح للمستخدم إدخال الاستعلامات، اختيار النماذج، التحكم بالمعاملات، واستعراض النتائج ومقاييس الأداء.
* **مستودع البيانات (data):** يحتوي على الفهارس المعكوسة (Inverted Indexes) ومخازن المتجهات (Vector Stores) المستخدمة أثناء عمليات البحث والاسترجاع.
* **الاختبارات (tests):** تضم اختبارات الوحدات (Unit Tests) واختبارات التكامل (Integration Tests) للتحقق من سلامة كل خدمة وضمان استقرار النظام بعد ربط جميع مكوناته.

---

## 4. تعليمات وإرشادات تشغيل الخدمات (Running Instructions)

قبل البدء، تأكد من تثبيت الحزم البرمجية المطلوبة داخل بيئة العمل:
```bash
pip install -r requirements.txt
```

يجب تشغيل كل خدمة في نافذة Terminal مستقلة وفق الترتيب التالي:

### أولاً: تشغيل الخدمات الخلفية (Core Microservices)

#### خدمة استرجاع وتصفية الوثائق (Retrieval Service - Port 8002)

```bash
python -m uvicorn apis.api_retrieval:app --port 8002 --reload
```

#### خدمة إعادة الترتيب والمطابقة (Ranking Service - Port 8003)

```bash
python -m uvicorn apis.api_ranking:app --port 8003 --reload
```

#### خدمة معالجة وتحسين الاستعلام (Refinement Service - Port 8004)

```bash
python -m uvicorn apis.api_refinement:app --port 8004 --reload
```

#### خدمة عنقودة الوثائق (Clustering Service - Port 8005)

```bash
python -m uvicorn apis.api_clustering:app --port 8005 --reload
```

#### خدمة الزحف وجلب البيانات (Crawling Service - Port 8007)

```bash
python -m uvicorn apis.api_crawling:app --port 8007 --reload
```

#### خدمة تقييم أداء النظام (Evaluation Service - Port 8008)

```bash
python -m uvicorn apis.api_evaluation:app --port 8008
```

### ثانياً: تشغيل البوابة والواجهة الرسومية (Gateway & UI)

#### بوابة النظام الرئيسية (API Gateway - Port 8006)

```bash
python -m uvicorn apis.gateway:app --port 8006 --reload
```

#### واجهة المستخدم الرسومية (Flask UI - Port 5000)

```bash
python app_ui.py
```

بعد تشغيل جميع الخدمات، افتح المتصفح وانتقل إلى الرابط التالي لاستخدام النظام:

```text
http://127.0.0.1:5000
```

---

## 5. التشغيل التلقائي بضغطة زر واحدة (Automation Script)

لتسهيل تشغيل النظام، يتضمن المشروع ملف تشغيل تلقائي باسم **run_all.bat** داخل المجلد الرئيسي.

يقوم هذا الملف بفتح جميع نوافذ Terminal اللازمة وتشغيل كافة خدمات النظام بالتوازي دون الحاجة إلى تشغيل كل خدمة يدوياً.

لتشغيل النظام:

```bash
run_all.bat
```

---

## 6. مقاييس التقييم المدعومة (Evaluation Metrics)

يقوم النظام بحساب مجموعة من المقاييس القياسية المستخدمة في تقييم أنظمة استرجاع المعلومات، وذلك بالاعتماد على ملفات **Qrels** الخاصة بمجموعة بيانات **TREC Precision Medicine**.

### MAP (Mean Average Precision)

يقيس متوسط الدقة عبر جميع الاستعلامات، ويعد من أهم المقاييس المستخدمة لتقييم جودة ترتيب الوثائق المسترجعة.

### Recall@10

يقيس قدرة النظام على استرجاع الوثائق ذات الصلة ضمن أول عشر نتائج يتم إرجاعها.

### Precision@10

يقيس نسبة الوثائق ذات الصلة الموجودة ضمن أول عشر نتائج معروضة للمستخدم.

### nDCG@10 (Normalized Discounted Cumulative Gain)

يقيس جودة ترتيب النتائج مع الأخذ بعين الاعتبار درجة ملاءمة الوثيقة وموقعها داخل قائمة النتائج، بحيث يمنح الوثائق الأكثر صلة وزناً أكبر إذا ظهرت في المراتب الأولى.

---

## 7. التقنيات المستخدمة (Technologies Used)

يعتمد المشروع على مجموعة من التقنيات والأدوات الحديثة، من أبرزها:

- Python
- FastAPI
- Flask
- Uvicorn
- Scikit-learn
- Rank-BM25
- Sentence Transformers (BERT)
- Word2Vec
- Pandas
- NumPy
- XML Parsing
- RESTful APIs
- Service-Oriented Architecture (SOA)

---

## 8. مكونات نظام الاسترجاع (Retrieval Pipeline)

يمر الاستعلام داخل النظام بعدة مراحل متتابعة كما يلي:

1. استقبال الاستعلام من واجهة المستخدم.
2. معالجة النص وإزالة الكلمات غير المهمة.
3. تنفيذ عمليات التجذير وتوسيع المصطلحات.
4. البحث داخل الفهرس المعكوس.
5. تطبيق نموذج الاسترجاع المختار (TF-IDF أو BM25 أو Semantic Retrieval).
6. دمج النتائج باستخدام خوارزميات Fusion عند الحاجة.
7. إعادة ترتيب النتائج.
8. تجميع الوثائق المتشابهة دلالياً.
9. حساب مقاييس الأداء.
10. عرض النتائج للمستخدم.

---

## 9. اختبارات النظام (Testing)

يحتوي المشروع على اختبارات وحدات (Unit Tests) واختبارات تكامل (Integration Tests) للتحقق من صحة عمل الخدمات بشكل مستقل، إضافةً إلى اختبار تكامل النظام بالكامل بعد تشغيل جميع الخدمات.

لتشغيل الاختبارات:

```bash
python -m pytest tests/
```

---

## 10. ملاحظات

- يعتمد النظام على معمارية الخدمات المستقلة (SOA)، بحيث تعمل كل خدمة بشكل منفصل وتتواصل مع بقية الخدمات عبر HTTP.
- يمكن تشغيل كل خدمة أو تطويرها بشكل مستقل دون التأثير على بقية مكونات النظام.
- تم تصميم المشروع بحيث يدعم إضافة نماذج استرجاع جديدة أو خوارزميات ترتيب إضافية بسهولة.
- يستخدم النظام مجموعة بيانات TREC Precision Medicine كمرجع معياري لتقييم جودة الاسترجاع.

---

