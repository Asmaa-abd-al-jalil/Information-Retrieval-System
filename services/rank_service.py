import os
import json
import math
from collections import defaultdict
from services.preprocessing_service import preprocess_text

# مسار حفظ الـ Index
INDEX_DIR = "data/indexes"
os.makedirs(INDEX_DIR, exist_ok=True)

INVERTED_INDEX_PATH = os.path.join(INDEX_DIR, "inverted_index.json")
DOC_LENGTHS_PATH = os.path.join(INDEX_DIR, "doc_lengths.json")
DOC_COUNT_PATH = os.path.join(INDEX_DIR, "doc_count.json")

def build_inverted_index(dataset, max_docs: int = None):
    """
    بناء الـ Inverted Index من الـ dataset
    max_docs: عدد الوثائق للمعالجة (None = كل الوثائق)
    """
    print("🔨 جاري بناء الـ Inverted Index...")

    inverted_index = defaultdict(dict)  # {term: {doc_id: tf}}
    doc_lengths = {}                    # {doc_id: عدد الكلمات}
    doc_count = 0

    for i, doc in enumerate(dataset.docs_iter()):
        if max_docs and i >= max_docs:
            break

        # معالجة النص
        result = preprocess_text(doc.text)
        tokens = result['final_tokens']

        if not tokens:
            continue

        # حساب الـ TF لكل كلمة
        tf_counts = defaultdict(int)
        for token in tokens:
            tf_counts[token] += 1

        # إضافة للـ Inverted Index
        for token, count in tf_counts.items():
            inverted_index[token][doc.doc_id] = count

        # حفظ طول الوثيقة
        doc_lengths[doc.doc_id] = len(tokens)
        doc_count += 1

        if doc_count % 10000 == 0:
            print(f"  📄 تمت معالجة {doc_count:,} وثيقة...")

    print(f"✅ تم بناء الـ Index لـ {doc_count:,} وثيقة")
    print(f"📚 عدد المصطلحات الفريدة: {len(inverted_index):,}")

    return dict(inverted_index), doc_lengths, doc_count


def save_index(inverted_index, doc_lengths, doc_count):
    """حفظ الـ Index على الـ disk"""
    print("💾 جاري حفظ الـ Index...")

    with open(INVERTED_INDEX_PATH, 'w', encoding='utf-8') as f:
        json.dump(inverted_index, f)

    with open(DOC_LENGTHS_PATH, 'w', encoding='utf-8') as f:
        json.dump(doc_lengths, f)

    with open(DOC_COUNT_PATH, 'w', encoding='utf-8') as f:
        json.dump({"doc_count": doc_count}, f)

    print("✅ تم حفظ الـ Index بنجاح")


def load_index():
    """تحميل الـ Index من الـ disk"""
    if not os.path.exists(INVERTED_INDEX_PATH):
        raise FileNotFoundError("❌ الـ Index غير موجود، شغّل build أولاً")

    with open(INVERTED_INDEX_PATH, 'r', encoding='utf-8') as f:
        inverted_index = json.load(f)

    with open(DOC_LENGTHS_PATH, 'r', encoding='utf-8') as f:
        doc_lengths = json.load(f)

    with open(DOC_COUNT_PATH, 'r', encoding='utf-8') as f:
        doc_count = json.load(f)["doc_count"]

    print(f"✅ تم تحميل الـ Index: {len(inverted_index):,} مصطلح، {doc_count:,} وثيقة")
    return inverted_index, doc_lengths, doc_count


def get_index_stats(inverted_index, doc_lengths, doc_count):
    """إحصائيات الـ Index"""
    avg_doc_length = sum(doc_lengths.values()) / len(doc_lengths) if doc_lengths else 0
    return {
        "doc_count": doc_count,
        "unique_terms": len(inverted_index),
        "avg_doc_length": round(avg_doc_length, 2)
    }