import os
import ir_datasets
from services.preprocessing_service import preprocess_text
# إعدادات البيئة
os.environ["PYTHONUTF8"] = "1"
os.environ["IR_DATASETS_TMP"] = r"C:\Users\MissanAlrifai\ir_tmp"
os.environ["IR_DATASETS_HOME"] = r"C:\Users\MissanAlrifai\.ir_datasets"
os.makedirs(r"C:\Users\MissanAlrifai\ir_tmp", exist_ok=True)

# تثبيت المجموعة المعتمدة
DATASET_ID = "msmarco-passage/trec-dl-2019/judged"
def get_dataset():
    """تحميل المجموعة التي تحتوي على المستندات + الاستعلامات"""
    return ir_datasets.load(DATASET_ID)

def docs_iter(dataset):
    return dataset.docs_iter()

def queries_iter(dataset):
    return dataset.queries_iter()

def qrels_iter(dataset):
    return dataset.qrels_iter()

def count_documents(dataset) -> int:
    return dataset.docs_count()

def get_processed_doc_text(doc):
    """دالة مساعدة لجلب النص ومعالجته فوراً"""
    return preprocess_text(doc.text)