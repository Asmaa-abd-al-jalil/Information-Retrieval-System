import os
import ir_datasets
from services.preprocessing_service import preprocess_text

# إعدادات البيئة
os.environ["PYTHONUTF8"] = "1"
# os.environ["IR_DATASETS_TMP"] = r"C:\Users\MissanAlrifai\ir_tmp"
# os.environ["IR_DATASETS_HOME"] = r"C:\Users\MissanAlrifai\.ir_datasets"
# os.makedirs(r"C:\Users\MissanAlrifai\ir_tmp", exist_ok=True)
os.environ["IR_DATASETS_TMP"] = r"C:\Users\safa\ir_tmp"
os.environ["IR_DATASETS_HOME"] = r"C:\Users\safa\.ir_datasets"
os.makedirs(r"C:\Users\safa\ir_tmp", exist_ok=True)

# الـ dataset المعتمد
# DATASET_ID = "msmarco-passage/trec-dl-2019"
DATASET_ID = "beir/scifact/test"
SUPPORTED_DATASETS = {
    "msmarco-passage": DATASET_ID
}

def get_dataset(ds_id: str = DATASET_ID):
    return ir_datasets.load(ds_id)

def docs_iter(dataset):
    return dataset.docs_iter()

def queries_iter(dataset):
    return dataset.queries_iter()

def qrels_iter(dataset):
    return dataset.qrels_iter()

def count_documents(dataset) -> int:
    return dataset.docs_count()

def get_processed_doc_text(doc):
    return preprocess_text(doc.text)