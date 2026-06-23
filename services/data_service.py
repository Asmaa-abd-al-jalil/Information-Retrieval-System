import os
import ir_datasets
from ir_datasets.util import fileio
from services.preprocessing_service import preprocess_text

# إعدادات البيئة
os.environ["PYTHONUTF8"] = "1"
os.environ["IR_DATASETS_HOME"] = "D:\\ir_storage"
os.environ["IR_DATASETS_TMP"] = "D:\\ir_tmp"
if not os.path.exists("D:\\ir_storage"): os.makedirs("D:\\ir_storage")
if not os.path.exists("D:\\ir_tmp"): os.makedirs("D:\\ir_tmp")
# os.environ["IR_DATASETS_TMP"] = r"C:\Users\MissanAlrifai\ir_tmp"
# os.environ["IR_DATASETS_HOME"] = r"C:\Users\MissanAlrifai\.ir_datasets"
# os.makedirs(r"C:\Users\MissanAlrifai\ir_tmp", exist_ok=True)
# os.environ["IR_DATASETS_TMP"] = r"C:\Users\safa\ir_tmp"
# os.environ["IR_DATASETS_HOME"] = r"C:\Users\safa\.ir_datasets"
# os.makedirs(r"C:\Users\safa\ir_tmp", exist_ok=True)
fileio.verify = lambda: None
# الـ dataset المعتمد
DATASET_ID = "msmarco-passage/trec-dl-2019"
# DATASET_ID = "beir/scifact/test"
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