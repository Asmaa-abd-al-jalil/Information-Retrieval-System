import os
import ir_datasets
from ir_datasets.util import fileio
from services.preprocessing_service import preprocess_text

os.environ["PYTHONUTF8"] = "1"
os.environ["IR_DATASETS_HOME"] = "D:\\ir_storage"
os.environ["IR_DATASETS_TMP"] = "D:\\ir_tmp"

os.makedirs("D:\\ir_storage", exist_ok=True)
os.makedirs("D:\\ir_tmp", exist_ok=True)

fileio.verify = lambda: None

DATASET_ID = "beir/nq"
SUPPORTED_DATASETS = {
    "beir-nq": DATASET_ID
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