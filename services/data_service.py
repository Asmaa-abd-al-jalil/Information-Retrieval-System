import os

os.environ["PYTHONUTF8"] = "1"
os.environ["IR_DATASETS_HOME"] = "D:\\ir_storage"
os.environ["IR_DATASETS_TMP"] = "D:\\ir_tmp"
os.makedirs("D:\\ir_storage", exist_ok=True)
os.makedirs("D:\\ir_tmp", exist_ok=True)

import ir_datasets
from ir_datasets.util import fileio
from services.preprocessing_service import preprocess_text

fileio.verify = lambda: None

DATASET_ID = "clinicaltrials/2019/trec-pm-2019"

SUPPORTED_DATASETS = {
    "clinicaltrials_pm": DATASET_ID
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

DOCS_CACHE = {doc.doc_id: doc for doc in get_dataset().docs_iter()}
def get_full_document_data(doc_id: str):
    doc = DOCS_CACHE.get(doc_id)
    if doc:
            return {
                "title": doc.title,
                "summary": doc.summary,
                "detailed_description": doc.detailed_description,
                "eligibility": doc.eligibility
            }
    return {"title": "غير متوفر", "summary": "غير متوفر", "detailed_description": "غير متوفر", "eligibility": "غير متوفر"}