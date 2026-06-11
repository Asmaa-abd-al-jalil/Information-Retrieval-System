import os
os.environ["PYTHONUTF8"] = "1"
os.environ["IR_DATASETS_TMP"] = r"C:\Users\MissanAlrifai\ir_tmp"
os.environ["IR_DATASETS_HOME"] = r"C:\Users\MissanAlrifai\.ir_datasets"

# إنشاء المجلد إذا ما موجود
os.makedirs(r"C:\Users\MissanAlrifai\ir_tmp", exist_ok=True)

import ir_datasets

SUPPORTED_DATASETS = {
    "msmarco-passage": "msmarco-passage/trec-dl-2019",
    "fever":           "beir/fever/test"
}

def get_dataset(ds_id: str):
    return ir_datasets.load(ds_id)

def docs_iter(dataset):
    return dataset.docs_iter()

def queries_iter(dataset):
    return dataset.queries_iter()

def qrels_iter(dataset):
    return dataset.qrels_iter()

def count_documents(dataset) -> int:
    return dataset.docs_count()