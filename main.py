import os
os.environ["PYTHONUTF8"] = "1"
os.environ["IR_DATASETS_TMP"] = r"C:\Users\MissanAlrifai\ir_tmp"
os.makedirs(r"C:\Users\MissanAlrifai\ir_tmp", exist_ok=True)

from services.data_service import get_dataset, count_documents, SUPPORTED_DATASETS





def main():
    print("🚀 IR System - التحقق من الـ Datasets")
    print("-" * 50)
    
    for name, ds_id in SUPPORTED_DATASETS.items():
        try:
            print(f"\n🔄 تحميل: {name}")
            ds = get_dataset(ds_id)
            
            count   = count_documents(ds)
            n_q     = sum(1 for _ in ds.queries_iter())
            n_qrels = sum(1 for _ in ds.qrels_iter())
            
            print(f"  ✅ الوثائق  : {count:,}")
            print(f"  ✅ Queries  : {n_q:,}")
            print(f"  ✅ Qrels    : {n_qrels:,}")
            print(f"  ✅ شرط 200K : {'محقق ✅' if count > 200_000 else 'غير محقق ❌'}")

        except Exception as e:
            print(f"  ❌ خطأ: {e}")

    print("\n" + "-" * 50)
    print("🏁 انتهى التحقق.")

if __name__ == "__main__":
    main()