import sys
import io
from services.data_service import get_dataset, count_documents, get_processed_doc_text

# ضبط الترميز للنظام
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def run_data_validation():
    """المرحلة الأولى: التحقق من البيانات"""
    print("\n🚀 المرحلة الأولى: التحقق من البيانات")
    print("-" * 50)
    try:
        ds = get_dataset()
        count = count_documents(ds)
        n_q = sum(1 for _ in ds.queries_iter())
        n_qrels = sum(1 for _ in ds.qrels_iter())
        
        print(f"  ✅ الوثائق  : {count:,}")
        print(f"  ✅ Queries  : {n_q:,}")
        print(f"  ✅ Qrels    : {n_qrels:,}")
        print(f"  ✅ شرط 200K : {'محقق ✅' if count > 200_000 else 'غير محقق ❌'}")
    except Exception as e:
        print(f"  ❌ خطأ أثناء التحميل: {e}")

def run_preprocessing_test():
    """المرحلة الثانية: اختبار المعالجة الأولية"""
    print("\n🚀 المرحلة الثانية: اختبار المعالجة (Preprocessing Test)")
    print("-" * 50)
    try:
        ds = get_dataset()
        sample_doc = next(iter(ds.docs_iter()))
        
        print("--- النص الأصلي ---")
        print(sample_doc.text[:200])
        
        print("\n--- النص بعد المعالجة ---")
        processed = get_processed_doc_text(sample_doc)
        print(processed[:200])
    except Exception as e:
        print(f"  ❌ خطأ أثناء المعالجة: {e}")

def run_real_data_test():
    """المرحلة الثالثة: اختبار المعالجة على جملة حقيقية من الداتا سيت"""
    print("\n🚀 [اختبار حقيقي] جاري جلب جملة من MS MARCO...")
    try:
        ds = get_dataset()
        # جلب أول وثيقة حقيقية من الداتا سيت
        sample_doc = next(iter(ds.docs_iter()))
        
        print(f"\n📌 النص الأصلي من الداتا سيت:")
        print(sample_doc.text[:200]) # طباعة أول 200 حرف
        
        print(f"\n✨ النتيجة بعد المعالجة:")
        print(get_processed_doc_text(sample_doc)[:200])
        
    except Exception as e:
        print(f"  ❌ خطأ أثناء جلب الداتا سيت: {e}")       

def main():
    print("✨ بدء نظام استرجاع المعلومات - IR System")
    
    # تحكم بالمراحل هنا: يمكنك تعطيل أي مرحلة بوضع # قبلها
   # run_data_validation()
    #run_preprocessing_test()
    run_real_data_test()
    
    print("\n" + "-" * 50)
    print("🏁 انتهت جميع المراحل بنجاح. النظام جاهز للخطوة التالية (Indexing)!")

if __name__ == "__main__":
    main()


# .........................






# from services.data_service import get_processed_doc_text

# def main():
#     print("🚀 بدء اختبار المعالجة على جملة مخصصة...")
    
#     # ضعي أي جملة تريدينها هنا للتجربة
#     my_test_sentence = "The researchers were studying the efFFfects of Information Retrieval!"
    
#     print(f"\n--- الجملة الأصلية ---")
#     print(my_test_sentence)
    
#     print(f"\n--- النتيجة بعد المعالجة ---")
#     processed = get_processed_doc_text(type("obj", (object,), {"text": my_test_sentence})())
#     print(processed)
    
#     print("\n✅ انتهى الاختبار.")

# if __name__ == "__main__":
#     main()