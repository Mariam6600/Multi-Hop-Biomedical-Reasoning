"""
src/verify_cache.py
====================
للتحقق من اكتمال وجودة الـ N-Terms في expansion_cache.json.

Usage:
    py -3.10 src/verify_cache.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    MEDHOP_FILE,
    OUTPUTS_DIR,
    EXPANSION_N_TERMS,
)

CACHE_FILE = os.path.join(OUTPUTS_DIR, "expansion_cache.json")

def main():
    print("\n" + "=" * 70)
    print("  N-TERMS CACHE VERIFICATION & QUALITY CHECK")
    print("=" * 70)

    # 1. تحميل بيانات MedHop
    if not os.path.exists(MEDHOP_FILE):
        print(f"  [FAIL] MedHop file not found: {MEDHOP_FILE}")
        return

    with open(MEDHOP_FILE, encoding="utf-8") as f:
        data = json.load(f)
    
    # استخراج أسماء الأدوية الفريدة
    drugs_in_data = set()
    for r in data:
        name = r.get("query_drug_name")
        if name:
            drugs_in_data.add(name)
    
    total_drugs = len(drugs_in_data)
    print(f"\n  [INFO] Drugs in dataset : {total_drugs}")

    # 2. تحميل الكاش
    if not os.path.exists(CACHE_FILE):
        print(f"  [FAIL] Cache file not found: {CACHE_FILE}")
        print("         Run inference_pipeline5.py first to generate cache.")
        return

    with open(CACHE_FILE, encoding="utf-8") as f:
        cache = json.load(f)
    
    print(f"  [INFO] Cache entries    : {len(cache)}")

    # 3. التحقق من التغطية (Coverage)
    found_drugs = []
    missing_drugs = []
    
    # مفاتيح الكاش المحتملة (قديم من السحابة، جديد من المحلي)
    for drug in drugs_in_data:
        old_key = f"qwen/qwen3.6-plus-preview:free::{drug}::{EXPANSION_N_TERMS}"
        new_key = f"local::{drug}::{EXPANSION_N_TERMS}"
        
        entry = None
        if old_key in cache:
            entry = cache[old_key]
        elif new_key in cache:
            entry = cache[new_key]
        
        if entry and entry.get("success"):
            found_drugs.append(drug)
        else:
            missing_drugs.append(drug)

    coverage = (len(found_drugs) / total_drugs * 100) if total_drugs > 0 else 0

    print("\n" + "-" * 70)
    print("  COVERAGE STATUS")
    print("-" * 70)
    print(f"  ✅ Found    : {len(found_drugs)} drugs ({coverage:.1f}%)")
    print(f"  ❌ Missing  : {len(missing_drugs)} drugs")
    
    if missing_drugs:
        print("\n  [WARN] Missing drugs (first 10):")
        for d in missing_drugs[:10]:
            print(f"    - {d}")
    else:
        print("\n  ✅ All drugs have N-terms cached!")

    # 4. تقييم الجودة (Quality Check)
    print("\n" + "-" * 70)
    print("  QUALITY CHECK (Sample Terms)")
    print("-" * 70)
    print("  Checking if terms are relevant, clean, and correct count...\n")

    good_quality = 0
    bad_quality = 0
    sample_size = min(10, len(found_drugs)) # فحص أول 10 أدوية كعينة
    
    for i, drug in enumerate(found_drugs[:sample_size]):
        old_key = f"qwen/qwen3.6-plus-preview:free::{drug}::{EXPANSION_N_TERMS}"
        new_key = f"local::{drug}::{EXPANSION_N_TERMS}"
        
        entry = cache.get(old_key) or cache.get(new_key)
        
        if entry:
            terms = entry.get("terms", [])
            count = len(terms)
            
            # معايير الجودة:
            # 1. العدد قريب من المطلوب (30)
            # 2. لا توجد فراغات طويلة (junk)
            
            status = "OK"
            notes = []
            
            if count < (EXPANSION_N_TERMS - 5): # مثلاً أقل من 25
                status = "WARN"
                notes.append(f"Low count ({count})")
            
            # التحقق من وجود كلمات غير مفيدة (heuristic)
            # معظم المصطلحات الطبية كلمات قصيرة أو متوسطة
            avg_len = sum(len(t) for t in terms) / count if count > 0 else 0
            if avg_len > 30: # متوسط طول الكلمة لا يجب أن يكون 30 حرف!
                status = "WARN"
                notes.append(f"Suspiciously long terms")

            if status == "OK":
                good_quality += 1
            else:
                bad_quality += 1

            print(f"  [{i+1}] {drug:<20} | Count: {count:<3} | Status: {status}")
            print(f"       Terms: {', '.join(terms[:5])} ...")
            if notes:
                print(f"       Notes: {', '.join(notes)}")
            print()

    print("-" * 70)
    print(f"  Quality Estimate: {good_quality}/{sample_size} good entries in sample.")
    print("-" * 70)
    
    print("\n" + "=" * 70)
    print("  CONCLUSION")
    print("=" * 70)
    if coverage == 100.0:
        print("  ✅ CACHE IS COMPLETE.")
    else:
        print(f"  ⚠️  CACHE IS INCOMPLETE. Need to expand {len(missing_drugs)} drugs.")
    
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()