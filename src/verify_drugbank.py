"""
verify_drugbank.py
==================
يتحقق أن ملف drugbank_vocabulary.csv يحتوي على أسماء الأدوية
المقابلة لجميع الـ IDs الموجودة في dataset المedHop.

Usage:
    py -3.10 verify_drugbank.py
"""

import json
import csv
import os
import sys

# ─────────────────────────────────────────────
# PATHS — عدّل هاي المسارات حسب جهازك
# ─────────────────────────────────────────────

MEDHOP_FILE    = "data/medhop.json"      # ملف الداتا المعالج
DRUGBANK_CSV   = "data/drugbank_all_drugbank_vocabulary.csv/drugbank vocabulary.csv"

# إذا ما عندك medhop.json جاهز بعد، استخدم الملف الأصلي مباشرة:
MEDHOP_RAW_DEV = "data/qangaroo_v1.1/qangaroo_v1.1/medhop/dev.json"


# ─────────────────────────────────────────────
# STEP 1 — تحميل الـ IDs من MedHop
# ─────────────────────────────────────────────

def load_medhop_ids(filepath: str) -> set:
    """اجمع كل الـ Drug IDs الموجودة في ملف MedHop."""
    if not os.path.exists(filepath):
        print(f"  [FAIL] الملف مو موجود: {filepath}")
        return set()

    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    ids = set()
    for record in data:
        # query drug
        q = record.get("query", "")
        # استخرج IDs من الـ query (صيغة: "DB00001 interacts with")
        # والـ candidates
        candidates = record.get("candidates", [])
        answer     = record.get("answer", "")

        for cid in candidates:
            if cid.startswith("DB"):
                ids.add(cid)
        if answer.startswith("DB"):
            ids.add(answer)

        # استخرج الـ query drug ID (الكلمة الأولى من الـ query)
        parts = q.split()
        if parts and parts[0].startswith("DB"):
            ids.add(parts[0])

    print(f"  [OK] وجدت {len(ids)} Drug ID مختلف في MedHop")
    return ids


# ─────────────────────────────────────────────
# STEP 2 — تحميل الـ Vocabulary من DrugBank CSV
# ─────────────────────────────────────────────

def load_drugbank_vocab(filepath: str) -> dict:
    """
    حمّل ملف DrugBank CSV وارجع dict:
      { "DB00001": "Lepirudin", ... }
    """
    if not os.path.exists(filepath):
        print(f"  [FAIL] ملف DrugBank مو موجود: {filepath}")
        return {}

    vocab = {}

    with open(filepath, encoding="utf-8", newline="") as f:
        # اكتشف الـ delimiter تلقائياً
        sample = f.read(2048)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        reader = csv.DictReader(f, dialect=dialect)

        print(f"\n  أعمدة الـ CSV: {reader.fieldnames}")

        for row in reader:
            # DrugBank CSV الرسمي فيه أعمدة:
            # DrugBank ID, Common name, CAS, UNII, Standard InChI Key, ...
            drug_id   = None
            drug_name = None

            # حاول إيجاد الـ ID والـ Name من الأعمدة المحتملة
            for id_col in ["DrugBank ID", "drugbank_id", "ID", "id"]:
                if id_col in row and row[id_col]:
                    drug_id = row[id_col].strip()
                    break

            for name_col in ["Common name", "Name", "name", "drug_name", "Generic Name"]:
                if name_col in row and row[name_col]:
                    drug_name = row[name_col].strip()
                    break

            if drug_id and drug_name:
                vocab[drug_id] = drug_name

    print(f"  [OK] حمّلت {len(vocab)} دواء من DrugBank CSV")
    return vocab


# ─────────────────────────────────────────────
# STEP 3 — المقارنة والتحقق
# ─────────────────────────────────────────────

def verify(medhop_ids: set, vocab: dict):
    """قارن بين الـ IDs في MedHop والأسماء في DrugBank."""

    print("\n" + "=" * 60)
    print("  نتائج التحقق")
    print("=" * 60)

    found     = {did: vocab[did] for did in medhop_ids if did in vocab}
    not_found = {did for did in medhop_ids if did not in vocab}

    coverage = len(found) / len(medhop_ids) * 100 if medhop_ids else 0

    print(f"\n  إجمالي الـ IDs في MedHop   : {len(medhop_ids)}")
    print(f"  موجودة في DrugBank CSV    : {len(found)}")
    print(f"  غير موجودة (missing)      : {len(not_found)}")
    print(f"  نسبة التغطية              : {coverage:.1f}%")

    # ─── عينة من الأسماء الموجودة ───
    print("\n  ✅ عينة من الأسماء المُعثور عليها:")
    for did, name in list(found.items())[:10]:
        print(f"     {did}  →  {name}")

    # ─── الـ IDs الناقصة ───
    if not_found:
        print(f"\n  ❌ الـ IDs غير الموجودة ({len(not_found)}):")
        for did in sorted(not_found)[:20]:
            print(f"     {did}")
        if len(not_found) > 20:
            print(f"     ... و {len(not_found) - 20} أخرى")
    else:
        print("\n  ✅ ممتاز! جميع الـ IDs موجودة في DrugBank CSV")

    print("\n" + "=" * 60)

    # ─── تقييم ───
    if coverage == 100:
        print("  ✅ الملف مثالي — يغطي جميع الأدوية في MedHop")
    elif coverage >= 90:
        print("  ✅ الملف جيد جداً — التغطية عالية")
        print("  ⚠️  الأسماء الناقصة ستبقى كـ ID في الـ prompt")
    elif coverage >= 70:
        print("  ⚠️  تغطية متوسطة — قد يؤثر على جودة الـ prompts")
    else:
        print("  ❌ تغطية ضعيفة — تحقق من صحة الملف أو نسخته")

    print("=" * 60 + "\n")

    return found, not_found


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  DrugBank Vocabulary — التحقق من التغطية")
    print("=" * 60)

    # اختر الملف المناسب
    if os.path.exists(MEDHOP_FILE):
        medhop_path = MEDHOP_FILE
        print(f"\n  [INFO] استخدم medhop.json المعالج: {medhop_path}")
    elif os.path.exists(MEDHOP_RAW_DEV):
        medhop_path = MEDHOP_RAW_DEV
        print(f"\n  [INFO] استخدم dev.json الأصلي: {medhop_path}")
    else:
        print(f"\n  [FAIL] ما لقيت أي ملف MedHop!")
        print(f"         تأكد من المسار: {MEDHOP_FILE}")
        sys.exit(1)

    print(f"  [INFO] ملف DrugBank: {DRUGBANK_CSV}\n")

    # Step 1
    print("--- Step 1: تحميل IDs من MedHop ---")
    medhop_ids = load_medhop_ids(medhop_path)
    if not medhop_ids:
        sys.exit(1)

    # Step 2
    print("\n--- Step 2: تحميل DrugBank Vocabulary ---")
    vocab = load_drugbank_vocab(DRUGBANK_CSV)
    if not vocab:
        sys.exit(1)

    # Step 3
    print("\n--- Step 3: المقارنة ---")
    found, not_found = verify(medhop_ids, vocab)


if __name__ == "__main__":
    main()