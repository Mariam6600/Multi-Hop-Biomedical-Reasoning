"""
analyze_results.py
تحليل نتائج Baseline 1 vs Baseline 2
"""

import json
import os

base_dir = r"C:\Users\LOQ\Desktop\Graduation Project2\Code Files\outputs"
b1_path = os.path.join(base_dir, "baseline1_predictions.json")
b2_path = os.path.join(base_dir, "baseline2_predictions.json")

print("=" * 60)
print("  تحليل نتائج Baseline 1 vs Baseline 2")
print("=" * 60)

# تحميل Baseline 1
with open(b1_path, encoding='utf-8') as f:
    b1 = json.load(f)

# تحميل Baseline 2
with open(b2_path, encoding='utf-8') as f:
    b2 = json.load(f)

# إحصائيات
b1_correct = sum(1 for p in b1 if p.get('is_correct', False))
b2_correct = sum(1 for p in b2 if p.get('is_correct', False))

print()
print("  المقارنة:")
print("  " + "-" * 40)
print(f"  Baseline 1 EM: {b1_correct}/{len(b1)} = {b1_correct/len(b1)*100:.2f}%")
print(f"  Baseline 2 EM: {b2_correct}/{len(b2)} = {b2_correct/len(b2)*100:.2f}%")
print()

# أسئلة تحسنت وتأزمت
improved = []
degraded = []

for p1, p2 in zip(b1, b2):
    qid = p1['question_id']
    if p1['is_correct'] != p2['is_correct']:
        if p2['is_correct']:
            improved.append(qid)
        else:
            degraded.append(qid)

print(f"  تحسنت: {len(improved)} سؤال")
print(f"  تأزمت: {len(degraded)} سؤال")
print()

# أمثلة إجابات صحيحة
print("  === أمثلة صحيحة في Baseline 2 ===")
correct_b2 = [p for p in b2 if p.get('is_correct', False)][:5]
for p in correct_b2:
    print(f"    {p['question_id']}: {p['query_drug_name']} -> {p['prediction']}")

print()
print("  === أمثلة خاطئة في Baseline 2 ===")
wrong_b2 = [p for p in b2 if not p.get('is_correct', False) and p.get('success', False)][:5]
for p in wrong_b2:
    print(f"    {p['question_id']}: {p['query_drug_name']} -> pred={p['prediction']}, ans={p['answer']}")

print()
print("=" * 60)