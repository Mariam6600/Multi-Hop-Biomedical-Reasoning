"""
visualize_results.py
====================
إنشاء تصورات بصرية شاملة لنتائج Meta-Classifier

المخططات المتاحة:
1. Recall Curves (منحنيات Recall@K)
2. Confusion Matrix (مصفوفة التشويش)
3. Classification Report (تقرير التصنيف)
4. Feature Importance (أهمية الميزات)
5. Model Comparison (مقارنة المصنفات)
6. Path Contribution (مساهمة كل Path)
7. AUC-ROC Curve (منحنى ROC)
8. Error Analysis (تحليل الأخطاء)

الاستخدام:
  python visualize_results.py --all
  python visualize_results.py --recall-curves
  python visualize_results.py --confusion-matrix
"""

import json
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import OUTPUTS_DIR

# تعيين الخطوط العربية (اختياري)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 100
sns.set_style("whitegrid")

# ══════════════════════════════════════════════════════════════════════════════
# 1. RECALL CURVES
# ══════════════════════════════════════════════════════════════════════════════

def plot_recall_curves():
    """رسم منحنيات Recall@K لكل المصنفات."""
    print("\n[1/8] Plotting Recall Curves...")
    
    # تحميل النتائج
    summary_file = os.path.join(OUTPUTS_DIR, "meta_classifier_v4f_summary_CORRECTED.json")
    with open(summary_file, encoding="utf-8") as f:
        data = json.load(f)
    
    # اختيار أفضل 5 مصنفات
    results = sorted(data["results"], key=lambda x: x["dev_em"], reverse=True)[:5]
    
    plt.figure(figsize=(12, 7))
    
    for result in results:
        recall = result["dev_recall"]
        k_values = sorted([int(k.split("@")[1]) for k in recall.keys()])
        recall_values = [recall[f"recall@{k}"] for k in k_values]
        
        label = f"{result['classifier']} ({result['feature_set']}) - EM: {result['dev_em']:.1f}%"
        plt.plot(k_values, recall_values, marker='o', linewidth=2, label=label)
    
    plt.xlabel('K (Top-K Candidates)', fontsize=14, fontweight='bold')
    plt.ylabel('Recall@K (%)', fontsize=14, fontweight='bold')
    plt.title('Recall@K Curves - Top 5 Classifiers', fontsize=16, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, 10))
    plt.ylim(0, 105)
    
    output_file = os.path.join(OUTPUTS_DIR, "visualization_recall_curves.png")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Saved: {output_file}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. CONFUSION MATRIX
# ══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix():
    """رسم مصفوفة التشويش (Correct vs Incorrect)."""
    print("\n[2/8] Plotting Confusion Matrix...")
    
    # تحميل predictions
    pred_file = os.path.join(OUTPUTS_DIR, "meta_classifier_v4f_S4_logistic_regression_CORRECTED_predictions.json")
    
    if not os.path.exists(pred_file):
        print(f"  ⚠️ File not found: {pred_file}")
        return
    
    with open(pred_file, encoding="utf-8") as f:
        predictions = json.load(f)
    
    # حساب Correct/Incorrect
    correct = sum(1 for p in predictions if p.get("is_correct", False))
    incorrect = len(predictions) - correct
    
    # مصفوفة 2x2
    cm = np.array([[correct, 0], [incorrect, 0]])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted', ''], 
                yticklabels=['Correct', 'Incorrect'],
                cbar_kws={'label': 'Count'})
    
    plt.title(f'Prediction Results\nEM: {correct/len(predictions)*100:.2f}%', 
              fontsize=16, fontweight='bold')
    plt.ylabel('Actual', fontsize=14, fontweight='bold')
    plt.xlabel('', fontsize=14)
    
    output_file = os.path.join(OUTPUTS_DIR, "visualization_confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Saved: {output_file}")

# ══════════════════════════════════════════════════════════════════════════════
# 3. CLASSIFICATION REPORT
# ══════════════════════════════════════════════════════════════════════════════

def generate_classification_report():
    """إنشاء تقرير تصنيف نصي."""
    print("\n[3/8] Generating Classification Report...")
    
    pred_file = os.path.join(OUTPUTS_DIR, "meta_classifier_v4f_S4_logistic_regression_CORRECTED_predictions.json")
    summary_file = os.path.join(OUTPUTS_DIR, "meta_classifier_v4f_summary_CORRECTED.json")
    
    if not os.path.exists(pred_file):
        print(f"  ⚠️ File not found: {pred_file}")
        return
    
    with open(pred_file, encoding="utf-8") as f:
        predictions = json.load(f)
    
    # حساب المقاييس
    total = len(predictions)
    correct = sum(1 for p in predictions if p.get("is_correct", False))
    
    # Recall@K من ملف summary
    recall_at_k = {}
    if os.path.exists(summary_file):
        with open(summary_file, encoding="utf-8") as f:
            summary = json.load(f)
            best_result = summary["results"][0]  # أفضل نموذج
            for k in range(1, 10):
                recall_at_k[k] = best_result["dev_recall"].get(f"recall@{k}", 0.0)
    else:
        for k in range(1, 10):
            recall_at_k[k] = 0.0
    
    # كتابة التقرير
    report = f"""
════════════════════════════════════════════════════════════════════
  CLASSIFICATION REPORT
  Meta-Classifier: Logistic Regression + S4
════════════════════════════════════════════════════════════════════

Dataset Statistics:
  Total Questions: {total}
  Correct Predictions: {correct}
  Incorrect Predictions: {total - correct}

Performance Metrics:
  Exact Match (EM): {correct/total*100:.2f}%
  Accuracy: {correct/total*100:.2f}%
  Error Rate: {(total-correct)/total*100:.2f}%

Recall@K:
"""
    
    for k in range(1, 10):
        report += f"  Recall@{k}: {recall_at_k[k]:.2f}%\n"
    
    report += "\n════════════════════════════════════════════════════════════════════\n"
    
    output_file = os.path.join(OUTPUTS_DIR, "classification_report.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"  ✅ Saved: {output_file}")
    print(report)

# ══════════════════════════════════════════════════════════════════════════════
# 4. FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════

def plot_feature_importance():
    """رسم أهمية الميزات."""
    print("\n[4/8] Plotting Feature Importance...")
    
    feat_file = os.path.join(OUTPUTS_DIR, "meta_classifier_v4f_S4_logistic_regression_CORRECTED_feature_importance.json")
    
    if not os.path.exists(feat_file):
        print(f"  ⚠️ File not found: {feat_file}")
        return
    
    with open(feat_file, encoding="utf-8") as f:
        data = json.load(f)
    
    # أخذ أفضل 15 ميزة من all_features
    all_features = data.get("all_features", data.get("top_10_features", {}))
    sorted_features = sorted(all_features.items(), key=lambda x: abs(x[1]), reverse=True)[:15]
    
    names = [f[0] for f in sorted_features]
    importances = [abs(f[1]) for f in sorted_features]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(names)), importances, color='steelblue')
    plt.yticks(range(len(names)), names, fontsize=10)
    plt.xlabel('Importance', fontsize=12, fontweight='bold')
    plt.title('Top 15 Feature Importance\n(Logistic Regression + S4)', 
              fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    
    output_file = os.path.join(OUTPUTS_DIR, "visualization_feature_importance.png")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Saved: {output_file}")

# ══════════════════════════════════════════════════════════════════════════════
# 5. MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

def plot_model_comparison():
    """مقارنة أداء المصنفات المختلفة."""
    print("\n[5/8] Plotting Model Comparison...")
    
    summary_file = os.path.join(OUTPUTS_DIR, "meta_classifier_v4f_summary_CORRECTED.json")
    with open(summary_file, encoding="utf-8") as f:
        data = json.load(f)
    
    # اختيار أفضل 8 مصنفات
    results = sorted(data["results"], key=lambda x: x["dev_em"], reverse=True)[:8]
    
    labels = [f"{r['classifier']}\n({r['feature_set']})" for r in results]
    em_values = [r["dev_em"] for r in results]
    auc_values = [r["val_auc"] * 100 for r in results]  # تحويل إلى نسبة مئوية
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 7))
    bars1 = ax.bar(x - width/2, em_values, width, label='DEV EM (%)', color='steelblue')
    bars2 = ax.bar(x + width/2, auc_values, width, label='Val AUC (%)', color='coral')
    
    ax.set_xlabel('Classifier', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('Model Comparison: Top 8 Classifiers', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # إضافة القيم فوق الأعمدة
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=8)
    
    output_file = os.path.join(OUTPUTS_DIR, "visualization_model_comparison.png")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Saved: {output_file}")

# ══════════════════════════════════════════════════════════════════════════════
# 6. PATH CONTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════

def plot_path_contribution():
    """رسم مساهمة كل Path في النتيجة النهائية."""
    print("\n[6/8] Plotting Path Contribution...")
    
    # Path performance على DEV
    path_em = {
        "Path A\n(BM25 + biomistral-7b)": 16.96,
        "Path B\n(MedCPT + qwen2.5-7b)": 19.59,
        "Path C\n(Hybrid + qwen3.5-9b)": 16.08,
        "Meta-Classifier\n(Ensemble)": 51.17
    }
    
    labels = list(path_em.keys())
    values = list(path_em.values())
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bar chart
    bars = ax1.bar(range(len(labels)), values, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_ylabel('Exact Match (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Individual Path vs Meta-Classifier Performance', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # إضافة القيم
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Pie chart للـ Paths فقط
    path_values = values[:3]
    path_labels = [l.split('\n')[0] for l in labels[:3]]
    ax2.pie(path_values, labels=path_labels, autopct='%1.1f%%', 
            colors=colors[:3], startangle=90, textprops={'fontsize': 11})
    ax2.set_title('Individual Path Contribution', fontsize=14, fontweight='bold')
    
    output_file = os.path.join(OUTPUTS_DIR, "visualization_path_contribution.png")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Saved: {output_file}")

# ══════════════════════════════════════════════════════════════════════════════
# 7. AUC-ROC CURVE (تقريبي)
# ══════════════════════════════════════════════════════════════════════════════

def plot_auc_comparison():
    """مقارنة AUC للمصنفات المختلفة."""
    print("\n[7/8] Plotting AUC Comparison...")
    
    summary_file = os.path.join(OUTPUTS_DIR, "meta_classifier_v4f_summary_CORRECTED.json")
    with open(summary_file, encoding="utf-8") as f:
        data = json.load(f)
    
    # اختيار أفضل 6 مصنفات
    results = sorted(data["results"], key=lambda x: x["val_auc"], reverse=True)[:6]
    
    labels = [f"{r['classifier']} ({r['feature_set']})" for r in results]
    auc_values = [r["val_auc"] for r in results]
    
    plt.figure(figsize=(10, 6))
    bars = plt.barh(range(len(labels)), auc_values, color='mediumseagreen', edgecolor='black')
    plt.yticks(range(len(labels)), labels, fontsize=10)
    plt.xlabel('Validation AUC', fontsize=12, fontweight='bold')
    plt.title('Validation AUC Comparison - Top 6 Classifiers', fontsize=14, fontweight='bold')
    plt.xlim(0.5, 0.85)
    plt.grid(axis='x', alpha=0.3)
    
    # إضافة القيم
    for i, (bar, val) in enumerate(zip(bars, auc_values)):
        plt.text(val + 0.01, i, f'{val:.4f}', va='center', fontsize=10, fontweight='bold')
    
    output_file = os.path.join(OUTPUTS_DIR, "visualization_auc_comparison.png")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Saved: {output_file}")

# ══════════════════════════════════════════════════════════════════════════════
# 8. ERROR ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def plot_error_analysis():
    """تحليل الأخطاء: أين يفشل المصنف؟"""
    print("\n[8/8] Plotting Error Analysis...")
    
    pred_file = os.path.join(OUTPUTS_DIR, "meta_classifier_v4f_S4_logistic_regression_CORRECTED_predictions.json")
    
    if not os.path.exists(pred_file):
        print(f"  ⚠️ File not found: {pred_file}")
        return
    
    with open(pred_file, encoding="utf-8") as f:
        predictions = json.load(f)
    
    # تحليل: في أي K تظهر الإجابة الصحيحة؟
    answer_positions = []
    for p in predictions:
        ranked = p.get("ranked_candidates", [])
        answer = p.get("answer", "").upper()
        try:
            pos = ranked.index(answer) + 1  # 1-indexed
            answer_positions.append(pos)
        except ValueError:
            answer_positions.append(10)  # لم يُعثر عليها
    
    # Histogram
    plt.figure(figsize=(12, 6))
    bins = range(1, 12)
    counts, _, _ = plt.hist(answer_positions, bins=bins, color='coral', edgecolor='black', alpha=0.7)
    
    plt.xlabel('Answer Position in Ranked List', fontsize=12, fontweight='bold')
    plt.ylabel('Frequency', fontsize=12, fontweight='bold')
    plt.title('Error Analysis: Where Does the Correct Answer Appear?', fontsize=14, fontweight='bold')
    plt.xticks(range(1, 11), [str(i) if i < 10 else '10+' for i in range(1, 11)])
    plt.grid(axis='y', alpha=0.3)
    
    # إضافة النسب المئوية
    total = len(answer_positions)
    for i, count in enumerate(counts):
        if count > 0:
            pct = count / total * 100
            plt.text(i + 1.5, count, f'{pct:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    output_file = os.path.join(OUTPUTS_DIR, "visualization_error_analysis.png")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Saved: {output_file}")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Generate visualizations for Meta-Classifier results")
    parser.add_argument("--all", action="store_true", help="Generate all visualizations")
    parser.add_argument("--recall-curves", action="store_true", help="Plot Recall@K curves")
    parser.add_argument("--confusion-matrix", action="store_true", help="Plot confusion matrix")
    parser.add_argument("--classification-report", action="store_true", help="Generate classification report")
    parser.add_argument("--feature-importance", action="store_true", help="Plot feature importance")
    parser.add_argument("--model-comparison", action="store_true", help="Plot model comparison")
    parser.add_argument("--path-contribution", action="store_true", help="Plot path contribution")
    parser.add_argument("--auc-comparison", action="store_true", help="Plot AUC comparison")
    parser.add_argument("--error-analysis", action="store_true", help="Plot error analysis")
    
    args = parser.parse_args()
    
    print("\n" + "=" * 68)
    print("  Meta-Classifier Results Visualization")
    print("=" * 68)
    
    if args.all or not any(vars(args).values()):
        # Generate all visualizations
        plot_recall_curves()
        plot_confusion_matrix()
        generate_classification_report()
        plot_feature_importance()
        plot_model_comparison()
        plot_path_contribution()
        plot_auc_comparison()
        plot_error_analysis()
    else:
        if args.recall_curves:
            plot_recall_curves()
        if args.confusion_matrix:
            plot_confusion_matrix()
        if args.classification_report:
            generate_classification_report()
        if args.feature_importance:
            plot_feature_importance()
        if args.model_comparison:
            plot_model_comparison()
        if args.path_contribution:
            plot_path_contribution()
        if args.auc_comparison:
            plot_auc_comparison()
        if args.error_analysis:
            plot_error_analysis()
    
    print("\n" + "=" * 68)
    print("  ✅ All visualizations generated successfully!")
    print(f"  📁 Output directory: {OUTPUTS_DIR}")
    print("=" * 68 + "\n")

if __name__ == "__main__":
    main()
