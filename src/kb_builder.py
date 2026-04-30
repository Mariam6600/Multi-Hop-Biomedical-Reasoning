"""
src/kb_builder.py
==================
بناء قاعدة المعرفة الهيكلية من مصدرين:
  1. DrugBank proteins.tsv   → علاقات دواء–بروتين (targets, enzymes, transporters, carriers)
  2. Hetionet edges.sif.gz   → علاقات دواء–دواء (CrC resembles, PCiC pharmacosim)
                              + علاقات دواء–جين  (CbG, CdG, CuG)
  3. similarity-slim.tsv     → تشابه كيميائي بين الأدوية

المخرج:
  data/knowledge_base.json   → قاموس: drug_id → {enzymes, targets, transporters,
                                                   carriers, actions, het_interactions,
                                                   chemical_similar}

الاستخدام:
    py -3.10 src/kb_builder.py              ← يبني ويحفظ
    from src.kb_builder import load_kb      ← يحمل في الذاكرة

ملاحظة: يحتاج الملفات في data/:
    data/proteins.tsv
    data/hetionet-v1.0-edges.sif.gz
    data/similarity-slim.tsv               (اختياري)
"""

import os, sys, json, gzip
import pandas as pd
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import OUTPUTS_DIR

# ─── مسارات الملفات ──────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR      = _PROJECT_ROOT / "data"

PROTEINS_TSV   = DATA_DIR / "proteins.tsv"
HETIONET_GZ    = DATA_DIR / "hetionet-v1.0-edges.sif.gz"
SIMILARITY_TSV = DATA_DIR / "similarity-slim.tsv"
KB_OUTPUT      = DATA_DIR / "knowledge_base.json"

# علاقات Hetionet المفيدة لمهمتنا
METAEDGE_MAP = {
    "CrC":   "resembles",           # chemical similarity (structural)
    "PCiC":  "pharmacosim",         # pharmacologically similar
    "CbG":   "binds_gene",          # compound binds gene/protein
    "CdG":   "downregulates_gene",  # compound downregulates gene
    "CuG":   "upregulates_gene",    # compound upregulates gene
    "CtD":   "treats_disease",      # compound treats disease (indirect signal)
}

# حد أدنى للتشابه الكيميائي (نأخذ فقط ≥ 0.4 لتقليل حجم الملف)
SIM_THRESHOLD = 0.40


def _parse_het_id(x: str) -> str:
    """'Compound::DB01234' → 'DB01234'"""
    return x.split("::")[-1] if "::" in x else x


def build_kb(
    proteins_path: Path  = PROTEINS_TSV,
    hetionet_path: Path  = HETIONET_GZ,
    sim_path:      Path  = SIMILARITY_TSV,
    output_path:   Path  = KB_OUTPUT,
    verbose: bool        = True,
) -> dict:
    """
    يبني قاعدة المعرفة من الملفات المتاحة ويحفظها.
    يُعيد dict جاهز للاستخدام.
    """
    kb = defaultdict(lambda: {
        "targets":      [],
        "enzymes":      [],
        "transporters": [],
        "carriers":     [],
        "actions":      {},       # uniprot_id → action string
        "het_interactions": [],   # [{"partner": "DB...", "relation": "..."}]
        "chemical_similar": [],   # [{"partner": "DB...", "sim": 0.xx}]
    })

    stats = {"proteins": 0, "het_edges": 0, "sim_pairs": 0, "drugs": set()}

    # ─── 1. DrugBank proteins.tsv ────────────────────────────────────────────
    if proteins_path.exists():
        if verbose: print(f"  [1/3] Loading DrugBank proteins: {proteins_path.name}")
        df = pd.read_csv(proteins_path, sep="\t", low_memory=False)
        for _, row in df.iterrows():
            drug    = str(row.get("drugbank_id", "")).strip()
            cat     = str(row.get("category", "")).lower().strip()
            uniprot = str(row.get("uniprot_id", "")).strip()
            action  = str(row.get("actions", "")).strip()
            if not drug or not uniprot or uniprot.lower() == "nan":
                continue
            stats["proteins"] += 1
            stats["drugs"].add(drug)

            if cat == "target"      and uniprot not in kb[drug]["targets"]:
                kb[drug]["targets"].append(uniprot)
            elif cat == "enzyme"    and uniprot not in kb[drug]["enzymes"]:
                kb[drug]["enzymes"].append(uniprot)
            elif cat == "transporter" and uniprot not in kb[drug]["transporters"]:
                kb[drug]["transporters"].append(uniprot)
            elif cat == "carrier"   and uniprot not in kb[drug]["carriers"]:
                kb[drug]["carriers"].append(uniprot)

            if action and action.lower() != "nan":
                kb[drug]["actions"][uniprot] = action

        if verbose: print(f"      → {stats['proteins']:,} rows | {len(stats['drugs']):,} drugs")
    else:
        if verbose: print(f"  [WARN] proteins.tsv not found: {proteins_path}")

    # ─── 2. Hetionet edges ───────────────────────────────────────────────────
    if hetionet_path.exists():
        if verbose: print(f"  [2/3] Loading Hetionet edges: {hetionet_path.name}")
        opener = gzip.open if str(hetionet_path).endswith(".gz") else open
        with opener(hetionet_path, "rt", encoding="utf-8", errors="ignore") as f:
            next(f)   # header
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                src, edge, tgt = parts[0], parts[1], parts[2]
                if edge not in METAEDGE_MAP:
                    continue

                s = _parse_het_id(src)
                t = _parse_het_id(tgt)
                rel = METAEDGE_MAP[edge]

                # علاقات دواء–دواء (ثنائية الاتجاه)
                if edge in ("CrC", "PCiC"):
                    if s.startswith("DB") and t.startswith("DB"):
                        kb[s]["het_interactions"].append({"partner": t, "relation": rel})
                        kb[t]["het_interactions"].append({"partner": s, "relation": rel})
                        stats["het_edges"] += 1
                # علاقات دواء–جين (أحادية)
                elif "Compound" in src and s.startswith("DB"):
                    kb[s]["het_interactions"].append({"partner": t, "relation": rel})
                    stats["het_edges"] += 1

        if verbose: print(f"      → {stats['het_edges']:,} compound edges")
    else:
        if verbose: print(f"  [WARN] Hetionet file not found: {hetionet_path}")

    # ─── 3. Chemical similarity ──────────────────────────────────────────────
    if sim_path.exists():
        if verbose: print(f"  [3/3] Loading chemical similarity: {sim_path.name}")
        try:
            for chunk in pd.read_csv(sim_path, sep="\t", chunksize=500_000):
                for _, row in chunk.iterrows():
                    c0  = str(row.get("compound0", "")).strip()
                    c1  = str(row.get("compound1", "")).strip()
                    sim = float(row.get("similarity", 0))
                    if sim >= SIM_THRESHOLD and c0.startswith("DB") and c1.startswith("DB"):
                        kb[c0]["chemical_similar"].append({"partner": c1, "sim": round(sim, 3)})
                        kb[c1]["chemical_similar"].append({"partner": c0, "sim": round(sim, 3)})
                        stats["sim_pairs"] += 1
        except Exception as e:
            if verbose: print(f"      [WARN] similarity error: {e}")
        if verbose: print(f"      → {stats['sim_pairs']:,} pairs ≥ {SIM_THRESHOLD}")
    else:
        if verbose: print(f"  [INFO] similarity-slim.tsv not found — skipping")

    # ─── تحويل sets إلى lists وحفظ ──────────────────────────────────────────
    kb_final = {k: dict(v) for k, v in kb.items()}

    if verbose:
        print(f"\n  [KB] Total drugs indexed: {len(kb_final):,}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(kb_final, f, ensure_ascii=False, separators=(",", ":"))
    if verbose:
        size_mb = output_path.stat().st_size / 1024 / 1024
        print(f"  [KB] Saved → {output_path} ({size_mb:.1f} MB)")

    return kb_final


def load_kb(kb_path: Path = KB_OUTPUT, rebuild_if_missing: bool = True) -> dict:
    """
    يحمّل قاعدة المعرفة من الملف.
    إذا لم يوجد الملف وrebuild_if_missing=True → يبني أولاً.
    """
    if not kb_path.exists():
        if rebuild_if_missing:
            print("  [KB] knowledge_base.json not found → building now...")
            return build_kb()
        else:
            print(f"  [WARN] KB not found: {kb_path}")
            return {}
    with open(kb_path, encoding="utf-8") as f:
        return json.load(f)


def score_candidate(query_drug: str, candidate: str, kb: dict) -> dict:
    """
    يحسب درجة احتمال التفاعل بين query_drug و candidate.
    يُعيد dict مع الدرجة وأسباب التفاعل.

    الدرجات:
      - إنزيم مشترك (CYP450):    8.0 نقطة لكل إنزيم
      - ناقل مشترك:               5.0 نقطة لكل ناقل
      - هدف مشترك:                2.0 نقطة لكل هدف
      - Hetionet resembles:        4.0 نقطة
      - Hetionet pharmacosim:      3.0 نقطة
      - تشابه كيميائي ≥ 0.6:      6.0 نقطة
      - تشابه كيميائي 0.4–0.6:    3.0 نقطة
    """
    q_info = kb.get(query_drug, {})
    c_info = kb.get(candidate, {})

    q_enz  = set(q_info.get("enzymes", []))
    c_enz  = set(c_info.get("enzymes", []))
    q_trp  = set(q_info.get("transporters", []))
    c_trp  = set(c_info.get("transporters", []))
    q_tgt  = set(q_info.get("targets", []))
    c_tgt  = set(c_info.get("targets", []))

    shared_enz = q_enz & c_enz
    shared_trp = q_trp & c_trp
    shared_tgt = q_tgt & c_tgt

    score   = 0.0
    reasons = []

    # إنزيمات مشتركة (أقوى إشارة — CYP450 competition)
    if shared_enz:
        score += len(shared_enz) * 8.0
        enz_actions = []
        for e in shared_enz:
            qa = q_info.get("actions", {}).get(e, "")
            ca = c_info.get("actions", {}).get(e, "")
            enz_actions.append(f"{e}(Q:{qa or '?'},C:{ca or '?'})")
        reasons.append(f"Shared enzyme(s): {', '.join(enz_actions)}")

    # نواقل مشتركة
    if shared_trp:
        score += len(shared_trp) * 5.0
        reasons.append(f"Shared transporter(s): {', '.join(shared_trp)}")

    # أهداف مشتركة (إشارة أضعف)
    if shared_tgt:
        score += len(shared_tgt) * 2.0
        reasons.append(f"Shared target(s): {', '.join(list(shared_tgt)[:3])}")

    # Hetionet interactions
    het_rels = [
        i["relation"] for i in q_info.get("het_interactions", [])
        if i.get("partner") == candidate
    ]
    for rel in het_rels:
        if rel == "resembles":
            score += 4.0
            reasons.append("Hetionet: structurally resembles")
        elif rel == "pharmacosim":
            score += 3.0
            reasons.append("Hetionet: pharmacologically similar")

    # تشابه كيميائي
    sim_entries = [
        e for e in q_info.get("chemical_similar", [])
        if e.get("partner") == candidate
    ]
    for se in sim_entries:
        s = se.get("sim", 0)
        if s >= 0.6:
            score += 6.0
            reasons.append(f"High chemical similarity ({s:.2f})")
        else:
            score += 3.0
            reasons.append(f"Moderate chemical similarity ({s:.2f})")

    return {
        "score":    round(score, 2),
        "reasons":  reasons,
        "shared_enzymes":      list(shared_enz),
        "shared_transporters": list(shared_trp),
        "shared_targets":      list(shared_tgt),
    }


def get_kb_evidence_text(query_drug: str, candidates: list, kb: dict, top_n: int = 3) -> tuple:
    """
    يحسب درجة كل مرشح ويُعيد:
      1. قائمة المرشحين مرتبة حسب الدرجة (top_n)
      2. نص الأدلة الهيكلية للـ prompt
    """
    scored = []
    for c in candidates:
        info = score_candidate(query_drug, c, kb)
        scored.append((c, info))

    scored.sort(key=lambda x: -x[1]["score"])
    top_candidates = [c for c, _ in scored[:top_n]]

    lines = ["[Structural Knowledge Base Evidence]"]
    for c, info in scored[:top_n]:
        if info["reasons"]:
            lines.append(f"  {c}: {' | '.join(info['reasons'][:3])}")
        else:
            lines.append(f"  {c}: No structural overlap found in KB")

    # تحذير: بقية المرشحين بدون أدلة هيكلية قوية
    others = [c for c, info in scored[top_n:] if info["score"] == 0]
    if others:
        lines.append(f"  Others ({', '.join(others[:4])}): No KB evidence → less likely")

    return top_candidates, "\n".join(lines)


# ─── تشغيل مباشر ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  KB Builder — Building knowledge_base.json")
    print("="*60)
    kb = build_kb(verbose=True)
    print(f"\n  Sample — DB00001 enzymes: {kb.get('DB00001', {}).get('enzymes', [])}")
    print(f"  Sample — DB00001 targets: {kb.get('DB00001', {}).get('targets', [])[:3]}")
    print("\n  Done! Run inference_pipeline_kb.py to use the KB.")
