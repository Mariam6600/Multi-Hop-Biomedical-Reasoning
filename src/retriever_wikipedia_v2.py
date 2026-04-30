"""
src/retriever_wikipedia_v2.py
================================
Biomedical Multi-Hop QA — Enhanced Wikipedia Retriever V2

منهجيات مستوحاة من أفضل الأبحاث في BioCreative IX (MedHopQA 2025):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  1. Query2Doc (DMIS Lab — المركز الأول):
     بدل البحث بـ "اسم الدواء" فقط، نبني استعلام موسّع يدمج:
     - اسم الدواء + آلية العمل (bridge) + مصطلحات تفاعلية
     مثال: "Moclobemide" → "Moclobemide MAO-A inhibitor drug interactions metabolism"

  2. Sequential/Anchor Retrieval (UETQuintet — المركز الثاني):
     نستخدم آلية العمل كـ "مرساة" للبحث عن القفزة الثانية:
     - بحث أول: drug_name + mechanism
     - بحث ثاني: mechanism_entity + "drug interactions" + "substrates"

  3. Medical Section Targeting (مستوحى من Orekhovich):
     نستخرج أقسام محددة من مقالة ويكيبيديا:
     - "Interactions", "Pharmacology", "Mechanism of action"
     بدل الملخص العام فقط

  4. Entity Pair Search (مستوحى من NHSRAG):
     نبحث بأزواج: (Drug_A, Bridge_Entity) و(Bridge_Entity, Candidate)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

المقارنة مع retriever_wikipedia.py القديم:
─────────────────────────────────────────
  القديم: retrieve_wikipedia_docs("Moclobemide")
    → يبحث بالاسم فقط → ملخص عام → غالباً غير مفيد

  الجديد: retrieve_wikipedia_v2("Moclobemide", bridge="inhibits MAO-A")
    → بحث موسّع بالآلية → أقسام طبية محددة → نتائج أدق
    → بحث ثانوي بـ "MAO-A inhibitor interactions"

Usage:
    from src.retriever_wikipedia_v2 import retrieve_wikipedia_v2

    # مع bridge info (الأساس)
    docs = retrieve_wikipedia_v2(
        drug_name="Moclobemide",
        bridge_info="inhibits MAO-A",
        candidate_names=["Fluoxetine", "Selegiline"]
    )

    # بدون bridge (fallback)
    docs = retrieve_wikipedia_v2(drug_name="Moclobemide")
"""

import os, sys, json, re, time, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import OUTPUTS_DIR

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────────────

WIKI_V2_CACHE_FILE = os.path.join(OUTPUTS_DIR, "wikipedia_v2_cache.json")
WIKI_CHUNK_SIZE    = 400    # حجم كل chunk بالأحرف
WIKI_MAX_CHUNKS    = 12     # أقصى عدد chunks (أكثر من V1 لأننا نجلب من مصادر متعددة)
WIKI_RATE_DELAY    = 1.0    # ثواني بين الطلبات (زيادة لتجنب rate limiting)
WIKI_MAX_CHARS     = 6000   # أقصى عدد أحرف من كل مقالة كاملة
WIKI_BATCH_PAUSE   = 10     # استراحة كل N دواء (للـ prefetch)
WIKI_BATCH_DELAY   = 8      # ثواني الاستراحة بين كل batch
WIKI_MAX_RETRIES   = 5      # محاولات إعادة عند الفشل
WIKI_DRUG_DELAY    = 2      # ثواني بين كل دواء في الـ prefetch

# الأقسام الطبية المهمة التي نبحث عنها في مقالات ويكيبيديا
MEDICAL_SECTIONS = [
    "interactions",
    "pharmacology",
    "mechanism of action",
    "adverse effects",
    "contraindications",
    "metabolism",
    "side effects",
    "contraindications and warnings",
]


# ─────────────────────────────────────────────────────
# CACHE MANAGEMENT
# ─────────────────────────────────────────────────────

def _load_cache() -> dict:
    if os.path.exists(WIKI_V2_CACHE_FILE):
        try:
            with open(WIKI_V2_CACHE_FILE, encoding="utf-8") as f:
                return json.load(f)
        except:
            pass
    return {}


def _save_cache(cache: dict):
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    with open(WIKI_V2_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


# ─────────────────────────────────────────────────────
# WIKIPEDIA API FUNCTIONS
# ─────────────────────────────────────────────────────

def _wikipedia_search(query: str, limit: int = 3) -> list:
    """بحث في ويكيبيديا وإرجاع عناوين النتائج."""
    try:
        import urllib.request, urllib.parse
        safe = urllib.parse.quote(query)
        url = (f"https://en.wikipedia.org/w/api.php?action=query"
               f"&list=search&srsearch={safe}&format=json&srlimit={limit}")
        req = urllib.request.Request(
            url, headers={"User-Agent": "BiomedQA/2.0 (academic research)"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return [h["title"] for h in data.get("query", {}).get("search", [])]
    except Exception as e:
        logger.debug(f"Wikipedia search failed for '{query}': {e}")
        return []


def _get_page_summary(title: str) -> str:
    """يجلب ملخص مقالة ويكيبيديا."""
    try:
        import urllib.request, urllib.parse
        safe = urllib.parse.quote(title.replace(" ", "_"))
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{safe}"
        req = urllib.request.Request(
            url, headers={"User-Agent": "BiomedQA/2.0 (academic research)"}
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("extract", "")
    except:
        return ""


def _get_page_sections(title: str, target_sections: list = None) -> str:
    """
    يجلب أقسام محددة من مقالة ويكيبيديا.
    مستوحى من Orekhovich: لا نأخذ كل المقالة، بل الأقسام الطبية فقط.
    """
    if target_sections is None:
        target_sections = MEDICAL_SECTIONS

    try:
        import urllib.request, urllib.parse

        # نحمّل محتوى الصفحة مع أرقام الأقسام
        safe = urllib.parse.quote(title.replace(" ", "_"))
        url = (f"https://en.wikipedia.org/w/api.php?action=parse"
               f"&page={safe}&prop=sections&format=json")
        req = urllib.request.Request(
            url, headers={"User-Agent": "BiomedQA/2.0 (academic research)"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        sections = data.get("parse", {}).get("sections", [])
        if not sections:
            # fallback: نأخذ المقالة كاملة
            return _get_full_page_text(title)

        # نبحث عن الأقسام الطبية
        medical_section_indices = []
        for sec in sections:
            sec_name = sec.get("line", "").lower()
            sec_index = sec.get("index")
            if any(med in sec_name for med in target_sections):
                if sec_index is not None:
                    medical_section_indices.append(str(sec_index))

        if not medical_section_indices:
            # لا أقسام طبية موجودة → نأخذ الملخص + أول قسمين
            return _get_page_summary(title)

        # نجلب محتوى الأقسام الطبية
        result_parts = []
        for sec_idx in medical_section_indices[:4]:  # أقصى 4 أقسام
            try:
                sec_url = (f"https://en.wikipedia.org/w/api.php?action=parse"
                           f"&page={safe}&prop=text&section={sec_idx}&format=json")
                sec_req = urllib.request.Request(
                    sec_url, headers={"User-Agent": "BiomedQA/2.0 (academic research)"}
                )
                with urllib.request.urlopen(sec_req, timeout=8) as sec_resp:
                    sec_data = json.loads(sec_resp.read().decode("utf-8"))

                html_text = sec_data.get("parse", {}).get("text", {}).get("*", "")
                # تنظيف HTML
                clean = re.sub(r'<[^>]+>', ' ', html_text)
                clean = re.sub(r'\s+', ' ', clean).strip()
                if len(clean) > 50:
                    result_parts.append(clean[:1500])  # أقصى 1500 حرف لكل قسم
            except:
                continue
            time.sleep(0.1)  # rate limiting بين الأقسام

        return "\n\n".join(result_parts) if result_parts else _get_page_summary(title)

    except Exception as e:
        logger.debug(f"Section extraction failed for '{title}': {e}")
        return _get_page_summary(title)


def _get_full_page_text(title: str) -> str:
    """يجلب نص مقالة ويكيبيديا كاملاً (plain text)."""
    try:
        import urllib.request, urllib.parse
        safe = urllib.parse.quote(title.replace(" ", "_"))
        url = (f"https://en.wikipedia.org/w/api.php?action=query"
               f"&prop=extracts&exintro=false&explaintext=true"
               f"&titles={safe}&format=json")
        req = urllib.request.Request(
            url, headers={"User-Agent": "BiomedQA/2.0 (academic research)"}
        )
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            pages = data.get("query", {}).get("pages", {})
            for page_id, page in pages.items():
                if page_id != "-1":
                    return page.get("extract", "")
    except:
        pass
    return ""


# ─────────────────────────────────────────────────────
# CHUNKING (نفس V1 لكن مع metadata)
# ─────────────────────────────────────────────────────

def _chunk_text(text: str, source: str = "wikipedia",
                chunk_size: int = WIKI_CHUNK_SIZE,
                max_chunks: int = WIKI_MAX_CHUNKS) -> list:
    """
    يقطّع النص إلى chunks مع إضافة metadata.

    Returns: list[str] — كل عنصر chunk نصي
    """
    if not text or len(text) < 50:
        return []

    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = []
    current_len = 0

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if current_len + len(sent) > chunk_size and current:
            chunk = " ".join(current).strip()
            if len(chunk) > 50:
                chunks.append(chunk)
            current = [sent]
            current_len = len(sent)
        else:
            current.append(sent)
            current_len += len(sent) + 1

    if current:
        chunk = " ".join(current).strip()
        if len(chunk) > 50:
            chunks.append(chunk)

    return chunks[:max_chunks]


# ─────────────────────────────────────────────────────
# QUERY2DOC BUILDER (DMIS Lab approach)
# ─────────────────────────────────────────────────────

def build_query2doc(drug_name: str, bridge_info: str = "",
                    candidate_names: list = None) -> str:
    """
    يبني استعلام موسّع مستوحى من DMIS Lab (المركز الأول).

    الفكرة: بدل البحث بـ "Moclobemide" فقط، نبحث بـ:
    "Moclobemide MAO-A inhibitor drug interactions metabolism"

    هذا يزيد احتمال مطابقة المصطلحات الطبية في ويكيبيديا.
    """
    parts = [drug_name]

    if bridge_info:
        # نستخرج الكلمات المفتاحية من bridge
        # مثل "inhibits MAO-A" → "MAO-A inhibitor"
        bridge_clean = bridge_info.lower()
        # نستبدل أفعال التفاعل بمصطلحات بحثية
        for verb, noun in [("inhibits", "inhibitor"), ("blocks", "blocker"),
                           ("activates", "activator"), ("induces", "inducer"),
                           ("metabolized", "metabolism"), ("substrate", "substrate")]:
            if verb in bridge_clean:
                parts.append(noun)

        # نضيف كلمات الـ bridge نفسها (بدون أفعال شائعة)
        stop_words = {"inhibits", "blocks", "acts", "via", "through",
                      "by", "the", "a", "an", "is", "and", "or", "of"}
        bridge_words = [w for w in bridge_clean.split()
                       if w not in stop_words and len(w) > 2]
        parts.extend(bridge_words[:4])  # أقصى 4 كلمات إضافية

    # مصطلحات تفاعل عامة (تزيد من فرص إيجاد قسم Interactions)
    parts.append("drug interactions")

    return " ".join(parts)


def build_sequential_queries(drug_name: str, bridge_info: str = "",
                            candidate_names: list = None) -> list:
    """
    يبني استعلامات متسلسلة مستوحاة من UETQuintet (المركز الثاني).

    الفكرة: نستخدم آلية العمل كـ "مرساة" للبحث عن القفزة الثانية:
    - بحث أول: Drug_A + mechanism (لإيجاد كيف يتفاعل الدواء)
    - بحث ثاني: Bridge_Entity + interactions (لإيجاد أدوية أخرى بنفس المسار)

    Returns: list of (query_str, search_type) pairs
    """
    queries = []

    # بحث أول: الدواء + آلية العمل
    q1 = drug_name
    if bridge_info:
        # استخراج الكيان الجوهري من bridge
        bridge_clean = bridge_info.lower()
        stop_words = {"inhibits", "blocks", "acts", "via", "through", "by"}
        bridge_entities = [w for w in bridge_clean.split()
                          if w not in stop_words and len(w) > 2]
        if bridge_entities:
            q1 = f"{drug_name} {' '.join(bridge_entities[:2])} mechanism"
    queries.append((q1, "primary"))

    # بحث ثاني: كيان الـ bridge + تفاعلات
    if bridge_info:
        bridge_clean = bridge_info.lower()
        stop_words = {"inhibits", "blocks", "acts", "via", "through", "by"}
        bridge_entities = [w for w in bridge_clean.split()
                          if w not in stop_words and len(w) > 2]
        if bridge_entities:
            q2 = f"{' '.join(bridge_entities[:2])} drug interactions substrates"
            queries.append((q2, "anchor"))

    # بحث ثالث: أسماء المرشحين المحددة (إن وجدت)
    if candidate_names and len(candidate_names) > 0:
        # نبحث عن التفاعل بين الدواء وأول مرشحين (أقصى 2)
        for cand in candidate_names[:2]:
            q3 = f"{drug_name} {cand} interaction"
            queries.append((q3, "pair"))

    return queries


# ─────────────────────────────────────────────────────
# MAIN PUBLIC FUNCTION
# ─────────────────────────────────────────────────────

def retrieve_wikipedia_v2(
    drug_name: str,
    bridge_info: str = "",
    candidate_names: list = None,
    query: str = "",
    max_chunks: int = WIKI_MAX_CHUNKS,
    use_cache: bool = True,
    verbose: bool = False,
    methodology: str = "all",
) -> list:
    """
    Enhanced Wikipedia Retrieval V2 — بناءً على أفضل الأبحاث.

    methodology parameter controls which methods to apply (for A/B comparison):
      "all"             ← كل المنهجيات معاً (الافتراضي)
      "query2doc"       ← Query2Doc فقط (DMIS Lab)
      "sequential"      ← Sequential/Anchor Retrieval فقط (UETQuintet)
      "medical_sections" ← Medical Section Targeting فقط (Orekhovich)
      "entity_pair"     ← Entity Pair Search فقط (NHSRAG)

    المنهجية:
      1. Query2Doc: يبني استعلام موسّع من bridge_info
      2. Sequential Retrieval: يبحث بآلية العمل كمرساة
      3. Medical Section Targeting: يستخرج أقسام طبية محددة
      4. Entity Pair Search: يبحث بأزواج (Drug, Candidate)

    Args:
        drug_name:      اسم الدواء
        bridge_info:    آلية العمل (من bridge_cache)
        candidate_names: أسماء الأدوية المرشحة
        query:          نص السؤال الأصلي (للـ logging)
        max_chunks:     أقصى عدد chunks
        use_cache:      استخدام الـ cache
        verbose:        طباعة تفاصيل
        methodology:    أي منهجية نطبق (للمقارنة المنفصلة)

    Returns: list[str] — chunks من ويكيبيديا جاهزة للـ retriever
    """
    if not drug_name or not drug_name.strip():
        return []

    drug_key = drug_name.strip().lower()

    # ── Check cache ──
    # مفتاح الـ cache يشمل bridge_info + methodology لأن النتائج تعتمد عليها
    bridge_key = bridge_info[:50] if bridge_info else "no_bridge"
    cache_key = f"{drug_key}::{bridge_key}::{methodology}"

    if use_cache:
        cache = _load_cache()
        if cache_key in cache:
            cached = cache[cache_key]
            if verbose:
                print(f"  [WikiV2] Cache hit: {drug_name} ({len(cached)} chunks) [method={methodology}]")
            return cached[:max_chunks]
        # [FIXED v2] FALLBACK ::all REMOVED — كل منهجية تجلب بياناتها المستقلة


    all_chunks = []
    seen_texts = set()  # لتجنب التكرار

    def _add_chunks(text: str, source_tag: str = ""):
        """يضيف chunks جديدة مع تجنب التكرار."""
        if not text or len(text) < 50:
            return
        chunks = _chunk_text(text, source=source_tag, max_chunks=max_chunks)
        for c in chunks:
            # تجنب التكرار (أول 80 حرف كـ fingerprint)
            fp = c[:80].lower().strip()
            if fp not in seen_texts:
                seen_texts.add(fp)
                all_chunks.append(c)

    # ── الخطوة 1: بحث أساسي بالاسم ──
    # medical_sections: يركز على الأقسام الطبية فقط
    # all/sequential/query2doc/entity_pair: يبحث بالاسم أولاً ثم يضيف
    if methodology in ("all", "medical_sections"):
        if verbose:
            print(f"  [WikiV2] Step 1: Primary search for '{drug_name}' [method={methodology}]")

        # محاولة مباشرة بالاسم
        titles = _wikipedia_search(drug_name, limit=2)
        if not titles:
            titles = [drug_name]  # fallback

        for title in titles[:1]:  # نأخذ أفضل نتيجة فقط
            time.sleep(WIKI_RATE_DELAY)
            # نجرب أقسام طبية أولاً
            section_text = _get_page_sections(title)
            _add_chunks(section_text, source_tag=f"wiki_sections:{title}")

            # إذا الأقسام قليلة، نضيف الملخص
            if len(all_chunks) < 3:
                summary = _get_page_summary(title)
                _add_chunks(summary, source_tag=f"wiki_summary:{title}")
    else:
        # للمنهجيات الأخرى: نجلب ملخص بسيط بالاسم كأساس
        titles = []
        if verbose:
            print(f"  [WikiV2] Basic name search for '{drug_name}' [method={methodology}]")
        title_results = _wikipedia_search(drug_name, limit=1)
        if title_results:
            titles = title_results
            time.sleep(WIKI_RATE_DELAY)
            summary = _get_page_summary(titles[0])
            _add_chunks(summary, source_tag=f"wiki_summary:{titles[0]}")

    # ── الخطوة 2: Query2Doc — بحث موسّع بالآلية (DMIS Lab) ──
    if bridge_info and methodology in ("all", "query2doc"):
        if verbose:
            print(f"  [WikiV2] Step 2: Query2Doc search [method={methodology}]")

        q2d = build_query2doc(drug_name, bridge_info, candidate_names)
        time.sleep(WIKI_RATE_DELAY)
        q2d_titles = _wikipedia_search(q2d, limit=2)

        for title in q2d_titles[:1]:
            if title.lower() not in [t.lower() for t in titles]:
                time.sleep(WIKI_RATE_DELAY)
                section_text = _get_page_sections(title)
                _add_chunks(section_text, source_tag=f"wiki_q2d:{title}")

    # ── الخطوة 3: Sequential Retrieval — بحث بآلية العمل كمرساة (UETQuintet) ──
    if bridge_info and methodology in ("all", "sequential"):
        if verbose:
            print(f"  [WikiV2] Step 3: Sequential/Anchor search [method={methodology}]")

        seq_queries = build_sequential_queries(drug_name, bridge_info, candidate_names)

        for q_text, q_type in seq_queries[1:]:  # الأول نفسه يشبه Step 1+2
            time.sleep(WIKI_RATE_DELAY)
            seq_titles = _wikipedia_search(q_text, limit=1)
            for title in seq_titles[:1]:
                if title.lower() not in [t.lower() for t in titles]:
                    time.sleep(WIKI_RATE_DELAY)
                    page_text = _get_full_page_text(title)
                    _add_chunks(page_text[:3000], source_tag=f"wiki_seq_{q_type}:{title}")

    # ── الخطوة 4: Entity Pair Search — أزواج (Drug, Candidate) (NHSRAG) ──
    if candidate_names and len(candidate_names) > 0 and methodology in ("all", "entity_pair"):
        if verbose:
            print(f"  [WikiV2] Step 4: Entity pair search [method={methodology}]")

        # نبحث عن تفاعل الدواء مع أول مرشحين
        for cand in candidate_names[:2]:
            pair_query = f"{drug_name} {cand} interaction"
            time.sleep(WIKI_RATE_DELAY)
            pair_titles = _wikipedia_search(pair_query, limit=1)
            for title in pair_titles[:1]:
                time.sleep(WIKI_RATE_DELAY)
                page_text = _get_full_page_text(title)
                # نأخذ فقط الجزء الذي يذكر الدواء أو المرشح
                relevant = _extract_relevant_paragraphs(
                    page_text, drug_name, cand
                )
                _add_chunks(relevant, source_tag=f"wiki_pair:{drug_name}-{cand}")

    # ── Cache ──
    result = all_chunks[:max_chunks]

    # ★ FIX: لا تحفظ نتائج فارغة — خلّي retry يعمل
    if use_cache and result:  # فقط إذا فيه محتوى فعلي
        cache = _load_cache()
        cache[cache_key] = result
        _save_cache(cache)

    if verbose:
        print(f"  [WikiV2] Total: {len(result)} chunks for {drug_name}")

    return result


def _extract_relevant_paragraphs(text: str, drug_name: str,
                                  candidate_name: str) -> str:
    """يستخرج الفقرات التي تذكر الدواء أو المرشح من النص الكامل."""
    if not text:
        return ""

    paragraphs = text.split("\n\n")
    relevant = []
    drug_lower = drug_name.lower()
    cand_lower = candidate_name.lower()

    for para in paragraphs:
        para_lower = para.lower()
        if drug_lower in para_lower or cand_lower in para_lower:
            relevant.append(para.strip())
            if len(relevant) >= 3:  # أقصى 3 فقرات
                break

    return "\n\n".join(relevant) if relevant else text[:2000]


# ─────────────────────────────────────────────────────
# PREFETCH UTILITY
# ─────────────────────────────────────────────────────

def prefetch_wikipedia_v2(data: list, bridge_cache: dict = None,
                          methodology: str = "all",
                          verbose: bool = True) -> dict:
    """
    [FIXED v2] يحمّل مسبقاً ويكيبيديا V2 لكل الأدوية في الداتاسيت.

    التغييرات عن النسخة السابقة:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    1. يقبل الآن methodology parameter → كل تجربة تُجلب بياناتها المستقلة
    2. bridge_cache: يستخدم mechanism:: فقط (نفس baseline 33.33%)
       حُذف fallback لـ 2hop_mechanism:: الذي كان يُعطي bridges رديئة
    3. cache_key يشمل methodology → منهجية sequential ≠ query2doc في الكاش

    مثال:
      methodology="all"       → cache key: drug::bridge::all       (مُحمّل مسبقاً)
      methodology="sequential" → cache key: drug::bridge::sequential (يُجلب من الإنترنت)
      methodology="query2doc"  → cache key: drug::bridge::query2doc  (يُجلب من الإنترنت)

    Args:
        data:          قائمة من records (من medhop.json)
        bridge_cache:  bridge cache لتحسين الاستعلامات
        methodology:   المنهجية المطلوبة ("all"|"sequential"|"query2doc"|"medical_sections"|"entity_pair")
        verbose:       طباعة التقدم
    """
    cache      = _load_cache()
    drug_names = set()

    for rec in data:
        dn = rec.get("query_drug_name", "")
        if dn and dn.strip():
            # [FIXED] استخدم mechanism:: فقط — نفس baseline 33.33%
            bridge = ""
            if bridge_cache:
                key_v1 = f"mechanism::{dn}"
                raw    = bridge_cache.get(key_v1, "")
                if raw:
                    raw    = re.sub(r"MECHANISM:\s*", "", raw, flags=re.IGNORECASE).strip()
                    bridge = raw.split("\n")[0].strip()[:100]

            bridge_key = bridge[:50] if bridge else "no_bridge"
            # [FIXED] مفتاح الكاش يشمل methodology → كل منهجية مستقلة
            cache_key  = f"{dn.strip().lower()}::{bridge_key}::{methodology}"
            if cache_key not in cache:
                drug_names.add((dn.strip(), bridge))

    if verbose:
        print(f"  [WikiV2] Prefetching {len(drug_names)} drugs from Wikipedia V2 (method={methodology})...")

    fetched = 0
    failed  = 0
    for i, (drug_name, bridge) in enumerate(sorted(drug_names)):
        chunks = None
        for attempt in range(WIKI_MAX_RETRIES):
            try:
                chunks = retrieve_wikipedia_v2(
                    drug_name=drug_name,
                    bridge_info=bridge,
                    use_cache=True,
                    methodology=methodology,   # [FIXED] pass methodology
                    verbose=False,
                )
                if chunks:
                    fetched += 1
                    break
                else:
                    time.sleep(WIKI_BATCH_DELAY * (attempt + 1))
            except Exception as e:
                logger.debug(f"Failed to fetch {drug_name} (attempt {attempt+1}/{WIKI_MAX_RETRIES}): {e}")
                time.sleep(WIKI_BATCH_DELAY * (attempt + 1))
        else:
            failed += 1

        if i + 1 < len(drug_names):
            time.sleep(WIKI_DRUG_DELAY)

        if (i + 1) % WIKI_BATCH_PAUSE == 0:
            if verbose:
                print(f"  [WikiV2] {i+1}/{len(drug_names)} — {fetched} fetched, {failed} failed")
            if i + 1 < len(drug_names):
                time.sleep(WIKI_BATCH_DELAY)

    if verbose:
        print(f"  [WikiV2] Done — {fetched} fetched, {failed} failed. Cache: {WIKI_V2_CACHE_FILE}")

    return cache


def score_combined_retrieval(
    internal_retrieved: list,
    wiki_retrieved: list,
    drug_name: str,
    bridge_info: str = "",
    candidate_names: list = None,
    internal_weight: float = 1.0,
    wiki_weight: float = 0.7,
    bridge_boost: float = 1.3,
    candidate_boost: float = 1.2,
) -> list:
    """
    [FIXED v2] Slot-Based Combination.
      Slot 1-2: دائماً internal top-2 (ضمان عدم التراجع عن baseline)
      Slot 3:   wiki doc إذا bridge≥1 AND candidate≥2، وإلا → internal[2]
    """
    result = []

    # ── SLOTS 1-2: دائماً top-2 internal ──
    for doc in internal_retrieved[:2]:
        text  = doc.get("text",  "") if isinstance(doc, dict) else str(doc)
        score = doc.get("score", 0.5) if isinstance(doc, dict) else 0.5
        text_lower = text.lower()
        boost = 1.0
        if bridge_info:
            bridge_terms = [w for w in bridge_info.lower().split() if len(w) > 3]
            if any(t in text_lower for t in bridge_terms):
                boost *= bridge_boost
        if candidate_names:
            if any(c.lower() in text_lower for c in candidate_names if c):
                boost *= candidate_boost
        result.append({
            "text": text, "score": score * internal_weight * boost,
            "source": "internal",
            "rank": doc.get("rank", 0) if isinstance(doc, dict) else 0,
        })

    # ── SLOT 3: أفضل wiki (إذا bridge≥1 AND candidate≥2) أو internal[2] ──
    best_wiki       = None
    best_wiki_score = 0.0

    for doc in wiki_retrieved:
        text       = doc if isinstance(doc, str) else doc.get("text", str(doc))
        text_lower = text.lower()

        bridge_match = 0
        if bridge_info:
            bridge_terms = [w for w in bridge_info.lower().split() if len(w) > 3]
            bridge_match = sum(1 for t in bridge_terms if t in text_lower)

        cand_match = 0
        if candidate_names:
            cand_match = sum(1 for c in candidate_names if c and c.lower() in text_lower)

        if bridge_match >= 1 and cand_match >= 2:
            relevance = bridge_match * 0.30 + cand_match * 0.25
            if drug_name.lower() in text_lower:
                relevance += 0.10
            if relevance > best_wiki_score:
                best_wiki_score = relevance
                best_wiki       = text

    if best_wiki:
        internal_min   = min(d["score"] for d in result) if result else 0.3
        wiki_final     = internal_min * 0.88
        result.append({"text": best_wiki, "score": wiki_final,
                        "source": "wikipedia", "rank": 3})
    elif len(internal_retrieved) >= 3:
        doc   = internal_retrieved[2]
        text  = doc.get("text",  "") if isinstance(doc, dict) else str(doc)
        score = doc.get("score", 0.5) if isinstance(doc, dict) else 0.5
        text_lower = text.lower()
        boost = 1.0
        if bridge_info:
            bridge_terms = [w for w in bridge_info.lower().split() if len(w) > 3]
            if any(t in text_lower for t in bridge_terms):
                boost *= bridge_boost
        if candidate_names:
            if any(c.lower() in text_lower for c in candidate_names if c):
                boost *= candidate_boost
        result.append({
            "text": text, "score": score * internal_weight * boost,
            "source": "internal",
            "rank": doc.get("rank", 0) if isinstance(doc, dict) else 0,
        })

    result.sort(key=lambda x: x["score"], reverse=True)
    for i, doc in enumerate(result):
        doc["rank"] = i + 1

    return result


if __name__ == "__main__":
    # اختبار سريع
    print("Testing Wikipedia V2 retriever...")
    print("="*50)

    # اختبار مع bridge info
    docs = retrieve_wikipedia_v2(
        drug_name="Moclobemide",
        bridge_info="inhibits MAO-A",
        candidate_names=["Fluoxetine", "Selegiline", "Tranylcypromine"],
        verbose=True,
    )
    print(f"\nMoclobemide (with bridge): {len(docs)} chunks")
    for i, c in enumerate(docs[:4], 1):
        print(f"  Chunk {i}: {c[:120]}...")

    print("\n" + "="*50)

    # اختبار بدون bridge
    docs2 = retrieve_wikipedia_v2(
        drug_name="Fluoxetine",
        verbose=True,
    )
    print(f"\nFluoxetine (no bridge): {len(docs2)} chunks")
