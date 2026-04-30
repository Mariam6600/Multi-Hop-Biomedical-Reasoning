"""
src/retriever_wikipedia.py
===========================
Biomedical Multi-Hop QA — Wikipedia External Retriever

المبدأ :
  

ما يفعله:
  1. يبحث في ويكيبيديا عن الدواء (باسمه أو drug_name)
  2. يجلب صفحة الدواء أو المقالة الأقرب
  3. يقطّعها لـ chunks مناسبة (حوالي 400 حرف)
  4. يُعيدها كـ list[str] بنفس شكل internal supports

الفائدة:
  - تغطية خارجية للحالات التي الـ internal supports فيها ضعيفة
  - معلومات عن الآلية وطريقة عمل الدواء بشكل عام
  - مجانية ولا تحتاج API key

الـ Cache:
  نحفظ نتائج ويكيبيديا في ملف cache لتجنب الطلبات المتكررة
  (cache_file: outputs/wikipedia_cache.json)

Usage:
    from src.retriever_wikipedia import retrieve_wikipedia_docs

    wiki_docs = retrieve_wikipedia_docs("Moclobemide", query="drug interaction")
    # Returns: list[str]  (قائمة نصوص جاهزة للـ hybrid_scored retriever)
"""

import os, sys, json, re, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import OUTPUTS_DIR

WIKI_CACHE_FILE = os.path.join(OUTPUTS_DIR, "wikipedia_cache.json")
WIKI_CHUNK_SIZE = 400     # حجم كل chunk بالأحرف
WIKI_MAX_CHUNKS = 8       # أقصى عدد chunks من ويكيبيديا لكل دواء
WIKI_RATE_DELAY = 0.3     # ثواني بين الطلبات (لتجنب الحظر)


# ─────────────────────────────────────────────
# CACHE MANAGEMENT
# ─────────────────────────────────────────────

def _load_wiki_cache() -> dict:
    if os.path.exists(WIKI_CACHE_FILE):
        try:
            with open(WIKI_CACHE_FILE, encoding="utf-8") as f:
                return json.load(f)
        except:
            pass
    return {}


def _save_wiki_cache(cache: dict):
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    with open(WIKI_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


# ─────────────────────────────────────────────
# FETCH FROM WIKIPEDIA
# ─────────────────────────────────────────────

def _fetch_wikipedia_text(drug_name: str, max_chars: int = 3000) -> str:
    """
    يجلب نص ويكيبيديا لاسم الدواء.
    يستخدم Wikipedia REST API (لا يحتاج API key).

    يجرب:
      1. اسم الدواء مباشرةً
      2. اسم الدواء + " (drug)"
      3. بحث Wikipedia وأول نتيجة

    Returns: النص أو "" إذا فشل
    """
    try:
        import urllib.request
        import urllib.parse

        def _get_page(title):
            safe = urllib.parse.quote(title.replace(" ", "_"))
            url  = f"https://en.wikipedia.org/api/rest_v1/page/summary/{safe}"
            req  = urllib.request.Request(
                url,
                headers={"User-Agent": "BiomedQA/1.0 (academic research)"}
            )
            try:
                with urllib.request.urlopen(req, timeout=8) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    extract = data.get("extract", "")
                    if len(extract) > 50:
                        return extract
            except:
                pass
            return ""

        def _search_wiki(query):
            """بحث Wikipedia وإرجاع أول نتيجة."""
            safe  = urllib.parse.quote(query)
            url   = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={safe}&format=json&srlimit=1"
            req   = urllib.request.Request(
                url,
                headers={"User-Agent": "BiomedQA/1.0 (academic research)"}
            )
            try:
                with urllib.request.urlopen(req, timeout=8) as resp:
                    data  = json.loads(resp.read().decode("utf-8"))
                    hits  = data.get("query", {}).get("search", [])
                    if hits:
                        return hits[0]["title"]
            except:
                pass
            return ""

        def _get_full_page(title):
            """يجلب محتوى الصفحة الكامل (أكثر تفصيلاً من summary)."""
            safe = urllib.parse.quote(title.replace(" ", "_"))
            url  = (f"https://en.wikipedia.org/w/api.php?action=query"
                    f"&prop=extracts&exintro=false&explaintext=true"
                    f"&titles={safe}&format=json")
            req  = urllib.request.Request(
                url,
                headers={"User-Agent": "BiomedQA/1.0 (academic research)"}
            )
            try:
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data  = json.loads(resp.read().decode("utf-8"))
                    pages = data.get("query", {}).get("pages", {})
                    for page_id, page in pages.items():
                        if page_id != "-1":
                            return page.get("extract", "")
            except:
                pass
            return ""

        # محاولة 1: اسم الدواء مباشرة
        text = _get_page(drug_name)
        if not text:
            # محاولة 2: اسم الدواء + " (drug)"
            text = _get_page(f"{drug_name} (drug)")
        if not text:
            # محاولة 3: بحث وأخذ أول نتيجة
            title = _search_wiki(f"{drug_name} drug mechanism interaction")
            if title:
                text = _get_full_page(title)

        # تنظيف النص
        if text:
            text = re.sub(r'\n{3,}', '\n\n', text)   # إزالة الأسطر الفارغة المتعددة
            text = text.strip()
            return text[:max_chars]

    except Exception as e:
        pass

    return ""


# ─────────────────────────────────────────────
# CHUNKING
# ─────────────────────────────────────────────

def _chunk_text(text: str, chunk_size: int = WIKI_CHUNK_SIZE, max_chunks: int = WIKI_MAX_CHUNKS) -> list:
    """
    يقطّع نص ويكيبيديا إلى chunks مناسبة.
    يحترم حدود الجمل قدر الإمكان.

    Returns: list[str]
    """
    if not text:
        return []

    # Split by sentences (rough)
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
            current     = [sent]
            current_len = len(sent)
        else:
            current.append(sent)
            current_len += len(sent) + 1

    if current:
        chunk = " ".join(current).strip()
        if len(chunk) > 50:
            chunks.append(chunk)

    return chunks[:max_chunks]


# ─────────────────────────────────────────────
# MAIN PUBLIC FUNCTION
# ─────────────────────────────────────────────

def retrieve_wikipedia_docs(
    drug_name: str,
    query: str = "",
    max_chunks: int = WIKI_MAX_CHUNKS,
    use_cache: bool = True,
) -> list:
    """
    يجلب وثائق ويكيبيديا للدواء كـ list[str] جاهزة للـ retriever.

    Args:
        drug_name:  اسم الدواء
        query:      نص السؤال الأصلي (للـ logging فقط)
        max_chunks: أقصى عدد chunks
        use_cache:  استخدام الـ cache لتجنب الطلبات المتكررة

    Returns: list[str] — كل عنصر هو chunk نصية من ويكيبيديا
    """
    if not drug_name or not drug_name.strip():
        return []

    drug_key = drug_name.strip().lower()

    # ── Check cache ──
    cache = _load_wiki_cache() if use_cache else {}
    if drug_key in cache:
        chunks = cache[drug_key]
        return chunks[:max_chunks]

    # ── Fetch from Wikipedia ──
    time.sleep(WIKI_RATE_DELAY)  # rate limiting
    text = _fetch_wikipedia_text(drug_name)

    if not text:
        # Cache empty result too (to avoid re-fetching failures)
        cache[drug_key] = []
        if use_cache:
            _save_wiki_cache(cache)
        return []

    # ── Chunk ──
    chunks = _chunk_text(text, chunk_size=WIKI_CHUNK_SIZE, max_chunks=max_chunks)

    # ── Cache ──
    cache[drug_key] = chunks
    if use_cache:
        _save_wiki_cache(cache)

    return chunks


# ─────────────────────────────────────────────
# PREFETCH UTILITY
# (لتحميل ويكيبيديا لكل الأدوية مسبقاً قبل التجربة)
# ─────────────────────────────────────────────

def prefetch_wikipedia_for_dataset(data: list, verbose: bool = True) -> dict:
    """
    يحمّل مسبقاً ويكيبيديا لكل الأدوية في الداتاسيت.
    مفيد للتشغيل مرة واحدة قبل تجربة طويلة.

    Args:
        data: قائمة من records (من medhop.json)
        verbose: طباعة التقدم

    Returns: dict {drug_name: [chunks]}
    """
    cache = _load_wiki_cache()
    drug_names = set()

    for rec in data:
        dn = rec.get("query_drug_name", "")
        if dn and dn.strip().lower() not in cache:
            drug_names.add(dn.strip())

    if verbose:
        print(f"  [Wiki] Prefetching {len(drug_names)} new drugs from Wikipedia...")

    fetched = 0
    failed  = 0
    for i, drug_name in enumerate(sorted(drug_names)):
        chunks = retrieve_wikipedia_docs(drug_name, use_cache=True)
        if chunks:
            fetched += 1
        else:
            failed += 1
        if verbose and (i + 1) % 20 == 0:
            print(f"  [Wiki] {i+1}/{len(drug_names)} — {fetched} fetched, {failed} failed")

    if verbose:
        print(f"  [Wiki] Done — {fetched} fetched, {failed} failed. Cache: {WIKI_CACHE_FILE}")

    return cache


if __name__ == "__main__":
    # اختبار سريع
    print("Testing Wikipedia retriever...")
    chunks = retrieve_wikipedia_docs("Moclobemide")
    print(f"Drug: Moclobemide → {len(chunks)} chunks")
    for i, c in enumerate(chunks[:3], 1):
        print(f"  Chunk {i}: {c[:150]}...")

    chunks2 = retrieve_wikipedia_docs("Fluoxetine")
    print(f"\nDrug: Fluoxetine → {len(chunks2)} chunks")
    for i, c in enumerate(chunks2[:2], 1):
        print(f"  Chunk {i}: {c[:150]}...")
