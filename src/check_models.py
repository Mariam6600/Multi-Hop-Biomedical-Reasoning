import urllib.request
import json
import sys
import os

# تحميل الإعدادات
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import GOOGLE_API_KEY

url = f"https://generativelanguage.googleapis.com/v1beta/models?key={GOOGLE_API_KEY}"

try:
    req = urllib.request.Request(url)
    response = urllib.request.urlopen(req, timeout=30)
    data = json.loads(response.read().decode("utf-8"))

    print("\n" + "=" * 60)
    print("  Available Models for your API Key:")
    print("=" * 60)

    # استخراج الأسماء
    if "models" in data:
        for model in data["models"]:
            # استخراج الاسم المختصر (بدون models/)
            full_name = model.get("name", "")
            short_name = full_name.replace("models/", "")
            
            # التحقق مما إذا كان يدعم generateContent (للمحادثة)
            methods = model.get("supportedGenerationMethods", [])
            if "generateContent" in methods:
                print(f"  ✅ {short_name}  (Supported)")
            else:
                print(f"  -- {short_name}  (Embedding/Other)")
    else:
        print("  [FAIL] No models found. Check API Key.")
        print(data)

    print("=" * 60 + "\n")

except Exception as e:
    print(f"\n  [ERROR] {e}\n")