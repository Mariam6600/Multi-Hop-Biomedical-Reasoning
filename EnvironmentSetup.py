"""
EnvironmentSetup.py
====================
Biomedical Multi-Hop QA Project G�� Baseline 1
Full environment preparation script.

What this script does:
  1. Checks Python version (3.10+)
  2. Checks Ollama service is running
  3. Checks GPU via nvidia-smi
  4. Checks GGUF model file exists
  5. Installs all required Python libraries
  6. Verifies all imports work correctly
  7. Creates project folder structure
  8. Creates Ollama Modelfile for BioMistral-7B-GGUF
  9. Registers BioMistral model inside Ollama
 10. Tests the model with a sample biomedical question
 11. Prints final summary

Usage:
    python EnvironmentSetup.py

Prerequisites (do these BEFORE running this script):
  - Ollama installed and running:
        ollama serve
  - GGUF file downloaded from HuggingFace and placed at:
        models\\BioMistral-7B.Q4_K_M.gguf
"""

import sys
import subprocess
import importlib
import os
import platform
import time

# G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��
# CONFIGURATION
# G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��

# Root of the project = folder containing this script
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Path to the downloaded GGUF model file
GGUF_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "BioMistral-7B.Q4_K_M.gguf")

# Modelfile will be created here
MODELFILE_PATH = os.path.join(PROJECT_ROOT, "models", "Modelfile")

# Name Ollama will use for this model
OLLAMA_MODEL_NAME = "biomistral-7b"

# Folders to create inside the project root
REQUIRED_FOLDERS = ["config", "src", "data", "outputs", "models"]

# Libraries to install via pip
PIP_PACKAGES = [
    "bm25s",
    "PyStemmer",
    "scispacy",
    "langchain",
    "ollama",
    "datasets",
]

# scispaCy model URL
SCISPACY_MODEL_URL = (
    "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/"
    "en_core_sci_sm-0.5.4.tar.gz"
)

# Minimum Python version
MIN_PYTHON = (3, 10)

# Ollama API base URL
OLLAMA_HOST = "http://localhost:11434"

# Test question for model verification
TEST_QUESTION = "What is the mechanism of action of aspirin?"


# G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��
# HELPERS
# G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��

def print_header(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def print_ok(msg: str):
    print(f"  [OK]   {msg}")

def print_warn(msg: str):
    print(f"  [WARN] {msg}")

def print_fail(msg: str):
    print(f"  [FAIL] {msg}")

def print_info(msg: str):
    print(f"  [INFO] {msg}")


# G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��
# STEP 1 G�� CHECK PYTHON VERSION
# G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��

def check_python_version():
    print_header("Step 1 G�� Python Version Check")
    current = sys.version_info
    print_info(f"Detected Python {current.major}.{current.minor}.{current.micro}")

    if (current.major, current.minor) >= MIN_PYTHON:
        print_ok(f"Python {current.major}.{current.minor} meets requirement (3.10+)")
        return True
    else:
        print_fail(
            f"Python {current.major}.{current.minor} is too old. "
            "Please install Python 3.10 from https://www.python.org/downloads/"
        )
        return False


# G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��
# STEP 2 G�� CHECK OLLAMA SERVICE
# G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��

def check_ollama():
    print_header("Step 2 G�� Ollama Service Check")
    try:
        import urllib.request
        urllib.request.urlopen(OLLAMA_HOST, timeout=3)
        print_ok("Ollama service is running and reachable.")
        return True
    except Exception:
        print_warn("Ollama service is NOT running.")
        print_info("Fix: open a terminal and run:  ollama serve")
        print_info("Then re-run this script.")
        return False


# G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��
# STEP 3 G�� CHECK GPU
# G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��

def check_gpu():
    print_header("Step 3 G�� GPU / CUDA Check")
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            print_ok(f"GPU detected: {result.stdout.strip()}")
        else:
            print_warn("nvidia-smi returned an error.")
    except FileNotFoundError:
        print_warn("nvidia-smi not found.")
    except Exception as e:
        print_warn(f"GPU check error: {e}")

    print_info("Ollama handles GPU inference internally G�� nvcc is not required.")


# G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��
# STEP 4 G�� CHECK GGUF FILE EXISTS
# G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��

def check_gguf_file():
    print_header("Step 4 G�� GGUF Model File Check")
    print_info(f"Looking for: {GGUF_MODEL_PATH}")

    if os.path.isfile(GGUF_MODEL_PATH):
        size_gb = os.path.getsize(GGUF_MODEL_PATH) / (1024 ** 3)
        print_ok(f"GGUF file found G�� size: {size_gb:.2f} GB")
        return True
    else:
        print_fail("GGUF file NOT found.")
        print_info("Download from: https://huggingface.co/BioMistral/BioMistral-7B-GGUF")
        print_info("File to download: BioMistral-7B.Q4_K_M.gguf")
        print_info(f"Place it at:     {GGUF_MODEL_PATH}")
        return False


# G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��
# STEP 5 G�� INSTALL PYTHON LIBRARIES
# G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��

def install_packages():
    print_header("Step 5 G�� Installing Python Libraries")

    print_info("Upgrading pip...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
        check=True, capture_output=True
    )
    print_ok("pip upgraded.")

    for package in PIP_PACKAGES:
        print_info(f"Installing: {package} ...")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", package],
                check=True, capture_output=True, text=True
            )
            print_ok(f"{package} installed.")
        except subprocess.CalledProcessError as e:
            print_fail(f"Failed to install {package}: {e.stderr[:200]}")

    print_info("Installing scispaCy language model...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", SCISPACY_MODEL_URL],
            check=True, capture_output=True, text=True
        )
        print_ok("scispaCy en_core_sci_sm installed.")
    except subprocess.CalledProcessError as e:
        print_fail(f"scispaCy model failed: {e.stderr[:200]}")
        print_info(f"Manual install:\n  pip install {SCISPACY_MODEL_URL}")


# G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��
# STEP 6 G�� VERIFY IMPORTS
# G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��

def verify_imports():
    print_header("Step 6 G�� Verifying Library Imports")

    libraries = {
        "bm25s":     "bm25s",
        "Stemmer":   "PyStemmer",
        "scispacy":  "scispacy",
        "langchain": "langchain",
        "ollama":    "ollama",
        "datasets":  "datasets",
    }

    all_ok = True
    for import_name, pip_name in libraries.items():
        try:
            importlib.import_module(import_name)
            print_ok(f"{pip_name} G�� importable G��")
        except ImportError:
            print_fail(f"{pip_name} G�� NOT importable. Run: pip install {pip_name}")
            all_ok = False

    return all_ok


# G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��
# STEP 7 G�� CREATE FOLDER STRUCTURE
# G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��

def create_folder_structure():
    print_header("Step 7 G�� Creating Project Folder Structure")

    for folder in REQUIRED_FOLDERS:
        path = os.path.join(PROJECT_ROOT, folder)
        if not os.path.exists(path):
            os.makedirs(path)
            print_ok(f"Created: {path}")
        else:
            print_info(f"Already exists: {path}")

    for pkg_folder in ["src", "config"]:
        init_file = os.path.join(PROJECT_ROOT, pkg_folder, "__init__.py")
        if not os.path.exists(init_file):
            open(init_file, "w").close()
            print_ok(f"Created: {init_file}")

    print_ok("Folder structure ready.")


# G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��
# STEP 8 G�� CREATE OLLAMA MODELFILE
# G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��

def create_modelfile():
    print_header("Step 8 G�� Creating Ollama Modelfile")

    # Ollama requires forward slashes even on Windows
    gguf_path_for_ollama = GGUF_MODEL_PATH.replace("\\", "/")

    modelfile_content = f"""FROM {gguf_path_for_ollama}

PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 2048
PARAMETER num_gpu 99

SYSTEM \"\"\"
You are a biomedical expert assistant. Answer biomedical questions accurately and concisely based on medical and scientific knowledge. Provide short, factual answers.
\"\"\"
"""

    os.makedirs(os.path.dirname(MODELFILE_PATH), exist_ok=True)

    with open(MODELFILE_PATH, "w", encoding="utf-8") as f:
        f.write(modelfile_content)

    print_ok(f"Modelfile created at: {MODELFILE_PATH}")
    return True


# G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��
# STEP 9 G�� REGISTER MODEL IN OLLAMA
# G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��

def register_model_in_ollama():
    print_header("Step 9 G�� Registering BioMistral in Ollama")

    # Check if already registered
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, timeout=10
        )
        if OLLAMA_MODEL_NAME in result.stdout:
            print_ok(f"Model '{OLLAMA_MODEL_NAME}' already registered in Ollama.")
            return True
    except Exception:
        pass

    print_info(f"Running: ollama create {OLLAMA_MODEL_NAME} -f Modelfile")
    print_info("This may take 1-2 minutes...")

    try:
        result = subprocess.run(
            ["ollama", "create", OLLAMA_MODEL_NAME, "-f", MODELFILE_PATH],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode == 0:
            print_ok(f"Model '{OLLAMA_MODEL_NAME}' registered successfully!")
            return True
        else:
            print_fail(f"Ollama create failed:\n{result.stderr[:400]}")
            print_info("Try manually in terminal:")
            print_info(f'  cd "{os.path.dirname(MODELFILE_PATH)}"')
            print_info(f"  ollama create {OLLAMA_MODEL_NAME} -f Modelfile")
            return False
    except subprocess.TimeoutExpired:
        print_fail("Ollama create timed out.")
        print_info("Try manually in terminal:")
        print_info(f'  cd "{os.path.dirname(MODELFILE_PATH)}"')
        print_info(f"  ollama create {OLLAMA_MODEL_NAME} -f Modelfile")
        return False
    except Exception as e:
        print_fail(f"Unexpected error: {e}")
        return False


# G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��
# STEP 10 G�� TEST THE MODEL
# G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��

def test_model():
    print_header("Step 10 G�� Testing BioMistral Model")
    print_info(f"Test question: '{TEST_QUESTION}'")
    print_info("First run may take 30-60 seconds to load the model into VRAM...")

    try:
        import ollama as ollama_client

        start = time.time()
        response = ollama_client.chat(
            model=OLLAMA_MODEL_NAME,
            messages=[{"role": "user", "content": TEST_QUESTION}]
        )
        elapsed = time.time() - start

        answer = response["message"]["content"].strip()
        print_ok(f"Model responded in {elapsed:.1f} seconds")
        print()
        print("  G��G�� Model Answer " + "G��" * 43)
        for line in answer[:500].split("\n"):
            print(f"  G�� {line}")
        print("  G��" + "G��" * 58)
        return True

    except Exception as e:
        print_fail(f"Model test failed: {e}")
        print_info("Make sure: ollama serve is running in another terminal")
        print_info(f"And model is registered: ollama list  (look for {OLLAMA_MODEL_NAME})")
        return False


# G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��
# FINAL SUMMARY
# G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��

def print_summary(results: dict):
    print_header("Final Setup Summary")

    all_passed = True
    for check, passed in results.items():
        if passed:
            print_ok(check)
        else:
            print_fail(check)
            all_passed = False

    print()
    if all_passed:
        print("  G��  Environment is FULLY READY.")
        print()
        print("  Next file to create: config/settings.py")
        print(f"  Ollama model name  : '{OLLAMA_MODEL_NAME}'")
        print(f"  Project root       : {PROJECT_ROOT}")
    else:
        print("  G��n+�  Some checks failed.")
        print("  Fix the [FAIL] items above, then re-run this script.")

    print("=" * 60 + "\n")


# G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��
# MAIN
# G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��G��

def main():
    print("\n" + "=" * 60)
    print("  Biomedical Multi-Hop QA Project")
    print("  Environment Setup G�� Baseline 1")
    print("=" * 60)
    print(f"  OS      : {platform.system()} {platform.release()}")
    print(f"  Python  : {sys.version.split()[0]}")
    print(f"  Project : {PROJECT_ROOT}")
    print(f"  Model   : BioMistral-7B-GGUF (Q4_K_M) via Ollama")

    results = {}

    # Step 1
    results["Python 3.10+"] = check_python_version()
    if not results["Python 3.10+"]:
        print_fail("Python requirement not met. Aborting.")
        sys.exit(1)

    # Step 2
    results["Ollama running"] = check_ollama()

    # Step 3 G�� informational only
    check_gpu()

    # Step 4
    results["GGUF file present"] = check_gguf_file()

    # Step 5
    install_packages()

    # Step 6
    results["Libraries importable"] = verify_imports()

    # Step 7
    create_folder_structure()
    results["Folders created"] = True

    # Steps 8-10 only if prerequisites met
    if results["Ollama running"] and results["GGUF file present"]:
        create_modelfile()
        results["Model registered in Ollama"] = register_model_in_ollama()

        if results["Model registered in Ollama"]:
            results["Model test passed"] = test_model()
        else:
            results["Model test passed"] = False
    else:
        print_warn("\nSkipping model registration and test due to missing prerequisites.")
        if not results["Ollama running"]:
            print_info("G�� Start Ollama: ollama serve")
        if not results["GGUF file present"]:
            print_info(f"G�� Place GGUF at: {GGUF_MODEL_PATH}")
        print_info("Re-run this script after fixing the above.")
        results["Model registered in Ollama"] = False
        results["Model test passed"] = False

    print_summary(results)


if __name__ == "__main__":
    main()
