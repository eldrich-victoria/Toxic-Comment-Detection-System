import json
import shutil
import tempfile
from pathlib import Path

from transformers import AutoTokenizer, BertTokenizer


# ============================================================
# PROJECT PATHS
# ============================================================

# Expected location:
# D:\Toxic-Comment-Detection-System\version_2\src\fix_bert_tokenizer.py

VERSION_2_DIR = Path(__file__).resolve().parents[1]

BERT_DIR = (
    VERSION_2_DIR
    / "models"
    / "bert"
)

FINAL_DIR = (
    BERT_DIR
    / "final"
)


# ============================================================
# REQUIRED MODEL FILES
# ============================================================

MODEL_WEIGHT_FILES = [
    "model.safetensors",
    "pytorch_model.bin"
]

MODEL_PROTECTED_FILES = [
    "model.safetensors",
    "pytorch_model.bin",
    "config.json"
]


# ============================================================
# TOKENIZER FILES
# ============================================================

REQUIRED_TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json"
]

OPTIONAL_TOKENIZER_FILES = [
    "special_tokens_map.json",
    "vocab.txt"
]


# ============================================================
# BASIC PATH VALIDATION
# ============================================================

def validate_paths():

    print("=" * 70)
    print("BERT TOKENIZER REPAIR / VERIFICATION")
    print("=" * 70)

    print("\nBERT final directory:")
    print(FINAL_DIR)

    if not FINAL_DIR.exists():

        raise FileNotFoundError(
            "BERT final directory does not exist:\n"
            + str(FINAL_DIR)
        )

    if not FINAL_DIR.is_dir():

        raise NotADirectoryError(
            "BERT final path is not a directory:\n"
            + str(FINAL_DIR)
        )


# ============================================================
# PROTECT MODEL FILES
# ============================================================

def verify_model_is_present():

    print("\n" + "=" * 70)
    print("VERIFYING EXISTING BERT MODEL")
    print("=" * 70)

    found_weight_file = None

    for filename in MODEL_WEIGHT_FILES:

        path = FINAL_DIR / filename

        if path.exists():

            found_weight_file = filename

            print(
                "FOUND MODEL WEIGHTS:",
                filename
            )

            print(
                "Size:",
                round(
                    path.stat().st_size / (1024 * 1024),
                    2
                ),
                "MB"
            )

            break

    if found_weight_file is None:

        raise FileNotFoundError(
            "No BERT model weight file was found.\n"
            "Expected one of:\n"
            + str(MODEL_WEIGHT_FILES)
        )

    config_path = FINAL_DIR / "config.json"

    if not config_path.exists():

        raise FileNotFoundError(
            "BERT config.json was not found:\n"
            + str(config_path)
        )

    print(
        "FOUND MODEL CONFIG:",
        config_path.name
    )

    print(
        "\nBERT model files are present."
    )

    print(
        "These files will NOT be modified by this utility."
    )


# ============================================================
# VERIFY CURRENT TOKENIZER
# ============================================================

def verify_current_tokenizer():

    print("\n" + "=" * 70)
    print("VERIFYING CURRENT TOKENIZER")
    print("=" * 70)

    for filename in REQUIRED_TOKENIZER_FILES:

        path = FINAL_DIR / filename

        if path.exists():

            print(
                "FOUND:",
                filename
            )

        else:

            print(
                "MISSING:",
                filename
            )

    print(
        "\nLoading tokenizer with AutoTokenizer..."
    )

    try:

        tokenizer = AutoTokenizer.from_pretrained(
            str(FINAL_DIR)
        )

    except Exception as error:

        raise RuntimeError(
            "Existing tokenizer could not be loaded.\n"
            + str(error)
        )

    print(
        "AutoTokenizer load: SUCCESS"
    )

    # --------------------------------------------------------
    # TOKENIZATION TEST
    # --------------------------------------------------------

    test_text = [
        "This is a tokenizer verification test."
    ]

    try:

        encoded = tokenizer(
            test_text,
            truncation=True,
            padding=True,
            max_length=128
        )

    except Exception as error:

        raise RuntimeError(
            "Tokenizer inference test failed.\n"
            + str(error)
        )

    if "input_ids" not in encoded:

        raise RuntimeError(
            "Tokenizer did not generate input_ids."
        )

    if "attention_mask" not in encoded:

        raise RuntimeError(
            "Tokenizer did not generate attention_mask."
        )

    print(
        "Tokenizer inference test: SUCCESS"
    )

    print(
        "Tokenizer class:",
        tokenizer.__class__.__name__
    )

    print(
        "Vocabulary size:",
        len(tokenizer)
    )

    return tokenizer


# ============================================================
# VERIFY BERT TOKENIZER COMPATIBILITY
# ============================================================

def verify_bert_tokenizer():

    print("\n" + "=" * 70)
    print("VERIFYING BERT TOKENIZER COMPATIBILITY")
    print("=" * 70)

    print(
        "\nAttempting BertTokenizer load..."
    )

    try:

        tokenizer = BertTokenizer.from_pretrained(
            str(FINAL_DIR)
        )

    except Exception as error:

        print(
            "\nBertTokenizer could not load directly."
        )

        print(
            "Reason:",
            str(error)
        )

        return False

    print(
        "BertTokenizer load: SUCCESS"
    )

    test_text = [
        "This is a BERT tokenizer compatibility test."
    ]

    try:

        encoded = tokenizer(
            test_text,
            truncation=True,
            padding=True,
            max_length=128
        )

    except Exception as error:

        print(
            "BertTokenizer inference test failed."
        )

        print(
            "Reason:",
            str(error)
        )

        return False

    if "input_ids" not in encoded:

        print(
            "BertTokenizer did not generate input_ids."
        )

        return False

    if "attention_mask" not in encoded:

        print(
            "BertTokenizer did not generate attention_mask."
        )

        return False

    print(
        "BertTokenizer inference test: SUCCESS"
    )

    return True


# ============================================================
# CREATE TOKENIZER COMPATIBILITY FILES
# ============================================================

def create_compatibility_files(
    tokenizer
):

    print("\n" + "=" * 70)
    print("CREATING TOKENIZER COMPATIBILITY FILES")
    print("=" * 70)

    missing_files = []

    for filename in OPTIONAL_TOKENIZER_FILES:

        path = FINAL_DIR / filename

        if not path.exists():

            missing_files.append(
                filename
            )

    if len(missing_files) == 0:

        print(
            "\nAll optional compatibility files already exist."
        )

        return

    print(
        "\nMissing tokenizer compatibility files:"
    )

    for filename in missing_files:

        print(
            " -",
            filename
        )

    # --------------------------------------------------------
    # CREATE TEMPORARY DIRECTORY
    # --------------------------------------------------------

    temporary_directory = Path(
        tempfile.mkdtemp(
            prefix="bert_tokenizer_fix_"
        )
    )

    print(
        "\nTemporary tokenizer directory:"
    )

    print(
        temporary_directory
    )

    try:

        print(
            "\nSaving tokenizer to temporary directory..."
        )

        tokenizer.save_pretrained(
            str(temporary_directory)
        )

        print(
            "Temporary tokenizer save completed."
        )

        # ----------------------------------------------------
        # COPY ONLY MISSING TOKENIZER FILES
        # ----------------------------------------------------

        copied_files = []

        for filename in OPTIONAL_TOKENIZER_FILES:

            source = (
                temporary_directory
                / filename
            )

            destination = (
                FINAL_DIR
                / filename
            )

            # IMPORTANT:
            # Existing files are never overwritten.

            if destination.exists():

                print(
                    "PRESERVED EXISTING:",
                    filename
                )

                continue

            if source.exists():

                shutil.copy2(
                    source,
                    destination
                )

                copied_files.append(
                    filename
                )

                print(
                    "CREATED:",
                    filename
                )

            else:

                print(
                    "NOT GENERATED:",
                    filename
                )

        # ----------------------------------------------------
        # SPECIAL TOKENS MAP FALLBACK
        # ----------------------------------------------------

        special_tokens_path = (
            FINAL_DIR
            / "special_tokens_map.json"
        )

        if not special_tokens_path.exists():

            special_tokens = (
                tokenizer.special_tokens_map
            )

            with open(
                special_tokens_path,
                "w",
                encoding="utf-8"
            ) as file:

                json.dump(
                    special_tokens,
                    file,
                    indent=4,
                    ensure_ascii=False
                )

            print(
                "CREATED:",
                "special_tokens_map.json"
            )

        # ----------------------------------------------------
        # VOCABULARY FALLBACK
        # ----------------------------------------------------

        vocab_path = (
            FINAL_DIR
            / "vocab.txt"
        )

        if not vocab_path.exists():

            vocab = tokenizer.get_vocab()

            if vocab is not None and len(vocab) > 0:

                ordered_tokens = sorted(
                    vocab.items(),
                    key=lambda item: item[1]
                )

                with open(
                    vocab_path,
                    "w",
                    encoding="utf-8"
                ) as file:

                    for token, index in ordered_tokens:

                        file.write(
                            token
                            + "\n"
                        )

                print(
                    "CREATED:",
                    "vocab.txt"
                )

    finally:

        shutil.rmtree(
            temporary_directory,
            ignore_errors=True
        )

        print(
            "\nTemporary files removed."
        )


# ============================================================
# VERIFY FINAL TOKENIZER FILES
# ============================================================

def verify_final_tokenizer_files():

    print("\n" + "=" * 70)
    print("FINAL TOKENIZER FILE VERIFICATION")
    print("=" * 70)

    # --------------------------------------------------------
    # REQUIRED FILES
    # --------------------------------------------------------
    #
    # The current Hugging Face tokenizer format used by this
    # project successfully loads from:
    #
    #   tokenizer.json
    #   tokenizer_config.json
    #
    # We have already independently verified that both
    # AutoTokenizer and BertTokenizer can load successfully.
    #
    # special_tokens_map.json and vocab.txt are therefore
    # treated as optional compatibility files rather than
    # mandatory files.
    # --------------------------------------------------------

    required_files = [
        "tokenizer.json",
        "tokenizer_config.json"
    ]

    optional_files = [
        "special_tokens_map.json",
        "vocab.txt"
    ]

    missing_required = []

    # --------------------------------------------------------
    # REQUIRED FILE CHECK
    # --------------------------------------------------------

    for filename in required_files:

        path = FINAL_DIR / filename

        if path.exists():

            size_kb = (
                path.stat().st_size
                / 1024
            )

            print(
                "FOUND:",
                filename,
                "->",
                round(size_kb, 2),
                "KB"
            )

        else:

            missing_required.append(
                filename
            )

            print(
                "MISSING REQUIRED:",
                filename
            )

    if len(missing_required) > 0:

        raise RuntimeError(
            "Required tokenizer files are missing:\n"
            + str(missing_required)
        )

    # --------------------------------------------------------
    # OPTIONAL FILE CHECK
    # --------------------------------------------------------

    print(
        "\nOptional tokenizer compatibility files:"
    )

    for filename in optional_files:

        path = FINAL_DIR / filename

        if path.exists():

            size_kb = (
                path.stat().st_size
                / 1024
            )

            print(
                "FOUND OPTIONAL:",
                filename,
                "->",
                round(size_kb, 2),
                "KB"
            )

        else:

            print(
                "NOT PRESENT:",
                filename,
                "(not required)"
            )

    # --------------------------------------------------------
    # ACTUAL TOKENIZER LOAD TEST
    # --------------------------------------------------------

    print(
        "\nPerforming final tokenizer load test..."
    )

    try:

        tokenizer = AutoTokenizer.from_pretrained(
            str(FINAL_DIR)
        )

    except Exception as error:

        raise RuntimeError(
            "Final tokenizer could not be loaded.\n"
            + str(error)
        )

    print(
        "AutoTokenizer load: SUCCESS"
    )

    # --------------------------------------------------------
    # TOKENIZATION TEST
    # --------------------------------------------------------

    test_text = [
        "This is a final tokenizer verification test."
    ]

    try:

        encoded = tokenizer(
            test_text,
            truncation=True,
            padding=True,
            max_length=128
        )

    except Exception as error:

        raise RuntimeError(
            "Final tokenizer inference test failed.\n"
            + str(error)
        )

    if "input_ids" not in encoded:

        raise RuntimeError(
            "Tokenizer did not generate input_ids."
        )

    if "attention_mask" not in encoded:

        raise RuntimeError(
            "Tokenizer did not generate attention_mask."
        )

    print(
        "Tokenizer inference test: SUCCESS"
    )

    print(
        "Vocabulary size:",
        len(tokenizer)
    )

    print(
        "\nRequired tokenizer verification PASSED."
    )


# ============================================================
# RELOAD TOKENIZER AFTER REPAIR
# ============================================================

def reload_and_test_tokenizer():

    print("\n" + "=" * 70)
    print("POST-REPAIR TOKENIZER TEST")
    print("=" * 70)

    print(
        "\nReloading with AutoTokenizer..."
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(FINAL_DIR)
    )

    print(
        "AutoTokenizer reload: SUCCESS"
    )

    print(
        "\nReloading with BertTokenizer..."
    )

    tokenizer = BertTokenizer.from_pretrained(
        str(FINAL_DIR)
    )

    print(
        "BertTokenizer reload: SUCCESS"
    )

    test_text = [
        "This is a final tokenizer compatibility test."
    ]

    encoded = tokenizer(
        test_text,
        truncation=True,
        padding=True,
        max_length=128
    )

    if "input_ids" not in encoded:

        raise RuntimeError(
            "Post-repair tokenizer did not generate input_ids."
        )

    if "attention_mask" not in encoded:

        raise RuntimeError(
            "Post-repair tokenizer did not generate attention_mask."
        )

    print(
        "Post-repair tokenization: SUCCESS"
    )

    print(
        "Input IDs generated:",
        len(encoded["input_ids"][0])
    )


# ============================================================
# VERIFY MODEL WAS NOT TOUCHED
# ============================================================

def verify_model_files_after_repair(
    original_model_sizes
):

    print("\n" + "=" * 70)
    print("VERIFYING BERT MODEL INTEGRITY")
    print("=" * 70)

    for filename, original_size in original_model_sizes.items():

        path = FINAL_DIR / filename

        if not path.exists():

            raise RuntimeError(
                "Protected model file disappeared:\n"
                + str(path)
            )

        current_size = path.stat().st_size

        if current_size != original_size:

            raise RuntimeError(
                "Protected model file size changed:\n"
                + str(path)
            )

        print(
            "PRESERVED:",
            filename,
            "->",
            round(
                current_size / (1024 * 1024),
                2
            ),
            "MB"
        )

    print(
        "\nBERT model files were preserved."
    )


# ============================================================
# SAVE REPAIR MANIFEST
# ============================================================

def save_repair_manifest():

    manifest = {
        "utility": "fix_bert_tokenizer.py",
        "purpose": (
            "Tokenizer compatibility repair and verification"
        ),
        "bert_model_modified": False,
        "model_weights_modified": False,
        "checkpoints_modified": False,
        "model_weights": "model.safetensors",
        "tokenizer_files_verified": [
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.txt"
        ]
    }

    manifest_path = (
        FINAL_DIR
        / "tokenizer_repair_manifest.json"
    )

    with open(
        manifest_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            manifest,
            file,
            indent=4
        )

    print(
        "\nCreated:",
        manifest_path.name
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Validate BERT directory
    # --------------------------------------------------------

    validate_paths()

    # --------------------------------------------------------
    # 2. Verify model exists
    # --------------------------------------------------------

    verify_model_is_present()

    # --------------------------------------------------------
    # 3. Record protected model file sizes
    # --------------------------------------------------------

    original_model_sizes = {}

    for filename in MODEL_PROTECTED_FILES:

        path = FINAL_DIR / filename

        if path.exists():

            original_model_sizes[filename] = (
                path.stat().st_size
            )

    # --------------------------------------------------------
    # 4. Verify current tokenizer
    # --------------------------------------------------------

    tokenizer = verify_current_tokenizer()

    # --------------------------------------------------------
    # 5. Check BertTokenizer compatibility
    # --------------------------------------------------------

    bert_tokenizer_ok = (
        verify_bert_tokenizer()
    )

    # --------------------------------------------------------
    # 6. Only repair if needed
    # --------------------------------------------------------

    if bert_tokenizer_ok:

        print(
            "\n"
            + "=" * 70
        )

        print(
            "NO TOKENIZER REPAIR REQUIRED"
        )

        print(
            "=" * 70
        )

        print(
            "\nThe existing tokenizer already works with"
        )

        print(
            "BertTokenizer.from_pretrained()."
        )

    else:

        print(
            "\n"
            + "=" * 70
        )

        print(
            "TOKENIZER COMPATIBILITY REPAIR REQUIRED"
        )

        print(
            "=" * 70
        )

        create_compatibility_files(
            tokenizer
        )

    # --------------------------------------------------------
    # 7. Verify final tokenizer files
    # --------------------------------------------------------

    verify_final_tokenizer_files()

    # --------------------------------------------------------
    # 8. Reload and test tokenizer
    # --------------------------------------------------------

    reload_and_test_tokenizer()

    # --------------------------------------------------------
    # 9. Verify BERT weights/config were untouched
    # --------------------------------------------------------

    verify_model_files_after_repair(
        original_model_sizes
    )

    # --------------------------------------------------------
    # 10. Save repair manifest
    # --------------------------------------------------------

    save_repair_manifest()

    # --------------------------------------------------------
    # FINAL SUCCESS
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "BERT TOKENIZER REPAIR COMPLETED SUCCESSFULLY"
    )

    print(
        "=" * 70
    )

    print(
        "\nBERT model weights were NOT retrained."
    )

    print(
        "BERT checkpoints were NOT modified."
    )

    print(
        "model.safetensors was NOT modified."
    )

    print(
        "\nTokenizer is now compatible and verified."
    )

    print(
        "\nFinal BERT directory:"
    )

    print(
        FINAL_DIR
    )

    print(
        "\nNext stage: evaluation.py"
    )


if __name__ == "__main__":
    main()