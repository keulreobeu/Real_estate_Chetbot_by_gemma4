from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from common import EXAMPLE_MODEL_CONFIG_FILE, LOCAL_MODEL_CONFIG_FILE, load_model_catalog, resolve_model_config, safe_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify local Gemma 4 inference setup before smoke tests.")
    parser.add_argument("--model", choices=["gemma4_2b", "gemma4_4b"], default=None)
    return parser.parse_args()


def check_runtime_packages() -> list[tuple[bool, str]]:
    results: list[tuple[bool, str]] = []
    for package_name in ["transformers", "torch", "accelerate"]:
        installed = importlib.util.find_spec(package_name) is not None
        if installed:
            results.append((True, f"PASS {package_name} is installed."))
        else:
            results.append((False, f"FAIL {package_name} is not installed. Install it in the active Python environment."))
    return results


def check_model_config_file() -> tuple[bool, str]:
    if LOCAL_MODEL_CONFIG_FILE.exists():
        return True, f"PASS local model config found: {LOCAL_MODEL_CONFIG_FILE}"
    if EXAMPLE_MODEL_CONFIG_FILE.exists():
        return False, (
            "FAIL models.local.json is missing. Copy "
            f"{EXAMPLE_MODEL_CONFIG_FILE.name} to {LOCAL_MODEL_CONFIG_FILE.name} and update model settings."
        )
    return False, "FAIL model config files are missing."


def check_transformers_model(model_id: str, model_config: dict[str, object]) -> tuple[bool, str]:
    required_fields = [
        "hf_model_id",
        "processor_id",
        "torch_dtype",
        "device_map",
        "max_input_tokens",
        "max_output_tokens",
    ]
    missing = [field for field in required_fields if not safe_text(model_config.get(field))]
    if missing:
        return False, f"FAIL {model_id} transformers config is missing fields: {missing}"

    local_dir = safe_text(model_config.get("local_dir"))
    if local_dir:
        local_path = Path(local_dir)
        if not local_path.exists():
            return False, f"FAIL {model_id} local snapshot directory not found: {local_path}"
        return True, (
            f"PASS {model_id} config resolved. runtime=transformers hf_model_id={model_config.get('hf_model_id')} "
            f"local_dir={local_path}"
        )

    return True, (
        f"PASS {model_id} config resolved. runtime=transformers hf_model_id={model_config.get('hf_model_id')} "
        "local_dir=<huggingface-cache>"
    )


def check_llama_cpp_model(model_id: str, model_config: dict[str, object]) -> tuple[bool, str]:
    required_fields = ["model_path", "n_ctx", "n_gpu_layers", "chat_format"]
    missing = [field for field in required_fields if field not in model_config or not safe_text(model_config.get(field))]
    if missing:
        return False, f"FAIL {model_id} llama_cpp config is missing fields: {missing}"

    model_path = Path(str(model_config["model_path"]))
    if not model_path.exists():
        return False, f"FAIL {model_id} model file not found: {model_path}"

    return True, (
        f"PASS {model_id} config resolved. runtime=llama_cpp quantization={model_config.get('quantization')} "
        f"model_path={model_path}"
    )


def check_model_entry(model_id: str) -> tuple[bool, str]:
    try:
        model_config = resolve_model_config(model_id)
    except Exception as exc:
        return False, f"FAIL {model_id} config could not be resolved: {exc}"

    runtime = str(model_config.get("runtime"))
    if runtime == "transformers":
        return check_transformers_model(model_id, model_config)
    if runtime == "llama_cpp":
        return check_llama_cpp_model(model_id, model_config)
    return False, f"FAIL {model_id} has unsupported runtime: {runtime}"


def main() -> None:
    args = parse_args()
    selected_models = [args.model] if args.model else ["gemma4_2b", "gemma4_4b"]

    overall_ok = True
    checks: list[tuple[bool, str]] = [check_model_config_file(), *check_runtime_packages()]

    for ok, message in checks:
        print(message)
        overall_ok = overall_ok and ok

    try:
        catalog = load_model_catalog()
        print(f"INFO default_model_id={catalog.get('default_model_id')}")
    except Exception as exc:
        print(f"INFO model catalog not loaded: {exc}")
        overall_ok = False

    for model_id in selected_models:
        ok, message = check_model_entry(model_id)
        print(message)
        overall_ok = overall_ok and ok

    if overall_ok:
        print("READY local inference smoke tests can run.")
        raise SystemExit(0)

    print("NOT_READY fix the failed items above before running local smoke tests.")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
