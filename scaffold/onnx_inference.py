"""ONNX export and inference for the detector model.

Exports a PyTorch sequence-classification model to ONNX and runs inference
via ONNX Runtime for low latency (<10ms).

Class structure:
- ONNXExporter: single responsibility to export a PyTorch model to ONNX.
- ONNXDetector: holds a loaded ONNX session and runs inference (load via .load()).
"""

import os
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union

import torch


# ── Protocols (Interface Segregation: depend on minimal interfaces) ──


class TokenizerProtocol(Protocol):
    """Minimal tokenizer interface needed for export and inference."""

    def __call__(
        self,
        text: Union[str, List[str]],
        *,
        return_tensors: str,
        truncation: bool,
        max_length: int,
        padding: Union[bool, str],
    ) -> Any: ...


# ── Exporter (Single Responsibility: only export) ──


class ONNXExporter:
    """Exports a PyTorch sequence-classification model to an ONNX file."""

    DEFAULT_OPSET = 14

    def __init__(self, opset_version: int = DEFAULT_OPSET):
        self.opset_version = opset_version

    def export(
        self,
        model: torch.nn.Module,
        tokenizer: TokenizerProtocol,
        onnx_path: str,
        model_dir: str,
        max_length: int = 512,
    ) -> None:
        """Write the model to onnx_path. Creates model_dir if needed."""
        dummy = self._dummy_inputs(tokenizer, max_length)
        args = dummy["args"]
        # Move dummy inputs to the model's device (e.g. CUDA) to avoid device mismatch during trace
        device = next(model.parameters()).device
        args = tuple(t.to(device) if torch.is_tensor(t) else t for t in args)
        input_names = dummy["input_names"]
        dynamic_axes = dummy["dynamic_axes"]

        os.makedirs(model_dir, exist_ok=True)
        with torch.no_grad():
            torch.onnx.export(
                model,
                args,
                onnx_path,
                input_names=input_names,
                output_names=["logits"],
                dynamic_axes=dynamic_axes,
                opset_version=self.opset_version,
                do_constant_folding=True,
            )
        print(f"Exported ONNX to {onnx_path}.")

    def _dummy_inputs(
        self,
        tokenizer: TokenizerProtocol,
        max_length: int,
    ) -> Dict[str, Any]:
        """Build dummy tokenizer outputs and ONNX dynamic_axes. Isolated for testability."""
        enc = tokenizer(
            "dummy",
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        if "token_type_ids" in enc:
            token_type_ids = enc["token_type_ids"]
            args = (input_ids, attention_mask, token_type_ids)
            input_names = ["input_ids", "attention_mask", "token_type_ids"]
            dynamic_axes = {
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "token_type_ids": {0: "batch", 1: "sequence"},
                "logits": {0: "batch"},
            }
        else:
            args = (input_ids, attention_mask)
            input_names = ["input_ids", "attention_mask"]
            dynamic_axes = {
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "logits": {0: "batch"},
            }
        return {"args": args, "input_names": input_names, "dynamic_axes": dynamic_axes}


# ── Session loader (Single Responsibility: load session + providers) ──


class ONNXSessionLoader:
    """Loads an ONNX Runtime session from file. Returns None when unavailable."""

    def __init__(self, use_cuda: bool):
        self._use_cuda = use_cuda
        self._ort = self._import_ort()

    def _import_ort(self) -> Any:
        try:
            import onnxruntime as ort
            return ort
        except ImportError:
            return None

    def load(self, onnx_path: str) -> Optional[Tuple[Any, List[str]]]:
        """Load session and input names. Returns (session, input_names) or None."""
        if self._ort is None or not os.path.isfile(onnx_path):
            return None
        try:
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if self._use_cuda
                else ["CPUExecutionProvider"]
            )
            session = self._ort.InferenceSession(onnx_path, providers=providers)
            in_use = session.get_providers()
            if self._use_cuda and "CUDAExecutionProvider" not in in_use:
                print("ONNX: CUDA not available, using CPU (install onnxruntime-gpu for GPU).")
            elif in_use:
                print(f"ONNX providers: {in_use}")
            input_names = [inp.name for inp in session.get_inputs()]
            return session, input_names
        except Exception as e:
            print(f"ONNX load failed ({e}), using PyTorch.")
            return None


# ── Detector (Single Responsibility: run inference) ──


class ONNXDetector:
    """Runs inference using a loaded ONNX session. Create via ONNXDetector.load()."""

    def __init__(self, session: Any, input_names: List[str]):
        self._session = session
        self._input_names = input_names

    @classmethod
    def load(cls, onnx_path: str, use_cuda: bool) -> Optional["ONNXDetector"]:
        """Load from onnx_path. Returns an instance or None if unavailable."""
        loader = ONNXSessionLoader(use_cuda=use_cuda)
        result = loader.load(onnx_path)
        if result is None:
            return None
        session, input_names = result
        return cls(session=session, input_names=input_names)

    def run(
        self,
        tokenizer: TokenizerProtocol,
        texts: List[str],
        max_length: int,
    ) -> List[float]:
        """Run inference on texts. Returns one raw logit per text."""
        if not texts:
            return []
        enc = tokenizer(
            texts,
            return_tensors="np",
            truncation=True,
            max_length=max_length,
            padding=True,
        )
        feed = {name: enc[name] for name in self._input_names if name in enc}
        logits = self._session.run(None, feed)[0]
        return [float(logits[i].squeeze()) for i in range(len(texts))]


# ── Convenience (backward compatibility / simple entrypoints) ──


def export_onnx(
    model: Any,
    tokenizer: Any,
    onnx_path: str,
    model_dir: str,
    max_length: int = 512,
) -> None:
    """Export model to ONNX. Delegates to ONNXExporter."""
    ONNXExporter().export(model, tokenizer, onnx_path, model_dir, max_length)


def load_session(onnx_path: str, use_cuda: bool) -> Union[Tuple[Any, List[str]], Tuple[None, None]]:
    """Load ONNX session and input names. Returns (session, input_names) or (None, None)."""
    loader = ONNXSessionLoader(use_cuda=use_cuda)
    result = loader.load(onnx_path)
    if result is None:
        return None, None
    return result
