"""RF-DETR model adapter isolating all rfdetr-specific logic.

This module implements the adapter pattern to wrap all RF-DETR model
interactions behind a stable interface. If the upstream rfdetr package
changes its API, only this file needs updating. No rfdetr or supervision
types leak beyond this boundary.

Module-level functions (detect_device, validate_tile_input, prepare_tile_image)
are available for independent use outside the adapter class.
"""

from __future__ import annotations

import os
import warnings
from typing import Any

import numpy as np
from numpy.typing import NDArray

from detr_geo._typing import DetectionResult, ModelSize
from detr_geo.exceptions import BandError, ModelError

# ---------------------------------------------------------------------------
# Module-level functions
# ---------------------------------------------------------------------------


def detect_device(preferred: str | None = None) -> str:
    """Detect the best available compute device.

    Priority: CUDA > MPS > CPU. If a preferred device is specified but
    unavailable, falls back with a warning.

    Args:
        preferred: Requested device string. If None, auto-detect.

    Returns:
        Device string: "cuda", "mps", or "cpu".
    """
    if preferred == "cpu":
        return "cpu"

    try:
        import torch

        cuda_available = torch.cuda.is_available()
        mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    except ImportError:
        cuda_available = False
        mps_available = False

    if preferred is not None:
        if preferred == "cuda" and not cuda_available:
            warnings.warn(
                f"Requested device '{preferred}' is not available. Falling back to CPU.",
                RuntimeWarning,
                stacklevel=2,
            )
            return "cpu"
        if preferred == "mps" and not mps_available:
            warnings.warn(
                f"Requested device '{preferred}' is not available. Falling back to CPU.",
                RuntimeWarning,
                stacklevel=2,
            )
            return "cpu"
        return preferred

    if cuda_available:
        return "cuda"
    if mps_available:
        return "mps"
    return "cpu"


def _warn_cpu_inference(model_size: str) -> None:
    """Issue warnings when running inference on CPU.

    Args:
        model_size: The model size string being used.
    """
    msg = f"Running '{model_size}' model on CPU. Inference will be slow."
    if model_size != "nano":
        msg += " Consider using model_size='nano' for CPU inference."
    warnings.warn(msg, RuntimeWarning, stacklevel=3)


def validate_tile_input(
    image: Any,
    block_size: int,
) -> None:
    """Validate a tile image before inference.

    Checks channel count, dtype, value range, and dimension alignment.

    Args:
        image: PIL Image or numpy ndarray to validate.
        block_size: Model's block_size for dimension alignment check.

    Raises:
        BandError: If channel count is wrong.
        ModelError: If float32 values are out of range.
    """
    if image is None:
        return

    # Handle PIL Image
    if hasattr(image, "mode"):
        if image.mode == "RGBA":
            raise BandError(
                "Image has 4 channels (RGBA). RF-DETR expects 3-channel RGB images. "
                "Strip the alpha channel first: image.convert('RGB')"
            )
        if image.mode == "L":
            raise BandError(
                "Image has 1 channel (grayscale). RF-DETR expects 3-channel RGB images. "
                "Triplicate the band: image.convert('RGB')"
            )
        # Check dimensions for alignment warning
        w, h = image.size
        if w % block_size != 0 or h % block_size != 0:
            warnings.warn(
                f"Image dimensions ({w}x{h}) are not divisible by block_size={block_size}. "
                f"This may affect inference performance.",
                RuntimeWarning,
                stacklevel=2,
            )
        return

    # Handle numpy array
    if hasattr(image, "shape"):
        arr = np.asarray(image)

        # Check channel count - support both (H, W, C) and (C, H, W)
        if arr.ndim == 3:
            if arr.shape[-1] == 4:
                raise BandError(
                    "Image has 4 channels (RGBA). RF-DETR expects 3-channel RGB images. "
                    "Remove the alpha channel before prediction."
                )
            if arr.shape[-1] == 1 or arr.shape[0] == 1:
                raise BandError(
                    "Image has 1 channel (grayscale). RF-DETR expects 3-channel RGB images. "
                    "Triplicate the channel: np.repeat(arr, 3, axis=0)"
                )

        # Check float32 value range
        if arr.dtype == np.float32:
            max_val = arr.max()
            if max_val > 1.0:
                raise ModelError(
                    f"Float32 image has values up to {max_val:.1f}, but RF-DETR expects "
                    f"values in [0, 1]. Normalize by dividing by 255.0 or the appropriate "
                    f"max value."
                )

        # Check float64 (warn about conversion overhead)
        if arr.dtype == np.float64:
            warnings.warn(
                "Image is float64. Will be converted to float32 for inference. "
                "Consider converting beforehand to avoid overhead.",
                RuntimeWarning,
                stacklevel=2,
            )

        # Check dimension alignment
        if arr.ndim >= 2:
            h, w = arr.shape[-2], arr.shape[-1]
            if arr.ndim == 3 and arr.shape[0] <= 4:
                # (C, H, W) format
                h, w = arr.shape[1], arr.shape[2]
            if w % block_size != 0 or h % block_size != 0:
                warnings.warn(
                    f"Image dimensions ({w}x{h}) are not divisible by block_size={block_size}. "
                    f"This may affect inference performance.",
                    RuntimeWarning,
                    stacklevel=2,
                )


def prepare_tile_image(tile_array: NDArray) -> Any:
    """Convert preprocessed tile array to PIL Image for RF-DETR.

    Args:
        tile_array: Array of shape (3, H, W) with float32 values in [0, 1].

    Returns:
        PIL.Image.Image in RGB mode.

    Raises:
        BandError: If array shape is wrong.
    """
    from PIL import Image

    if tile_array.ndim != 3:
        raise BandError(f"Expected 3D array (channels, height, width), got {tile_array.ndim}D array.")

    if tile_array.shape[0] != 3:
        raise BandError(
            f"Expected 3 channels in first dimension, got {tile_array.shape[0]}. Array shape should be (3, H, W)."
        )

    # Convert (3, H, W) float32 [0, 1] -> (H, W, 3) uint8 [0, 255]
    arr = np.clip(tile_array, 0.0, 1.0)
    arr = (arr * 255).astype(np.uint8)
    arr = np.transpose(arr, (1, 2, 0))  # (C, H, W) -> (H, W, C)

    return Image.fromarray(arr, mode="RGB")


# ---------------------------------------------------------------------------
# RFDETRAdapter class
# ---------------------------------------------------------------------------


class RFDETRAdapter:
    """Adapter isolating RF-DETR model details from geospatial logic.

    The adapter provides:
    - Model variant mapping (size string -> rfdetr class, resolution, block_size)
    - Lazy model loading (weights download on first predict, not at init)
    - Input validation (band count, value range)
    - Detection normalization (supervision.Detections -> DetectionResult)
    - Device detection (cuda > mps > cpu)
    - Batch inference and JIT optimization support
    """

    # Detailed variant mapping with all properties
    VARIANT_DETAILS: dict[str, dict[str, Any]] = {
        "nano": {
            "class_name": "RFDETRNano",
            "resolution": 384,
            "block_size": 32,
            "patch_size": 16,
            "num_windows": 2,
            "license": "Apache 2.0",
            "approx_params": "15M",
        },
        "small": {
            "class_name": "RFDETRSmall",
            "resolution": 512,
            "block_size": 32,
            "patch_size": 16,
            "num_windows": 2,
            "license": "Apache 2.0",
            "approx_params": "22M",
        },
        "medium": {
            "class_name": "RFDETRMedium",
            "resolution": 576,
            "block_size": 32,
            "patch_size": 16,
            "num_windows": 2,
            "license": "Apache 2.0",
            "approx_params": "25M",
        },
        "base": {
            "class_name": "RFDETRBase",
            "resolution": 560,
            "block_size": 56,
            "patch_size": 14,
            "num_windows": 4,
            "license": "Apache 2.0",
            "approx_params": "29M",
        },
        "large": {
            "class_name": "RFDETRLarge",
            "resolution": 704,
            "block_size": 32,
            "patch_size": 16,
            "num_windows": 2,
            "license": "Apache 2.0",
            "approx_params": "30M",
        },
    }

    # Backward-compatible simple variant mapping
    VARIANTS: dict[str, tuple[str, int, int]] = {
        size: (details["class_name"], details["resolution"], details["block_size"])
        for size, details in VARIANT_DETAILS.items()
    }

    UNSUPPORTED_VARIANTS: dict[str, str] = {
        "xlarge": ("XLarge uses Roboflow Platform License (non-redistributable). Use 'large' instead."),
        "2xlarge": ("2XLarge uses Roboflow Platform License (non-redistributable). Use 'large' instead."),
    }

    # Standard COCO 80-class category ID to human-readable name mapping.
    # RF-DETR returns COCO category IDs (1-90 with gaps) as "class names",
    # so we translate them to actual labels for user-facing filtering.
    COCO_LABELS: dict[int, str] = {
        1: "person",
        2: "bicycle",
        3: "car",
        4: "motorcycle",
        5: "airplane",
        6: "bus",
        7: "train",
        8: "truck",
        9: "boat",
        10: "traffic light",
        11: "fire hydrant",
        13: "stop sign",
        14: "parking meter",
        15: "bench",
        16: "bird",
        17: "cat",
        18: "dog",
        19: "horse",
        20: "sheep",
        21: "cow",
        22: "elephant",
        23: "bear",
        24: "zebra",
        25: "giraffe",
        27: "backpack",
        28: "umbrella",
        31: "handbag",
        32: "tie",
        33: "suitcase",
        34: "frisbee",
        35: "skis",
        36: "snowboard",
        37: "sports ball",
        38: "kite",
        39: "baseball bat",
        40: "baseball glove",
        41: "skateboard",
        42: "surfboard",
        43: "tennis racket",
        44: "bottle",
        46: "wine glass",
        47: "cup",
        48: "fork",
        49: "knife",
        50: "spoon",
        51: "bowl",
        52: "banana",
        53: "apple",
        54: "sandwich",
        55: "orange",
        56: "broccoli",
        57: "carrot",
        58: "hot dog",
        59: "pizza",
        60: "donut",
        61: "cake",
        62: "chair",
        63: "couch",
        64: "potted plant",
        65: "bed",
        67: "dining table",
        70: "toilet",
        72: "tv",
        73: "laptop",
        74: "mouse",
        75: "remote",
        76: "keyboard",
        77: "cell phone",
        78: "microwave",
        79: "oven",
        80: "toaster",
        81: "sink",
        82: "refrigerator",
        84: "book",
        85: "clock",
        86: "vase",
        87: "scissors",
        88: "teddy bear",
        89: "hair drier",
        90: "toothbrush",
    }

    def __init__(
        self,
        model_size: ModelSize = "medium",
        device: str | None = None,
        pretrain_weights: str | None = None,
        confidence_threshold: float = 0.5,
        custom_class_names: dict[int, str] | None = None,
    ) -> None:
        """Initialize the adapter with model configuration.

        Args:
            model_size: One of "nano", "small", "medium", "base", "large".
            device: Compute device ("cuda", "mps", "cpu"). Auto-detected if None.
            pretrain_weights: Path to custom pretrained weights. Uses default if None.
            confidence_threshold: Default confidence threshold for predictions.
            custom_class_names: Override class name mapping for fine-tuned models.
                Maps class_id (int) to label (str), e.g. ``{0: "Car", 1: "Bus"}``.
                When provided, bypasses the default COCO label lookup.

        Raises:
            ModelError: If model_size is not supported or pretrain_weights path invalid.
        """
        # Check unsupported variants first (clear license error)
        if model_size in self.UNSUPPORTED_VARIANTS:
            raise ModelError(
                f"Model size '{model_size}' is not available. "
                f"{self.UNSUPPORTED_VARIANTS[model_size]} "
                f"Valid sizes: {', '.join(sorted(self.VARIANT_DETAILS.keys()))}",
                requested_size=model_size,
            )

        # Check supported variants
        if model_size not in self.VARIANT_DETAILS:
            valid_sizes = ", ".join(sorted(self.VARIANT_DETAILS.keys()))
            raise ModelError(
                f"Unsupported model size '{model_size}'. Valid sizes: {valid_sizes}",
                requested_size=model_size,
            )

        # Validate pretrain_weights path if provided
        if pretrain_weights is not None and not os.path.exists(pretrain_weights):
            raise ModelError(
                f"Custom weights path does not exist: '{pretrain_weights}'. "
                f"Provide a valid file path or omit to use default weights."
            )

        self._model_size = model_size
        self._details = self.VARIANT_DETAILS[model_size]
        self._device = detect_device(device)
        self._pretrain_weights = pretrain_weights
        self._confidence_threshold = confidence_threshold
        self._custom_class_names = custom_class_names
        self._model: Any = None
        self._is_optimized = False

    @property
    def resolution(self) -> int:
        """Model's native square input resolution in pixels."""
        return self._details["resolution"]

    @property
    def block_size(self) -> int:
        """patch_size * num_windows -- the divisibility constraint."""
        return self._details["block_size"]

    @property
    def patch_size(self) -> int:
        """The patch size used by the model's vision transformer."""
        return self._details["patch_size"]

    @property
    def num_windows(self) -> int:
        """Number of windows used in the model's attention mechanism."""
        return self._details["num_windows"]

    @property
    def class_names(self) -> dict[int, str]:
        """Mapping of class_id to human-readable class_name.

        If ``custom_class_names`` was provided at init (for fine-tuned models),
        that mapping is returned directly.

        Otherwise, RF-DETR models store COCO category IDs (integers like 1, 2, 3)
        as their "class names". This property translates those to human-readable
        labels (e.g., "person", "bicycle", "car") so that user-facing class
        filtering like ``classes=["car"]`` works correctly.

        Returns COCO class names by default. Only available after model is loaded.
        """
        if self._custom_class_names is not None:
            return self._custom_class_names
        if self._model is None:
            return {}
        try:
            raw_names = list(self._model.class_names)
            result = {}
            for idx, raw in enumerate(raw_names):
                # RF-DETR stores COCO category IDs; translate to labels
                try:
                    coco_id = int(raw)
                    result[idx] = self.COCO_LABELS.get(coco_id, str(raw))
                except (ValueError, TypeError):
                    # Already a human-readable name (custom model)
                    result[idx] = str(raw)
            return result
        except (AttributeError, TypeError):
            return {}

    @property
    def num_select(self) -> int:
        """Max detections per image (top-K limit).

        Returns the model's num_select parameter if loaded, default 300 otherwise.
        """
        if self._model is not None:
            try:
                return self._model.num_select
            except AttributeError:
                pass
        return 300

    @property
    def is_loaded(self) -> bool:
        """Whether the model has been loaded (lazy loading check)."""
        return self._model is not None

    def valid_tile_sizes(self, min_size: int = 128, max_size: int = 2048) -> list[int]:
        """All tile sizes divisible by block_size within range.

        Args:
            min_size: Minimum tile size (inclusive).
            max_size: Maximum tile size (inclusive).

        Returns:
            Sorted list of valid tile sizes.
        """
        bs = self.block_size
        return [s for s in range(min_size, max_size + 1) if s % bs == 0]

    def model_info(self) -> dict[str, Any]:
        """Return a summary dict of model configuration.

        Returns:
            Dict with model_size, resolution, block_size, patch_size,
            num_windows, license, approx_params, device, and is_loaded.
        """
        return {
            "model_size": self._model_size,
            "resolution": self.resolution,
            "block_size": self.block_size,
            "patch_size": self.patch_size,
            "num_windows": self.num_windows,
            "license": self._details["license"],
            "approx_params": self._details["approx_params"],
            "device": self._device,
            "is_loaded": self.is_loaded,
            "confidence_threshold": self._confidence_threshold,
        }

    def _ensure_model(self) -> None:
        """Lazy-load the model on first use.

        Raises:
            ModelError: If rfdetr is not installed, with install instructions.
        """
        if self._model is not None:
            return

        try:
            import rfdetr  # noqa: F401
        except ImportError as err:
            raise ModelError("rfdetr is not installed. Install it with: pip install detr-geo[rfdetr]") from err

        class_name = self._details["class_name"]
        try:
            model_class = getattr(rfdetr, class_name)
        except AttributeError as err:
            raise ModelError(
                f"rfdetr does not have class '{class_name}'. "
                f"You may need to update rfdetr: pip install --upgrade rfdetr"
            ) from err

        # Instantiate the model
        kwargs: dict[str, Any] = {}
        if self._pretrain_weights is not None:
            kwargs["pretrain_weights"] = self._pretrain_weights

        self._model = model_class(**kwargs)

        # Warn about CPU inference
        if self._device == "cpu":
            _warn_cpu_inference(self._model_size)

    def predict_tile(
        self,
        image: Any,  # PIL.Image.Image
        threshold: float | None = None,
    ) -> DetectionResult:
        """Run inference on a single pre-processed tile image.

        Args:
            image: A PIL Image in RGB format.
            threshold: Confidence threshold override. Uses default if None.

        Returns:
            DetectionResult dict with bbox, confidence, and class_id lists.

        Raises:
            ModelError: If rfdetr is not installed.
            BandError: If image has wrong number of channels.
        """
        validate_tile_input(image, self.block_size)
        self._ensure_model()

        effective_threshold = threshold if threshold is not None else self._confidence_threshold

        # Run prediction through rfdetr
        detections = self._model.predict(image, threshold=effective_threshold)
        return self._normalize_detections(detections)

    def predict_tiles(
        self,
        images: list[Any],  # list[PIL.Image.Image]
        threshold: float | None = None,
    ) -> list[DetectionResult]:
        """Run inference on multiple tiles sequentially.

        True GPU batching is not yet supported. This method processes tiles
        one at a time using predict_tile().

        Args:
            images: List of PIL Images in RGB format.
            threshold: Confidence threshold override. Uses default if None.

        Returns:
            List of DetectionResult dicts, one per input image.
        """
        return [self.predict_tile(img, threshold=threshold) for img in images]

    def predict_batch(
        self,
        images: list[Any],  # list[PIL.Image.Image]
        threshold: float | None = None,
    ) -> list[DetectionResult]:
        """Deprecated: Use predict_tiles() instead.

        This method is sequential, not batched. Use predict_tiles() for clarity.

        Args:
            images: List of PIL Images in RGB format.
            threshold: Confidence threshold override. Uses default if None.

        Returns:
            List of DetectionResult dicts, one per input image.
        """
        warnings.warn(
            "predict_batch() is deprecated and will be removed in a future version. "
            "Use predict_tiles() instead. Note: inference is sequential, not batched.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.predict_tiles(images, threshold=threshold)

    def _normalize_detections(self, sv_detections: Any) -> DetectionResult:
        """Convert supervision.Detections to internal DetectionResult format.

        Preserves COCO non-contiguous class IDs without renumbering.

        Args:
            sv_detections: A supervision.Detections object with xyxy,
                confidence, and class_id ndarray attributes.

        Returns:
            DetectionResult dict with Python-native lists.
        """
        # Handle empty detections
        if sv_detections is None or len(sv_detections) == 0:
            return DetectionResult(bbox=[], confidence=[], class_id=[])

        # Extract arrays from supervision.Detections
        xyxy = sv_detections.xyxy  # shape: (N, 4)
        confidence = sv_detections.confidence  # shape: (N,)
        class_id = sv_detections.class_id  # shape: (N,)

        # Convert to Python-native lists
        bbox_list: list[list[float]] = []
        for box in xyxy:
            bbox_list.append([float(box[0]), float(box[1]), float(box[2]), float(box[3])])

        confidence_list = [float(c) for c in confidence] if confidence is not None else []
        class_id_list = [int(c) for c in class_id] if class_id is not None else []

        return DetectionResult(
            bbox=bbox_list,
            confidence=confidence_list,
            class_id=class_id_list,
        )

    def train(
        self,
        dataset_dir: str,
        **kwargs: Any,
    ) -> Any:
        """Train the model on a COCO-format dataset directory.

        Maintains the adapter as the sole boundary to rfdetr -- training.py
        calls adapter.train(), never adapter._model directly.

        Args:
            dataset_dir: Path to COCO-format dataset directory containing
                train/, valid/ subdirectories with images/ and
                _annotations.coco.json.
            **kwargs: Additional keyword arguments passed to rfdetr's
                model.train() (e.g., epochs, batch_size, lr).

        Returns:
            Training result from rfdetr's model.train().

        Raises:
            ModelError: If rfdetr is not installed.
        """
        self._ensure_model()
        return self._model.train(dataset_dir=dataset_dir, **kwargs)

    def auto_batch_size(self) -> int:
        """Estimate optimal batch size based on available memory.

        Returns 1 for CPU. For GPU, estimates based on available memory
        and model resolution.

        Returns:
            Recommended batch size (minimum 1).
        """
        if self._device == "cpu":
            return 1

        try:
            import torch

            if self._device == "cuda" and torch.cuda.is_available():
                free_memory, _ = torch.cuda.mem_get_info()
                resolution = self.resolution
                # Rough estimate: 3 channels * 4 bytes float32 * 8x for activations
                per_tile_bytes = resolution * resolution * 3 * 4 * 8
                usable_memory = int(free_memory * 0.7)
                batch_size = max(1, min(64, usable_memory // per_tile_bytes))
                return batch_size
        except (ImportError, RuntimeError):
            pass

        return 1

    def optimize(self, batch_size: int = 1) -> None:
        """Optimize the model for repeated inference.

        Calls rfdetr's optimize_for_inference if available, otherwise
        attempts JIT tracing.

        Args:
            batch_size: The batch size to optimize for.
        """
        self._ensure_model()
        try:
            if hasattr(self._model, "optimize_for_inference"):
                self._model.optimize_for_inference(batch_size=batch_size)
                self._is_optimized = True
            else:
                # Fallback: attempt JIT trace
                import torch

                torch.jit.trace(self._model, example_inputs=None)
                self._is_optimized = True
        except Exception:
            # Optimization is best-effort
            self._is_optimized = False

    def remove_optimization(self) -> None:
        """Clear optimized/JIT-compiled state."""
        self._is_optimized = False
        # Reload model without optimization on next use
        if self._model is not None and hasattr(self._model, "remove_optimization"):
            try:
                self._model.remove_optimization()
            except Exception:
                pass
