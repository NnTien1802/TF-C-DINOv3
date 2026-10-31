"""TF-C adapted to two independent pretrained DINOv3 ViT encoders.

The original TF-C API ``model(time, frequency) -> (h_t, z_t, h_f, z_f)``
is retained for inference. ``forward_ssl`` exposes CLS and patch embeddings
for DINO/iBOT/KoLeo student/EMA-teacher objectives.

IMPORTANT: This is a time-series adaptation of DINOv3, not Meta's full
original image-training pipeline. Its 3-axis -> image representation should be
validated experimentally.
"""

from contextlib import nullcontext
import copy
import torch
from torch import nn
import torch.nn.functional as F


class AccelerometerImageAdapter(nn.Module):
    """Map Bx3xL time series to Bx3xHxW for an image-pretrained ViT.

    Each acceleration axis forms one image colour channel; the temporal axis
    is resized along image width and repeated over height. The frequency
    branch uses log(1+abs(FFT)) before standardizing separately per axis.
    This is deliberately simple and reproducible, not a learned spectrogram.
    """

    def __init__(self, image_size=112):
        super().__init__()
        self.image_size = int(image_size)
        self.register_buffer("rgb_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("rgb_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x, frequency=False):
        if x.ndim != 3:
            raise ValueError(f"Expected BxCxL, got {tuple(x.shape)}")
        if x.size(1) != 3:
            raise ValueError(f"The DINOv3 adapter requires 3 accelerometer axes, got {x.size(1)}")
        x = x.float()
        if frequency:
            x = torch.log1p(x.abs())
        mean = x.mean(dim=-1, keepdim=True)
        scale = x.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-5)
        x = ((x - mean) / scale).clamp(-3, 3) / 6.0 + 0.5
        # Resize time/frequency bins; replicate row to get a 2D tensor.
        x = F.interpolate(x, size=self.image_size, mode="linear", align_corners=False)
        x = x.unsqueeze(2).expand(-1, -1, self.image_size, -1)
        return (x - self.rgb_mean) / self.rgb_std


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Linear(256, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class PrototypeHead(nn.Module):
    """Lightweight prototype prediction head (not Meta's exact DINOHead)."""

    def __init__(self, in_dim, prototypes=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.GELU(), nn.Linear(256, prototypes, bias=False)
        )

    def forward(self, x):
        return self.net(x)


class TFC(nn.Module):
    def __init__(self, configs=None, *, model_id="facebook/dinov3-vits16-pretrain-lvd1689m",
                 image_size=112, freeze_backbone=True, prototypes=256,
                 backbone_factory=None):
        super().__init__()
        self.image_adapter = AccelerometerImageAdapter(image_size)
        self.freeze_backbone = bool(freeze_backbone)
        self.model_id = model_id

        if backbone_factory is None:
            try:
                from transformers import DINOv3ViTModel
            except ImportError as exc:
                raise ImportError("Install transformers>=4.57.0 for DINOv3 support") from exc
            # Two physically independent encoder copies, initialized from the
            # same official pretrained checkpoint, then trained independently.
            self.encoder_t = DINOv3ViTModel.from_pretrained(model_id)
            self.encoder_f = copy.deepcopy(self.encoder_t)
        else:
            # For offline architecture/unit tests, not for final experiments.
            self.encoder_t = backbone_factory()
            self.encoder_f = copy.deepcopy(self.encoder_t)

        dim_t = int(self.encoder_t.config.hidden_size)
        dim_f = int(self.encoder_f.config.hidden_size)
        if dim_t != dim_f:
            raise ValueError("Time and frequency backbones must have equal hidden_size")
        self.patch_size = int(self.encoder_t.config.patch_size)
        if image_size % self.patch_size:
            raise ValueError(f"image_size must be divisible by patch_size={self.patch_size}")
        self.num_patches = (image_size // self.patch_size) ** 2
        self.projector_t = ProjectionHead(dim_t, 128)
        self.projector_f = ProjectionHead(dim_f, 128)
        self.dino_head_t = PrototypeHead(128, prototypes)
        self.dino_head_f = PrototypeHead(128, prototypes)
        self.ibot_head_t = PrototypeHead(dim_t, prototypes)
        self.ibot_head_f = PrototypeHead(dim_f, prototypes)
        if self.freeze_backbone:
            self.encoder_t.requires_grad_(False)
            self.encoder_f.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            # Keep frozen pretrained backbones in inference mode.
            self.encoder_t.eval()
            self.encoder_f.eval()
        return self

    def _one_branch(self, backbone, projector, x, *, frequency, masked_pos=None):
        image = self.image_adapter(x, frequency=frequency)
        kwargs = {"pixel_values": image, "return_dict": True}
        if masked_pos is not None:
            if masked_pos.shape != (x.shape[0], self.num_patches):
                raise ValueError("masked_pos has incorrect shape")
            kwargs["bool_masked_pos"] = masked_pos
        context = torch.no_grad() if self.freeze_backbone else nullcontext()
        with context:
            output = backbone(**kwargs)
        tokens = output.last_hidden_state
        if tokens.ndim != 3 or tokens.size(1) < self.num_patches + 1:
            raise ValueError(f"Unexpected DINOv3 last_hidden_state shape {tuple(tokens.shape)}")
        # Last P tokens are patch tokens; preceding tokens include CLS/registers.
        cls = tokens[:, 0, :]
        patches = tokens[:, -self.num_patches:, :]
        return {"h": cls, "z": projector(cls), "patches": patches}

    def forward_features(self, x_in_t, x_in_f, *, mask_t=None, mask_f=None):
        return {
            "time": self._one_branch(self.encoder_t, self.projector_t, x_in_t,
                                     frequency=False, masked_pos=mask_t),
            "freq": self._one_branch(self.encoder_f, self.projector_f, x_in_f,
                                     frequency=True, masked_pos=mask_f),
        }

    def forward_ssl(self, x_in_t, x_in_f, *, mask_t=None, mask_f=None):
        features = self.forward_features(x_in_t, x_in_f, mask_t=mask_t, mask_f=mask_f)
        for name, dino_head, ibot_head in (
            ("time", self.dino_head_t, self.ibot_head_t),
            ("freq", self.dino_head_f, self.ibot_head_f),
        ):
            features[name]["dino_logits"] = dino_head(features[name]["z"])
            features[name]["ibot_logits"] = ibot_head(features[name]["patches"])
        return features

    def forward(self, x_in_t, x_in_f):
        feats = self.forward_features(x_in_t, x_in_f)
        t, f = feats["time"], feats["freq"]
        return t["h"], t["z"], f["h"], f["z"]


class target_classifier(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.logits = nn.Sequential(nn.Linear(256, 64), nn.GELU(),
                                    nn.Linear(64, int(configs.num_classes_target)))

    def forward(self, emb):
        return self.logits(emb.reshape(emb.size(0), -1))
