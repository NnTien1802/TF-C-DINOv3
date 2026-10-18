"""TF-C with both encoders replaced by pretrained DINOv3 ViT-S/16.

CHANGED FILE: code/TFC/model.py (only)
The 1D-to-image bridge is intentionally minimal for a CODE-ONLY architecture
submission. It is not a validated representation for scientific experiments.
No training or runtime validation is claimed.
"""

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel


# [UPDATED 1] Official pretrained DINOv3 backbone identifier.
DINOV3_MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"


class SignalToDINOImage(nn.Module):
    """[NEW] Minimal shape adapter: [B, 1, L] -> [B, 3, 224, 224].

    The original TF-C loader produces 1D time-domain and FFT-magnitude data.
    This bridge lets model.py keep the original `TFC.forward(x_t, x_f)` API.
    It replicates the resized 1D signal as RGB rows; it is a *placeholder
    representation*, not a spectrogram/GAF or a validated feature design.
    """

    def __init__(self, image_size=224):
        super().__init__()
        self.image_size = image_size
        self.register_buffer(
            "image_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, x):
        if x.ndim == 2:                    # [B, L]
            x = x.unsqueeze(1)
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError("Expected TF-C tensor of shape [B, 1, L].")

        x = x.float()
        low = x.amin(dim=-1, keepdim=True)
        high = x.amax(dim=-1, keepdim=True)
        x = (x - low) / (high - low).clamp_min(1e-6)
        x = x.unsqueeze(2)                # [B, 1, 1, L]
        x = F.interpolate(x, size=(self.image_size, self.image_size),
                          mode="bilinear", align_corners=False)
        x = x.expand(-1, 3, -1, -1)       # [B, 3, 224, 224]
        return (x - self.image_mean) / self.image_std


"""Two contrastive encoders -- DINOv3 replaces both Transformer encoders."""
class TFC(nn.Module):
    def __init__(self, configs):
        super(TFC, self).__init__()

        # [UPDATED 2] Replace original Time TransformerEncoder with DINOv3.
        self.dinov3_time = AutoModel.from_pretrained(DINOV3_MODEL_ID)

        # [UPDATED 3] Replace original Frequency TransformerEncoder with DINOv3.
        # The pretrained download is cached, but the two instances are separate.
        self.dinov3_freq = AutoModel.from_pretrained(DINOV3_MODEL_ID)

        # [NEW] Adapters for original 1D dataloader tensors.
        self.time_to_image = SignalToDINOImage()
        self.freq_to_image = SignalToDINOImage()

        # [UPDATED 4] DINOv3 ViT-S features are 384D, not TSlength_aligned.
        d_time = self.dinov3_time.config.hidden_size
        d_freq = self.dinov3_freq.config.hidden_size

        # Original time projector layout retained; only input dimension changed.
        self.projector_t = nn.Sequential(
            nn.Linear(d_time, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        # Original frequency projector layout retained; only input dimension changed.
        self.projector_f = nn.Sequential(
            nn.Linear(d_freq, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, x_in_t, x_in_f):
        # [UPDATED 5] Time feature: 1D -> 2D RGB -> DINOv3 CLS embedding.
        time_image = self.time_to_image(x_in_t)
        time_tokens = self.dinov3_time(pixel_values=time_image).last_hidden_state
        h_time = time_tokens[:, 0, :]                 # [B, 384]
        z_time = self.projector_t(h_time)              # [B, 128]

        # [UPDATED 6] Frequency feature: FFT magnitude -> image -> DINOv3.
        freq_image = self.freq_to_image(x_in_f)
        freq_tokens = self.dinov3_freq(pixel_values=freq_image).last_hidden_state
        h_freq = freq_tokens[:, 0, :]                 # [B, 384]
        z_freq = self.projector_f(h_freq)              # [B, 128]

        # [KEPT] Keep trainer.py interface unchanged.
        return h_time, z_time, h_freq, z_freq


"""Downstream classifier only used in finetuning -- UNCHANGED."""
class target_classifier(nn.Module):
    def __init__(self, configs):
        super(target_classifier, self).__init__()
        self.logits = nn.Linear(2*128, 64)
        self.logits_simple = nn.Linear(64, configs.num_classes_target)

    def forward(self, emb):
        emb_flat = emb.reshape(emb.shape[0], -1)
        emb = torch.sigmoid(self.logits(emb_flat))
        pred = self.logits_simple(emb)
        return pred
