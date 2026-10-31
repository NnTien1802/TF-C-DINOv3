"""Time/frequency adaptation of core DINOv3 losses: DINO CLS, iBOT, KoLeo.

Unlike official DINOv3's enormous distributed image pipeline, this lightweight
version uses a local EMA teacher, per-batch Sinkhorn-Knopp assignments, masked
patch tokens, and no optional Gram teacher. NOT an exact reproduction of Meta's
training setup. Source concepts: facebookresearch/dinov3/dinov3/loss/.
"""

import torch
import torch.nn.functional as F
from torch import nn


@torch.no_grad()
def sinkhorn_knopp_teacher(logits, teacher_temp=0.04, iterations=3, eps=1e-6):
    """Numerically stable, non-distributed balanced assignments, (N,K)->(N,K)."""
    if logits.ndim != 2:
        raise ValueError("Teacher logits must be N x prototypes")
    # Computationally affordable adaptation to smaller, local batch sizes.
    q = (logits.detach().float() / teacher_temp).T
    q = torch.exp(q - q.max())
    q = q / q.sum().clamp_min(eps)
    k, n = q.shape
    for _ in range(iterations):
        q = q / q.sum(dim=1, keepdim=True).clamp_min(eps)
        q = q / k
        q = q / q.sum(dim=0, keepdim=True).clamp_min(eps)
        q = q / n
    q = (q * n).T
    return q / q.sum(dim=-1, keepdim=True).clamp_min(eps)


def dino_cross_entropy(student_logits, teacher_probs, student_temp=0.1):
    """Stop-gradient teacher distribution supervising student prototypes."""
    return -(teacher_probs.detach() * F.log_softmax(
        student_logits.float() / student_temp, dim=-1
    )).sum(dim=-1).mean()


def koleo_loss(student_embeddings, eps=1e-8):
    """Non-distributed Kozachenko-Leonenko nearest-neighbour regularization."""
    if student_embeddings.size(0) < 2:
        return student_embeddings.sum() * 0.0
    z = F.normalize(student_embeddings.float(), dim=-1, eps=eps)
    with torch.no_grad():
        cosine = z @ z.T
        cosine.fill_diagonal_(-float("inf"))
        indices = cosine.argmax(dim=1)
    distances = torch.linalg.vector_norm(z - z[indices], dim=1)
    return -torch.log(distances.clamp_min(eps)).mean()


class DINOv3AdaptedLoss(nn.Module):
    """DINO CLS + iBOT masked patch distillation + KoLeo, for TF-C.

    dino: student augmented/patch-masked CLS versus original teacher CLS,
          both within each domain and across time/frequency.
    ibot: compare masked student patch predictions to unmasked teacher patches
          WITHIN the same domain (frequency bins != time positions).
    koleo: encourage diversity of student projection embeddings.
    """

    def __init__(self, *, student_temp=0.1, teacher_temp=0.04,
                 dino_weight=1.0, ibot_weight=1.0, koleo_weight=0.1,
                 cross_domain_weight=0.25, sinkhorn_iterations=3):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.dino_weight = dino_weight
        self.ibot_weight = ibot_weight
        self.koleo_weight = koleo_weight
        self.cross_domain_weight = cross_domain_weight
        self.sinkhorn_iterations = sinkhorn_iterations

    def forward(self, student, teacher, masks):
        teacher_cls = {}
        teacher_patches = {}
        for name in ("time", "freq"):
            with torch.no_grad():
                teacher_cls[name] = sinkhorn_knopp_teacher(
                    teacher[name]["dino_logits"], self.teacher_temp,
                    self.sinkhorn_iterations
                )
                # Only teacher patches at student-masked locations are targets.
                teacher_patches[name] = sinkhorn_knopp_teacher(
                    teacher[name]["ibot_logits"][masks[name]],
                    self.teacher_temp, self.sinkhorn_iterations
                )

        dino_within = 0.5 * (
            dino_cross_entropy(student["time"]["dino_logits"], teacher_cls["time"], self.student_temp)
            + dino_cross_entropy(student["freq"]["dino_logits"], teacher_cls["freq"], self.student_temp)
        )
        dino_cross = 0.5 * (
            dino_cross_entropy(student["time"]["dino_logits"], teacher_cls["freq"], self.student_temp)
            + dino_cross_entropy(student["freq"]["dino_logits"], teacher_cls["time"], self.student_temp)
        )
        ibot = 0.5 * (
            dino_cross_entropy(student["time"]["ibot_logits"][masks["time"]],
                               teacher_patches["time"], self.student_temp)
            + dino_cross_entropy(student["freq"]["ibot_logits"][masks["freq"]],
                                 teacher_patches["freq"], self.student_temp)
        )
        koleo = 0.5 * (
            koleo_loss(student["time"]["z"]) + koleo_loss(student["freq"]["z"])
        )
        total = (self.dino_weight * (dino_within + self.cross_domain_weight * dino_cross)
                 + self.ibot_weight * ibot + self.koleo_weight * koleo)
        return total, {
            "dino_within": dino_within.detach(), "dino_cross": dino_cross.detach(),
            "ibot": ibot.detach(), "koleo": koleo.detach(),
        }


def random_patch_mask(batch_size, n_patches, ratio, device):
    """Fixed count per sample; ensure at least one masked patch and one visible."""
    if n_patches < 2:
        raise ValueError("iBOT needs at least two patches")
    if not 0 < ratio < 1:
        raise ValueError("mask_ratio must lie strictly between zero and one")
    n_masked = max(1, min(n_patches - 1, round(n_patches * ratio)))
    noise = torch.rand(batch_size, n_patches, device=device)
    indices = noise.argsort(dim=1)[:, :n_masked]
    return torch.zeros(batch_size, n_patches, device=device, dtype=torch.bool).scatter_(1, indices, True)
