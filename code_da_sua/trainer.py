"""Two-stage TF-C training with DINOv3-style SSL objective and EMA teacher.

Pre-train: (DINO CLS + iBOT masked patch + KoLeo) on source train only.
Fine-tune: supervised CrossEntropy on target train, optional low-weight SSL.
Checkpoint selection: target validation only; target test evaluated once.
"""
from copy import deepcopy
from pathlib import Path
import csv
import torch
from torch import nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from loss import DINOv3AdaptedLoss, random_patch_mask


def make_ssl_loss(args):
    return DINOv3AdaptedLoss(
        student_temp=args.student_temp, teacher_temp=args.teacher_temp,
        dino_weight=args.dino_weight, ibot_weight=args.ibot_weight,
        koleo_weight=args.koleo_weight, cross_domain_weight=args.cross_domain_weight,
    )


def create_ema_teacher(model):
    teacher = deepcopy(model)
    teacher.requires_grad_(False)
    teacher.eval()
    return teacher


@torch.no_grad()
def update_ema_teacher(teacher, student, momentum):
    for t, s in zip(teacher.parameters(), student.parameters()):
        t.lerp_(s.detach(), 1.0 - momentum)
    for t, s in zip(teacher.buffers(), student.buffers()):
        t.copy_(s)


def ssl_batch_loss(model, teacher, batch, device, loss_fn, mask_ratio):
    x_t, _, x_t_aug, x_f, x_f_aug = batch
    x_t, x_t_aug = x_t.to(device), x_t_aug.to(device)
    x_f, x_f_aug = x_f.to(device), x_f_aug.to(device)
    mask_t = random_patch_mask(x_t.size(0), model.num_patches, mask_ratio, device)
    mask_f = random_patch_mask(x_f.size(0), model.num_patches, mask_ratio, device)
    student_out = model.forward_ssl(x_t_aug, x_f_aug, mask_t=mask_t, mask_f=mask_f)
    with torch.no_grad():
        teacher_out = teacher.forward_ssl(x_t, x_f)
    return loss_fn(student_out, teacher_out, {"time": mask_t, "freq": mask_f})


def pretrain_epoch(model, teacher, optimizer, loader, device, loss_fn, args):
    model.train()
    teacher.eval()
    totals = []
    components = {"dino_within": [], "dino_cross": [], "ibot": [], "koleo": []}
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)  # fixed: must be inside batch loop
        loss, terms = ssl_batch_loss(model, teacher, batch, device, loss_fn, args.mask_ratio)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite SSL loss: {loss.item()}")
        loss.backward()
        nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        update_ema_teacher(teacher, model, args.ema_momentum)
        totals.append(loss.item())
        for key in components:
            components[key].append(terms[key].item())
    if not totals:
        raise RuntimeError("Empty pre-training loader")
    return {"loss": sum(totals) / len(totals), **{
        name: sum(vals) / len(vals) for name, vals in components.items()
    }}


def finetune_epoch(model, classifier, teacher, optimizer, classifier_optimizer,
                   loader, device, loss_fn, args):
    model.train()
    classifier.train()
    if teacher is not None:
        teacher.eval()
    losses = []
    for batch in loader:
        x_t, y, _, x_f, _ = batch
        x_t, y, x_f = x_t.to(device), y.to(device), x_f.to(device)
        optimizer.zero_grad(set_to_none=True)
        classifier_optimizer.zero_grad(set_to_none=True)
        _, z_t, _, z_f = model(x_t, x_f)
        logits = classifier(torch.cat((z_t, z_f), dim=-1))
        loss_ce = nn.functional.cross_entropy(logits, y)
        loss = loss_ce
        if args.finetune_ssl_weight > 0:
            ssl, _ = ssl_batch_loss(model, teacher, batch, device, loss_fn, args.mask_ratio)
            loss = loss + args.finetune_ssl_weight * ssl
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite fine-tune loss: {loss.item()}")
        loss.backward()
        nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        classifier_optimizer.step()
        if teacher is not None:
            update_ema_teacher(teacher, model, args.ema_momentum)
        losses.append(loss.item())
    if not losses:
        raise RuntimeError("Empty target training loader")
    return sum(losses) / len(losses)


@torch.no_grad()
def evaluate(model, classifier, loader, device, num_classes):
    model.eval()
    classifier.eval()
    true, pred = [], []
    total_loss, n = 0.0, 0
    for x_t, y, _, x_f, _ in loader:
        x_t, y, x_f = x_t.to(device), y.to(device), x_f.to(device)
        _, z_t, _, z_f = model(x_t, x_f)
        logits = classifier(torch.cat((z_t, z_f), dim=-1))
        total_loss += nn.functional.cross_entropy(logits, y, reduction="sum").item()
        n += y.numel()
        true.extend(y.cpu().tolist())
        pred.extend(logits.argmax(dim=-1).cpu().tolist())
    if n == 0:
        raise RuntimeError("Empty evaluation loader")
    precision, recall, f1, _ = precision_recall_fscore_support(
        true, pred, labels=list(range(num_classes)), average="macro", zero_division=0
    )
    return {
        "loss": total_loss / n,
        "accuracy": accuracy_score(true, pred),
        "precision_macro": precision, "recall_macro": recall, "f1_macro": f1,
        "confusion_matrix": confusion_matrix(true, pred, labels=list(range(num_classes))).tolist(),
    }


def Trainer(model, model_optimizer, classifier, classifier_optimizer,
            pretrain_dl, target_train_dl, target_val_dl, target_test_dl,
            device, logger, config, experiment_log_dir, training_mode, args):
    root = Path(experiment_log_dir)
    root.mkdir(parents=True, exist_ok=True)
    ckpt_dir = root / "saved_models"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = root / "epoch_metrics.csv"
    loss_fn = make_ssl_loss(args).to(device)

    if training_mode == "pre_train":
        teacher = create_ema_teacher(model)
        rows = []
        for epoch in range(1, config.num_epoch + 1):
            row = pretrain_epoch(model, teacher, model_optimizer, pretrain_dl, device, loss_fn, args)
            row = {"epoch": epoch, **row}
            rows.append(row)
            logger.info("Pretrain epoch %d/%d: %s", epoch, config.num_epoch, row)
        torch.save({"model_state_dict": model.state_dict(),
                    "teacher_state_dict": teacher.state_dict(),
                    "training_mode": "pre_train", "model_id": args.model_id,
                    "image_size": args.image_size, "prototypes": args.prototypes},
                   ckpt_dir / "ckp_last.pt")
    elif training_mode == "fine_tune_test":
        teacher = create_ema_teacher(model) if args.finetune_ssl_weight > 0 else None
        best_score = -1.0
        rows = []
        best_file = ckpt_dir / "ckp_best.pt"
        for epoch in range(1, config.num_epoch + 1):
            train_loss = finetune_epoch(model, classifier, teacher, model_optimizer,
                                       classifier_optimizer, target_train_dl, device,
                                       loss_fn, args)
            val = evaluate(model, classifier, target_val_dl, device, config.num_classes_target)
            row = {"epoch": epoch, "train_loss": train_loss,
                   **{f"val_{k}": v for k, v in val.items() if k != "confusion_matrix"}}
            rows.append(row)
            logger.info("Finetune epoch %d/%d: %s", epoch, config.num_epoch, row)
            if val["f1_macro"] > best_score:
                best_score = val["f1_macro"]
                torch.save({"model_state_dict": model.state_dict(),
                            "classifier_state_dict": classifier.state_dict(),
                            "best_val_f1": best_score, "epoch": epoch,
                            "model_id": args.model_id, "image_size": args.image_size,
                            "prototypes": args.prototypes}, best_file)

        # No test leakage: run target test only once, after selecting best val F1.
        best = torch.load(best_file, map_location=device, weights_only=True)
        model.load_state_dict(best["model_state_dict"])
        classifier.load_state_dict(best["classifier_state_dict"])
        result = evaluate(model, classifier, target_test_dl, device, config.num_classes_target)
        logger.info("Held-out test (best validation checkpoint): %s", result)
        with open(root / "test_metrics.txt", "w", encoding="utf-8") as f:
            f.write(f"Best validation F1: {best_score:.6f}\nHeld-out test: {result}\n")
    else:
        raise ValueError(f"Unknown training_mode: {training_mode}")

    if rows:
        with metrics_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return rows
