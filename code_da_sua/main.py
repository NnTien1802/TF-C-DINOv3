"""CLI for TF-C with two DINOv3 ViT encoders and DINO+iBOT+KoLeo losses."""
import argparse
import importlib
import logging
import os
from pathlib import Path
import random
import numpy as np
import torch
from dataloader import data_generator
from model import TFC, target_classifier
from trainer import Trainer


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="TF-C / DINOv3 student + EMA teacher")
    p.add_argument("--training_mode", choices=["pre_train", "fine_tune_test"], default="pre_train")
    p.add_argument("--pretrain_dataset", default="HAR")
    p.add_argument("--target_dataset", default="SGR")
    p.add_argument("--dataset_root", default="datasets")
    p.add_argument("--logs_save_dir", default="experiments_logs")
    p.add_argument("--run_description", default="dinov3_loss")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--target_batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--model_id", default="facebook/dinov3-vits16-pretrain-lvd1689m")
    p.add_argument("--image_size", type=int, default=112)
    p.add_argument("--prototypes", type=int, default=256)
    p.add_argument("--unfreeze_backbone", action="store_true", help="High VRAM cost; default: frozen ViT")
    p.add_argument("--teacher_temp", type=float, default=0.04)
    p.add_argument("--student_temp", type=float, default=0.1)
    p.add_argument("--ema_momentum", type=float, default=0.996)
    p.add_argument("--mask_ratio", type=float, default=0.3)
    p.add_argument("--dino_weight", type=float, default=1.0)
    p.add_argument("--ibot_weight", type=float, default=1.0)
    p.add_argument("--koleo_weight", type=float, default=0.1)
    p.add_argument("--cross_domain_weight", type=float, default=0.25)
    p.add_argument("--finetune_ssl_weight", type=float, default=0.0,
                   help="Optional DINOv3 SSL regularizer while training downstream classifier")
    p.add_argument("--subset", action="store_true", help="Debug with small dataset slices")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    config_module = importlib.import_module(f"config_files.{args.pretrain_dataset}_Configs")
    configs = config_module.Config()
    configs.num_epoch = args.epochs
    configs.batch_size = args.batch_size
    configs.target_batch_size = args.target_batch_size
    configs.lr = args.lr
    configs.time_jitter_sigma = 0.08
    configs.freq_mask_ratio = 0.1

    source_dir = Path(args.dataset_root) / args.pretrain_dataset
    target_dir = Path(args.dataset_root) / args.target_dataset
    pretrain_dl, target_train_dl, target_val_dl, target_test_dl = data_generator(
        source_dir, target_dir, configs, args.training_mode, subset=args.subset
    )
    # Infer class count directly from target's labeled training split.
    target_labels = target_train_dl.dataset.y
    configs.num_classes_target = int(target_labels.max().item()) + 1

    experiment_root = Path(args.logs_save_dir) / f"{args.pretrain_dataset}_2_{args.target_dataset}" / args.run_description
    output_dir = experiment_root / f"{args.training_mode}_seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(output_dir / "train.log", encoding="utf-8")])
    logger = logging.getLogger("tfc_dinov3")
    logger.info("Using device %s; target classes %s", device, configs.num_classes_target)
    model = TFC(configs, model_id=args.model_id, image_size=args.image_size,
                freeze_backbone=not args.unfreeze_backbone,
                prototypes=args.prototypes).to(device)
    classifier = target_classifier(configs).to(device)

    if args.training_mode == "fine_tune_test":
        checkpoint = experiment_root / f"pre_train_seed_{args.seed}" / "saved_models" / "ckp_last.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Pretraining checkpoint not found: {checkpoint}. Run --training_mode pre_train first.")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state.get("model_id") != args.model_id or state.get("image_size") != args.image_size or state.get("prototypes") != args.prototypes:
            raise ValueError("Checkpoint architecture differs from current model flags")
        model.load_state_dict(state["model_state_dict"], strict=True)
        logger.info("Loaded pretrained student checkpoint %s", checkpoint)

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters")
    model_optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    classifier_optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.lr, weight_decay=1e-4)
    Trainer(model, model_optimizer, classifier, classifier_optimizer,
            pretrain_dl, target_train_dl, target_val_dl, target_test_dl,
            device, logger, configs, output_dir, args.training_mode, args)


if __name__ == "__main__":
    main()
