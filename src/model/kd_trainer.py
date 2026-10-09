import warnings
from pathlib import Path

import torch
import torch.nn as nn
from .architecture import MeatGuardS
from .distillation import KDLoss, FeatureAlignmentLoss

class KDTrainer:
    """
    Knowledge Distillation Trainer for MeatGuard-S.
    """
    def __init__(self, student, teacher, train_loader, val_loader, optimizer, device="cuda"):
        self.student = student.to(device)
        # Ensure teacher is a model instance, not just a state_dict
        if isinstance(teacher, (str, Path)):   # checkpoint written by src/train_teacher.py
            from .teacher import load_teacher
            teacher, _ = load_teacher(teacher, map_location=device)
        elif isinstance(teacher, dict):
            raise TypeError("Pass a teacher model or a train_teacher.py checkpoint path, not a bare state_dict.")
        self.teacher = teacher
            
        self.teacher.to(device).eval()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        
        self.kd_loss_fn = KDLoss()
        self.align_loss_fn = FeatureAlignmentLoss()

        # Feature alignment projects the student's Stage-3 descriptor onto the teacher's
        # penultimate descriptor via a trainable adaptor (built lazily once both dims are
        # known). Enabled whenever the teacher can expose an intermediate/penultimate map.
        self.adaptor = None
        self.align_enabled = any(
            hasattr(self.teacher, attr)
            for attr in ("penultimate_features", "stage3_output")
        )
        if not self.align_enabled:
            warnings.warn(
                f"Teacher {type(self.teacher).__name__} exposes no alignable features; "
                "feature-alignment loss is disabled (KD uses logits only, beta has no effect).",
                stacklevel=2,
            )

    def _pool(self, feat):
        # Collapse spatial dims to a per-channel descriptor so alignment is robust
        # to differing teacher/student spatial resolutions (14x14 vs 7x7).
        return feat.mean(dim=(2, 3)) if feat.dim() == 4 else feat

    def _ensure_adaptor(self, in_dim, out_dim):
        # Lazily create the projection once both feature dims are known, and register
        # its params with the optimizer so they train alongside the student.
        if self.adaptor is None:
            self.adaptor = nn.Linear(in_dim, out_dim).to(self.device)
            self.optimizer.add_param_group({"params": self.adaptor.parameters()})

    def train_epoch(self, beta=0.1):
        self.student.train()
        total_loss = 0
        
        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            
            with torch.no_grad():
                teacher_logits = self.teacher(images)
                teacher_feat = self._get_teacher_features(images) if self.align_enabled else None

            self.optimizer.zero_grad()
            
            # Student predictions
            student_logits = self.student(images)
            
            # Loss computation
            loss_kd = self.kd_loss_fn(student_logits, teacher_logits, labels)
            total_loss_val = loss_kd
            if teacher_feat is not None:
                student_feat = self._pool(self._get_student_stage3(images))
                teacher_feat = self._pool(teacher_feat)
                self._ensure_adaptor(student_feat.size(1), teacher_feat.size(1))
                loss_align = self.align_loss_fn(self.adaptor(student_feat), teacher_feat)
                total_loss_val = loss_kd + beta * loss_align
            
            total_loss_val.backward()
            self.optimizer.step()
            
            total_loss += total_loss_val.item()
            
        return total_loss / len(self.train_loader)

    def _get_student_stage3(self, x):
        return self.student.stage3_output(x)

    def _get_teacher_features(self, x):
        if hasattr(self.teacher, 'penultimate_features'):
            return self.teacher.penultimate_features(x)
        if hasattr(self.teacher, 'stage3_output'):
            return self.teacher.stage3_output(x)
        return None
