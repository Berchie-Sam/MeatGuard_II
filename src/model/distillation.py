import torch
import torch.nn as nn
import torch.nn.functional as F

class KDLoss(nn.Module):
    """
    Knowledge Distillation Loss.
    L_total = alpha * CE(student, labels) + (1-alpha) * KL(student_soft, teacher_soft)
    """
    def __init__(self, alpha=0.3, temperature=4.0, class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_logits, teacher_logits, labels):
        # Soften logits
        soft_student = F.softmax(student_logits / self.temperature, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        
        # Hard loss
        hard_loss = self.ce_loss(student_logits, labels)
        
        # Soft loss (KL Divergence)
        # KLDivLoss expects log-probabilities for the input
        soft_loss = self.kl_loss(
            F.log_softmax(student_logits / self.temperature, dim=1),
            soft_teacher
        ) * (self.temperature ** 2)
        
        return self.alpha * hard_loss + (1 - self.alpha) * soft_loss

class FeatureAlignmentLoss(nn.Module):
    """
    L2 loss between student intermediate features and teacher penultimate features.
    """
    def __init__(self):
        super().__init__()
        self.l2_loss = nn.MSELoss()

    def forward(self, student_feat, teacher_feat):
        return self.l2_loss(student_feat, teacher_feat)
