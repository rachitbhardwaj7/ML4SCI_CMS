import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class CustomSuperResolutionLoss(nn.Module):
    def __init__(self, vgg_layer=9, lambda_adv=1e-3, lambda_incidence=1.0, lambda_cardinality=0.1):
        super(CustomSuperResolutionLoss, self).__init__()
        
        # Content Loss (L1 Loss)
        self.l1_loss = nn.L1Loss()

        # Perceptual Loss (VGG-based feature loss)
        vgg = models.vgg19(pretrained=True).features[:vgg_layer].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg

        # Adversarial Loss (BCE Loss for GAN)
        self.adv_loss = nn.BCEWithLogitsLoss()

        # Hungarian Matching for Permutation-Invariant Loss
        self.lambda_adv = lambda_adv
        self.lambda_incidence = lambda_incidence
        self.lambda_cardinality = lambda_cardinality

    def perceptual_loss(self, hr_pred, hr_real):
        """Compute feature-based perceptual loss"""
        features_pred = self.vgg(hr_pred)
        features_real = self.vgg(hr_real)
        return F.mse_loss(features_pred, features_real)

    def incidence_loss(self, predicted_incidence, true_incidence):
        """Enforces attention-based incidence matrix constraints"""
        return F.kl_div(predicted_incidence.log(), true_incidence, reduction='batchmean')

    def cardinality_loss(self, predicted_cardinality, true_cardinality):
        """Encourages correct particle count predictions"""
        return F.cross_entropy(predicted_cardinality, true_cardinality)

    def forward(self, hr_pred, hr_real, predicted_incidence, true_incidence, predicted_cardinality, true_cardinality, discriminator=None):
        """
        Compute total loss:
        - hr_pred: Super-resolved image
        - hr_real: Ground truth high-resolution image
        - predicted_incidence: Model's attention-based energy fraction predictions
        - true_incidence: Ground truth incidence matrix
        - predicted_cardinality: Model's predicted particle count
        - true_cardinality: Ground truth particle count
        - discriminator: Discriminator for GAN loss
        """
        
        # Content + Perceptual Loss
        content_loss = self.l1_loss(hr_pred, hr_real) + self.perceptual_loss(hr_pred, hr_real)

        # Incidence Matrix Loss
        inc_loss = self.incidence_loss(predicted_incidence, true_incidence)

        # Cardinality Loss
        card_loss = self.cardinality_loss(predicted_cardinality, true_cardinality)

        # Adversarial Loss (if using GAN training)
        adv_loss = 0
        if discriminator is not None:
            real_labels = torch.ones_like(discriminator(hr_real))
            fake_labels = torch.zeros_like(discriminator(hr_pred))
            adv_loss = self.adv_loss(discriminator(hr_pred), real_labels) + \
                       self.adv_loss(discriminator(hr_real), fake_labels)

        # Final loss combination
        total_loss = content_loss + self.lambda_incidence * inc_loss + \
                     self.lambda_cardinality * card_loss + self.lambda_adv * adv_loss

        return total_loss
