"""Differentiable Weighted Pearson loss for PyTorch."""
import torch
import torch.nn as nn


class WeightedPearsonLoss(nn.Module):
    """Directly optimizes the competition metric: Global Weighted Pearson correlation.
    
    Weights are w_i = |clip(y_i, -2, 2)|.
    Minimizes 1 - mean(rho_w(t0), rho_w(t1)).
    """
    def __init__(self, clip_val: float = 2.0, eps: float = 1e-8):
        super().__init__()
        self.clip_val = clip_val
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            pred: Tensor of shape (..., 2)
            target: Tensor of shape (..., 2)
            mask: Optional boolean mask of shape (...) indicating valid/scored rows
        Returns:
            Scalar loss in [0, 2]
        """
        if mask is not None:
            pred = pred[mask]
            target = target[mask]
        else:
            pred = pred.reshape(-1, 2)
            target = target.reshape(-1, 2)

        # Soft/straight-through clipping or direct clamping
        y = torch.clamp(target, -self.clip_val, self.clip_val).double()
        p = torch.clamp(pred, -self.clip_val, self.clip_val).double()

        corrs = []
        for side in range(2):
            y_s = y[:, side]
            p_s = p[:, side]
            w = torch.abs(y_s)
            w_sum = torch.sum(w)

            if w_sum < self.eps:
                corrs.append(torch.tensor(0.0, device=pred.device, dtype=torch.float32))
                continue

            mean_y = torch.sum(w * y_s) / w_sum
            mean_p = torch.sum(w * p_s) / w_sum

            yc = y_s - mean_y
            pc = p_s - mean_p

            cov = torch.sum(w * yc * pc)
            var_y = torch.sum(w * yc * yc)
            var_p = torch.sum(w * pc * pc)

            denom = torch.sqrt(var_y * var_p + self.eps)
            corr = cov / denom
            corrs.append(torch.clamp(corr, -1.0, 1.0).float())

        avg_corr = (corrs[0] + corrs[1]) / 2.0
        return 1.0 - avg_corr


class CombinedWeightedLoss(nn.Module):
    """Blend of Weighted Pearson Loss and Weighted MSE for training stability."""
    def __init__(self, wp_weight: float = 0.8, wmse_weight: float = 0.2, clip_val: float = 2.0):
        super().__init__()
        self.wp_loss = WeightedPearsonLoss(clip_val=clip_val)
        self.wp_weight = wp_weight
        self.wmse_weight = wmse_weight
        self.clip_val = clip_val

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        wp = self.wp_loss(pred, target, mask)
        
        if mask is not None:
            p = pred[mask]
            y = target[mask]
        else:
            p = pred.reshape(-1, 2)
            y = target.reshape(-1, 2)

        w = torch.abs(torch.clamp(y, -self.clip_val, self.clip_val))
        wmse = torch.mean(w * (p - y) ** 2) / (torch.mean(w) + 1e-8)
        
        return self.wp_weight * wp + self.wmse_weight * wmse
