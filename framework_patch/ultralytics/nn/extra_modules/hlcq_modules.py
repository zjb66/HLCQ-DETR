"""Core feature-enhancement modules used by HLCQ-DETR."""

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv

__all__ = (
    "HighFrequencyPreservingDownsample",
    "LocalContrastNeighborhoodConsensus",
)


class HighFrequencyPreservingDownsample(nn.Module):
    """Retain max-pooled responses and add a weak high-frequency residual."""

    def __init__(self, c1, c2, init_gain=0.05):
        super().__init__()
        self.max_pool = nn.MaxPool2d(3, 2, 1)
        self.avg_pool = nn.AvgPool2d(3, 2, 1)
        self.proj = nn.Identity() if c1 == c2 else nn.Conv2d(c1, c2, 1, bias=False)
        self.detail = nn.Sequential(
            Conv(c1, c1, 3, 2, g=c1),
            Conv(c1, c2, 1, 1, act=False),
        )
        hidden = max(c2 // 8, 8)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, c2, 1, bias=True),
            nn.Sigmoid(),
        )
        self.gain = nn.Parameter(torch.tensor(float(init_gain)))

    def forward(self, x):
        base = self.proj(self.max_pool(x))
        smooth = self.proj(self.avg_pool(x))
        detail = self.detail(x) - smooth
        return base + self.gain * self.channel_gate(detail.abs()) * detail


class LocalContrastNeighborhoodConsensus(nn.Module):
    """Enhance local contrast only when neighboring responses provide support."""

    def __init__(self, c1, c2, support_kernel=5, init_gain=0.05, consensus_scale=4.0):
        super().__init__()
        if support_kernel < 3 or support_kernel % 2 == 0:
            raise ValueError("support_kernel must be an odd integer greater than or equal to 3")
        self.proj = nn.Identity() if c1 == c2 else Conv(c1, c2, 1, 1)
        self.local_detail = Conv(c2, c2, 3, 1, g=c2, act=False)
        self.context_detail = Conv(c2, c2, 3, 1, g=c2, d=2, act=False)
        self.mix = Conv(c2 * 2, c2, 1, 1, act=False)

        hidden = max(c2 // 8, 8)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, c2, 1, bias=True),
            nn.Sigmoid(),
        )
        self.support_pool = nn.AvgPool2d(support_kernel, 1, support_kernel // 2)
        self.gain = nn.Parameter(torch.tensor(float(init_gain)))
        self.consensus_scale = float(consensus_scale)

    def forward(self, x):
        base = self.proj(x)
        contrast = base - torch.nn.functional.avg_pool2d(base, 3, stride=1, padding=1)
        support = self.support_pool(contrast.abs())

        point_energy = contrast.abs().mean(1, keepdim=True)
        support_energy = support.mean(1, keepdim=True)
        support_ratio = support_energy / (point_energy + support_energy + 1e-6)
        consensus = torch.sigmoid(self.consensus_scale * (support_ratio - 0.35))

        local = self.local_detail(contrast)
        context = self.context_detail(support)
        refinement = self.mix(torch.cat((local, context), 1))
        channel = self.channel_gate(support)
        return base + self.gain * consensus * channel * refinement
