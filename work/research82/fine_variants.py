"""Fine-only backbone/head variants; append after the audited research engine.

No automatic downloads and no implicit random-pretraining fallback. The caller
validates official file SHA256 before use. PRETRAINED=False is an explicit mode
for checkpoint reload or functional tests, never an automatic recovery path.
"""

from dataclasses import dataclass
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import efficientnet_b0


@dataclass
class ImprovedConfig(ResearchConfig):
    FINE_VARIANT: str = "resnet18"
    EFFNET_WEIGHTS_PATH: str | None = None


def _fine_variant_local_state(path):
    if path is None or not Path(path).is_file():
        raise FileNotFoundError(f"Explicit local pretrained weights are required: {path}")
    # Official torchvision state dictionaries contain only tensors.
    try:
        return torch.load(Path(path), map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(Path(path), map_location="cpu")


class FineEfficientNetB0(nn.Module):
    """ImageNet EfficientNet-B0 encoder + the existing three skip decoders."""

    def __init__(self, pretrained=True, pretrained_weights_path=None,
                 heatmap_size=64, freeze_bn=True):
        super().__init__()
        if pretrained and (pretrained_weights_path is None or
                           not Path(pretrained_weights_path).is_file()):
            raise FileNotFoundError("EfficientNet-B0 requires explicit verified local ImageNet weights.")
        encoder = efficientnet_b0(weights=None)
        if pretrained:
            encoder.load_state_dict(_fine_variant_local_state(pretrained_weights_path), strict=True)
        # Keep the native stem and MBConv feature blocks through stride-32/320.
        # Drop the unused classification expansion and classifier AFTER loading
        # the complete official state dictionary strictly.
        encoder.features = encoder.features[:8]
        encoder.classifier = nn.Identity()
        self.encoder = encoder
        self.decoder3 = DecoderBlock(320, 112, 256)
        self.decoder2 = DecoderBlock(256, 40, 128)
        self.decoder1 = DecoderBlock(128, 24, 64)
        self.heatmap_head = nn.Conv2d(64, 1, kernel_size=1)
        self.heatmap_size = int(heatmap_size)
        self.freeze_bn = bool(freeze_bn)
        self.use_visibility = False

    def train(self, mode=True):
        super().train(mode)
        if mode and self.freeze_bn:
            for module in self.encoder.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return self

    def forward(self, image):
        features = {}
        value = image
        for index, block in enumerate(self.encoder.features):
            value = block(value)
            if index in (2, 3, 5, 7):
                features[index] = value
        decoded = self.decoder3(features[7], features[5])
        decoded = self.decoder2(decoded, features[3])
        decoded = self.decoder1(decoded, features[2])
        logits = self.heatmap_head(decoded)
        if logits.shape[-2:] != (self.heatmap_size, self.heatmap_size):
            logits = F.interpolate(logits, size=(self.heatmap_size, self.heatmap_size),
                                   mode="bilinear", align_corners=False)
        return {"logits": logits, "coords": heatmap_to_coords(logits),
                "visibility_logits": logits.new_zeros(image.shape[0])}


class FineResNet18HighRes(FoveaNet):
    """Learned stride-2 decoder with the pretrained ResNet18 stem skip."""

    def __init__(self, pretrained=True, pretrained_weights_path=None,
                 heatmap_size=128, freeze_bn=True):
        if pretrained and (pretrained_weights_path is None or
                           not Path(pretrained_weights_path).is_file()):
            raise FileNotFoundError("High-resolution ResNet18 requires explicit verified local ImageNet weights.")
        super().__init__(pretrained=pretrained, allow_download=False,
                         pretrained_weights_path=pretrained_weights_path,
                         use_visibility=False, heatmap_size=heatmap_size,
                         freeze_bn=freeze_bn)
        self.decoder0 = DecoderBlock(64, 64, 32)
        self.heatmap_head = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, image):
        encoder = self.encoder
        stem = encoder.relu(encoder.bn1(encoder.conv1(image)))
        c1 = encoder.layer1(encoder.maxpool(stem))
        c2 = encoder.layer2(c1)
        c3 = encoder.layer3(c2)
        c4 = encoder.layer4(c3)
        decoded = self.decoder3(c4, c3)
        decoded = self.decoder2(decoded, c2)
        decoded = self.decoder1(decoded, c1)
        decoded = self.decoder0(decoded, stem)
        logits = self.heatmap_head(decoded)
        if logits.shape[-2:] != (self.heatmap_size, self.heatmap_size):
            raise ValueError("HighRes learned output requires FINE_HEATMAP_SIZE = FINE_INPUT_SIZE // 2.")
        return {"logits": logits, "coords": heatmap_to_coords(logits),
                "visibility_logits": logits.new_zeros(image.shape[0])}


# Preserve the base entrypoint once, also when a Notebook cell is run twice.
if "_research_build_model_before_fine_variants" not in globals():
    _research_build_model_before_fine_variants = research_build_model


def research_build_model(cfg, stage, device):
    variant = getattr(cfg, "FINE_VARIANT", "resnet18")
    if stage != "fine" or variant == "resnet18":
        return _research_build_model_before_fine_variants(cfg, stage, device)
    common = {"pretrained": bool(cfg.PRETRAINED),
              "heatmap_size": int(cfg.FINE_HEATMAP_SIZE),
              "freeze_bn": bool(cfg.FREEZE_BN)}
    if variant == "efficientnet_b0":
        model = FineEfficientNetB0(pretrained_weights_path=cfg.EFFNET_WEIGHTS_PATH, **common)
    elif variant == "resnet18_highres":
        if int(cfg.FINE_HEATMAP_SIZE) != int(cfg.FINE_INPUT_SIZE) // 2:
            raise ValueError("resnet18_highres requires FINE_HEATMAP_SIZE = FINE_INPUT_SIZE // 2.")
        model = FineResNet18HighRes(pretrained_weights_path=cfg.PRETRAINED_WEIGHTS_PATH, **common)
    else:
        raise ValueError(f"Unknown FINE_VARIANT: {variant!r}")
    return model.to(device)
