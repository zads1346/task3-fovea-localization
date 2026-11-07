"""Append-only, paired local/context data and shared-encoder fine models.

The existing research engine defines ResearchConfig, ResearchDataset, FoveaNet,
and training/prediction helpers before this module. No data is read on import.
"""


@dataclass
class DualConfig(ResearchConfig):
    FINE_VARIANT: str = "single"
    CONTEXT_RATIO: float = 2.0
    FINE_INPUT_SIZE: int = 256
    FINE_HEATMAP_SIZE: int = 64
    HEATMAP_SIGMA: float = 1.5


def context_output_transform(input_size, ratio):
    """Map augmented local pixel centres to the wider context pixel grid.

    With context side r times local side, c=(S-1)/2, a local pixel x maps to
    c+(x-c)/r. Applying this AFTER the full original->local transform keeps
    rotation, horizontal reflection and translation physically registered.
    """
    size, ratio = int(input_size), float(ratio)
    if size < 2 or not math.isfinite(ratio) or ratio <= 1:
        raise ValueError("Context requires input_size>=2 and finite ratio>1.")
    scale, centre = 1.0 / ratio, (size - 1.0) / 2.0
    return np.array([[scale, 0., centre * (1. - scale)],
                     [0., scale, centre * (1. - scale)],
                     [0., 0., 1.]], dtype=np.float64)


class DualContextDataset(ResearchDataset):
    """Same local samples/RNG as the base dataset plus a registered context.

    The base dataset remains responsible for local geometry and photometric
    augmentation, labels and training-only crop enlargement. Context pixels
    receive the same physical geometry but no extra brightness/color jitter.
    Both experimental arms use this exact dataset and six-channel layout;
    therefore selecting single vs dual cannot change augmentation RNG draws.
    Inference crops never consult labels, including for context construction.
    """

    def __getitem__(self, index):
        item = super().__getitem__(index)
        if self.stage != "fine":
            return item
        local_inverse = np.vstack([item["input_to_orig"].numpy().astype(np.float64),
                                   np.array([0., 0., 1.])])
        local_transform = np.linalg.inv(local_inverse)
        mapping = context_output_transform(self.size, self.cfg.CONTEXT_RATIO)
        context_transform = mapping @ local_transform
        row = self.frame.iloc[index]
        original_rgb = read_rgb(row.path)
        context = cv2.warpAffine(original_rgb, context_transform[:2],
                                 (self.size, self.size), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        context = context.astype(np.float32) / 255.
        context = ((context - np.array([.485, .456, .406], np.float32)) /
                   np.array([.229, .224, .225], np.float32))
        context = torch.from_numpy(np.ascontiguousarray(context.transpose(2, 0, 1)))
        item["image"] = torch.cat([item["image"], context], dim=0)
        item["context_input_to_orig"] = torch.from_numpy(
            np.linalg.inv(context_transform)[:2].astype(np.float32))
        item["context_ratio"] = torch.tensor(float(self.cfg.CONTEXT_RATIO), dtype=torch.float32)
        return item


def align_context_feature(context, input_size, stride, ratio):
    """Sample context feature locations corresponding to local feature centres.

    Torchvision ResNet18's odd kernels with symmetric padding place c1..c4
    centres at 0, stride, 2*stride,... in the input image, with strides 4/8/16/32.
    Its feature-grid midpoint is NOT the input midpoint for even-sized inputs.
    Consequently shrinking a generic [-1,1] grid by r is not pixel-aligned.
    Here the input-centre mapping is applied first, then divided by stride and
    converted to grid_sample coordinates. align_corners=True refers only to
    the feature-index grid; no heatmap decode convention is changed.
    """
    if context.ndim != 4:
        raise ValueError("Expected BCHW context features.")
    size, stride, ratio = int(input_size), int(stride), float(ratio)
    if size < 2 or stride <= 0 or not math.isfinite(ratio) or ratio <= 1:
        raise ValueError("Invalid context feature geometry.")
    height, width = context.shape[-2:]
    if height < 2 or width < 2 or height != (size + stride - 1) // stride or width != height:
        raise ValueError("Feature shape does not match the square ResNet input/stride.")
    dtype = torch.float32  # Stable geometry under AMP; cast only at sampling.
    centre = (size - 1.0) / 2.0
    ys = torch.arange(height, device=context.device, dtype=dtype) * stride
    xs = torch.arange(width, device=context.device, dtype=dtype) * stride
    ys = (centre + (ys - centre) / ratio) / stride
    xs = (centre + (xs - centre) / ratio) / stride
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([2. * xx / (width - 1.) - 1.,
                        2. * yy / (height - 1.) - 1.], dim=-1)
    grid = grid.unsqueeze(0).expand(context.shape[0], -1, -1, -1)
    # CPU grid_sample does not accept float16; use float32 in all cases for
    # deterministic, finite sampling, then restore feature dtype for fusion.
    aligned = F.grid_sample(context.float(), grid, mode="bilinear",
                            padding_mode="border", align_corners=True)
    return aligned.to(dtype=context.dtype)


class FineSingleLocal(FoveaNet):
    """Exact original fine architecture operating on the local half only."""

    def forward(self, image):
        if image.ndim != 4 or image.shape[1] != 6:
            raise ValueError("Fine inputs must concatenate local RGB and context RGB.")
        return super().forward(image[:, :3])


class FineDualContext(FoveaNet):
    """Shared ImageNet ResNet18 with registered multi-level context residuals."""

    def __init__(self, context_ratio=2.0, **kwargs):
        super().__init__(**kwargs)
        self.context_ratio = float(context_ratio)
        if not math.isfinite(self.context_ratio) or self.context_ratio <= 1:
            raise ValueError("context_ratio must be finite and >1.")
        self.context_fusions = nn.ModuleList(
            [nn.Conv2d(2 * channels, channels, kernel_size=1, bias=False)
             for channels in (64, 128, 256, 512)])
        # Begin close to the local model while retaining nonzero gradients
        # into the context input and shared encoder from the very first step.
        for layer in self.context_fusions:
            nn.init.normal_(layer.weight, mean=0., std=.01 / math.sqrt(layer.in_channels))

    def _features(self, image):
        encoder = self.encoder
        stem = encoder.relu(encoder.bn1(encoder.conv1(image)))
        c1 = encoder.layer1(encoder.maxpool(stem))
        c2 = encoder.layer2(c1)
        c3 = encoder.layer3(c2)
        c4 = encoder.layer4(c3)
        return (c1, c2, c3, c4)

    def forward(self, image):
        if image.ndim != 4 or image.shape[1] != 6 or image.shape[-1] != image.shape[-2]:
            raise ValueError("FineDualContext needs square local/context RGB concatenation.")
        batch, size = image.shape[0], image.shape[-1]
        # One encoder and one call; no duplicated or separately pretrained net.
        joint = torch.cat([image[:, :3], image[:, 3:]], dim=0)
        shared = self._features(joint)
        fused = []
        for stride, features, layer in zip((4, 8, 16, 32), shared, self.context_fusions):
            local, context = features[:batch], features[batch:]
            aligned = align_context_feature(context, size, stride, self.context_ratio)
            fused.append(local + layer(torch.cat([local, aligned], dim=1)))
        c1, c2, c3, c4 = fused
        decoded = self.decoder3(c4, c3)
        decoded = self.decoder2(decoded, c2)
        decoded = self.decoder1(decoded, c1)
        logits = self.heatmap_head(decoded)
        if logits.shape[-2:] != (self.heatmap_size, self.heatmap_size):
            logits = F.interpolate(logits, size=(self.heatmap_size, self.heatmap_size),
                                   mode="bilinear", align_corners=False)
        return {"logits": logits, "coords": heatmap_to_coords(logits),
                "visibility_logits": logits.new_zeros(batch)}


# Notebook re-execution must not capture the override as its own base function.
if "_dual_context_base_loader" not in globals():
    _dual_context_base_loader = make_research_loader
if "_dual_context_base_model" not in globals():
    _dual_context_base_model = research_build_model


def make_research_loader(frame, cfg, training=False, stage="coarse", crop_spec=None, seed=0):
    if stage != "fine":
        return _dual_context_base_loader(frame, cfg, training, stage, crop_spec, seed)
    return DataLoader(DualContextDataset(frame, cfg, training, stage, crop_spec),
                      batch_size=int(cfg.BATCH_SIZE), shuffle=training,
                      num_workers=int(cfg.NUM_WORKERS), pin_memory=torch.cuda.is_available(),
                      drop_last=False, worker_init_fn=seed_worker,
                      generator=torch.Generator().manual_seed(int(seed)), persistent_workers=False)


def research_build_model(cfg, stage, device):
    if stage != "fine":
        return _dual_context_base_model(cfg, stage, device)
    variant = getattr(cfg, "FINE_VARIANT", "single")
    path = getattr(cfg, "PRETRAINED_WEIGHTS_PATH", None)
    if cfg.PRETRAINED and (not path or not Path(path).is_file()):
        raise FileNotFoundError("Fine training requires explicit verified ImageNet ResNet18 weights.")
    common = dict(pretrained=bool(cfg.PRETRAINED), allow_download=False,
                  use_visibility=False, freeze_bn=bool(cfg.FREEZE_BN),
                  heatmap_size=int(cfg.FINE_HEATMAP_SIZE),
                  pretrained_weights_path=path if cfg.PRETRAINED else None)
    if variant == "single":
        model = FineSingleLocal(**common)
    elif variant == "dual":
        model = FineDualContext(context_ratio=cfg.CONTEXT_RATIO, **common)
    else:
        raise ValueError(f"Unknown fine variant: {variant}")
    return model.to(device)
