# Official pretrained weights: resolve local Kaggle inputs without internet.
# The bundle contains ImageNet-1K initialization, not fovea-trained weights.
RESEARCH_WEIGHT_FILENAME = "resnet18-f37072fd.pth"
RESEARCH_WEIGHT_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
RESEARCH_WEIGHT_BYTES = 46830571
RESEARCH_WEIGHT_URL = "https://download.pytorch.org/models/resnet18-f37072fd.pth"


def resolve_research_weights(cfg):
    """Return a SHA256-verified .pth path, or None only if PRETRAINED=False.

    PRETRAINED_WEIGHTS_PATH accepts a .pth, the supplied ZIP, or a directory.
    With no explicit path, recursively discover the official filename or
    fovea_pretrained_weights.zip under /kaggle/input. Kaggle may unpack uploaded
    ZIPs automatically; both layouts work. All matching copies must have the
    expected complete SHA256. A ZIP is streamed to a fixed filename under
    OUTPUT_DIR/pretrained_cache; input datasets are never modified.
    """
    from pathlib import Path, PurePosixPath
    import hashlib
    import os
    import urllib.request
    import zipfile

    if not bool(getattr(cfg, "PRETRAINED", True)):
        print("PRETRAINED=False: explicitly using random encoder initialization.")
        return None

    def checked_stream(stream, label, destination=None):
        checksum = hashlib.sha256()
        total = 0
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            total += len(block)
            if total > RESEARCH_WEIGHT_BYTES:
                raise ValueError(f"Pretrained weight is larger than expected: {label}")
            checksum.update(block)
            if destination is not None:
                destination.write(block)
        actual = checksum.hexdigest()
        if total != RESEARCH_WEIGHT_BYTES or actual != RESEARCH_WEIGHT_SHA256:
            raise ValueError(
                f"Pretrained weight verification failed: {label}\n"
                f"Expected {RESEARCH_WEIGHT_BYTES} bytes / SHA256 {RESEARCH_WEIGHT_SHA256}; "
                f"got {total} bytes / SHA256 {actual}. "
                "Remove the corrupt or incompatible input and attach the supplied official bundle."
            )

    def check_file(path):
        with path.open("rb") as stream:
            checked_stream(stream, str(path))

    explicit = str(getattr(cfg, "PRETRAINED_WEIGHTS_PATH", "") or "").strip()
    root = Path(explicit).expanduser() if explicit else Path("/kaggle/input")
    if explicit and not root.exists():
        raise FileNotFoundError(f"PRETRAINED_WEIGHTS_PATH does not exist: {root}")
    if root.is_file():
        inputs = [root]
    elif root.is_dir():
        inputs = sorted(set(root.rglob(RESEARCH_WEIGHT_FILENAME)) |
                        set(root.rglob("fovea_pretrained_weights.zip")))
    else:
        inputs = []

    # Verify every matching source before choosing one: conflicting copies fail.
    candidates = []
    for path in inputs:
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path, "r") as archive:
                members = [info for info in archive.infolist()
                           if not info.is_dir() and
                           PurePosixPath(info.filename.replace("\\", "/")).name == RESEARCH_WEIGHT_FILENAME]
                if not members:
                    raise ValueError(f"No {RESEARCH_WEIGHT_FILENAME} found inside {path}")
                for info in members:
                    entry = PurePosixPath(info.filename.replace("\\", "/"))
                    if entry.is_absolute() or ".." in entry.parts:
                        raise ValueError(f"Unsafe ZIP member name: {info.filename}")
                    if info.file_size != RESEARCH_WEIGHT_BYTES:
                        raise ValueError(f"Wrong pretrained member size: {path}!{info.filename}")
                    with archive.open(info, "r") as stream:
                        checked_stream(stream, f"{path}!{info.filename}")
                    candidates.append((path, info.filename))
        else:
            check_file(path)
            candidates.append((path, None))

    if candidates:
        # Prefer already-unpacked input when duplicate copies are identical.
        plain = [path for path, member in candidates if member is None]
        if plain:
            selected = plain[0].resolve()
            print(f"Verified official ImageNet-1K ResNet18 weights: {selected}")
            return str(selected)

    if not candidates and explicit:
        raise FileNotFoundError(
            f"No {RESEARCH_WEIGHT_FILENAME} or fovea_pretrained_weights.zip found under {root}."
        )

    cache_dir = (Path(cfg.OUTPUT_DIR).expanduser() / "pretrained_cache").resolve()
    kaggle_input = Path("/kaggle/input").resolve()
    if cache_dir == kaggle_input or kaggle_input in cache_dir.parents:
        raise ValueError("OUTPUT_DIR must be writable and outside /kaggle/input.")
    cache = cache_dir / RESEARCH_WEIGHT_FILENAME
    if cache.exists():
        check_file(cache)
        print(f"Verified pretrained weight cache: {cache}")
        return str(cache)

    if not candidates and not bool(getattr(cfg, "ALLOW_WEIGHT_DOWNLOAD", False)):
        raise FileNotFoundError(
            "Pretrained weights were not found. Upload fovea_pretrained_weights.zip as a Kaggle "
            "Dataset and attach it with Add Input, or set PRETRAINED_WEIGHTS_PATH to its .pth, "
            "ZIP, or folder. For an online download, explicitly set ALLOW_WEIGHT_DOWNLOAD=True "
            "and enable Notebook Internet. No random-initialization fallback is performed."
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = cache_dir / f"{RESEARCH_WEIGHT_FILENAME}.{os.getpid()}.partial"
    try:
        if candidates:
            archive_path, member = candidates[0]
            with zipfile.ZipFile(archive_path, "r") as archive:
                with archive.open(member, "r") as stream, temporary.open("wb") as output:
                    checked_stream(stream, f"{archive_path}!{member}", output)
        else:
            print(f"Downloading official ImageNet-1K weights: {RESEARCH_WEIGHT_URL}")
            request = urllib.request.Request(RESEARCH_WEIGHT_URL, headers={"User-Agent": "fovea-research-notebook/1.0"})
            with urllib.request.urlopen(request, timeout=30) as stream, temporary.open("wb") as output:
                checked_stream(stream, RESEARCH_WEIGHT_URL, output)
        temporary.replace(cache)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"Verified pretrained weights available at {cache}")
    return str(cache)
