"""Prepare a multi-teacher OPD student from specialists fine-tuned from one base."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory


def load_checkpoint(path):
    from transformers import AutoConfig, AutoModelForCausalLM

    if (path / "adapter_config.json").is_file():
        raise ValueError(f"Merge the adapter into its base before using this checkpoint: {path}")
    config = AutoConfig.from_pretrained(path, local_files_only=True)
    if getattr(config, "quantization_config", None):
        raise ValueError(f"Use a full floating-point checkpoint: {path}")
    model, info = AutoModelForCausalLM.from_pretrained(
        path, config=config, dtype="auto", local_files_only=True, output_loading_info=True
    )
    for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        if info.get(key):
            raise ValueError(f"Incomplete or incompatible checkpoint {path} ({key})")
    return model


def add_expert(path, student, parameters, vocab, share):
    from transformers import AutoTokenizer

    if AutoTokenizer.from_pretrained(path, local_files_only=True).get_vocab() != vocab:
        raise ValueError(f"Expert token IDs differ from the base: {path}")
    teacher = load_checkpoint(path)
    # Attention heads and position encoding can differ even with identical tensor shapes.
    layout = ("num_attention_heads", "num_key_value_heads", "head_dim", "rope_theta",
              "rope_scaling", "rope_parameters", "position_embedding_type")
    if type(teacher) is not type(student) or any(
        getattr(teacher.config, name, None) != getattr(student.config, name, None) for name in layout
    ):
        raise ValueError(f"Expert architecture differs from the base: {path}")
    source = dict(teacher.named_parameters())
    if source.keys() != parameters.keys():
        raise ValueError(f"Expert parameter names differ from the base: {path}")
    for name, param in parameters.items():
        weight = source[name]
        if weight.shape != param.shape or not weight.is_floating_point():
            raise ValueError(f"Incompatible expert parameter {name} in {path}")
        param.add_(weight, alpha=share)


def prepare(base, experts, output):
    import torch
    from transformers import AutoTokenizer

    if not experts or len(set(experts)) != len(experts) or base in experts:
        raise ValueError("Provide distinct expert checkpoints and their shared base checkpoint")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")

    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    student = load_checkpoint(base)
    vocab = tokenizer.get_vocab()
    parameters = dict(student.named_parameters())
    dtypes = {name: param.dtype for name, param in parameters.items()}
    expert_share = 0.5 / len(experts)
    with torch.no_grad():
        for param in parameters.values():
            if not param.is_floating_point():
                raise ValueError("A full floating-point base checkpoint is required")
            param.data = param.float()
            param.mul_(0.5)

        for path in experts:
            add_expert(path, student, parameters, vocab, expert_share)

        for name, param in parameters.items():
            param.data = param.to(dtype=dtypes[name])

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".opd-init-", dir=output.parent) as tmp:
        stage = Path(tmp) / "weights"
        student.save_pretrained(stage)
        tokenizer.save_pretrained(stage)
        sources = {"base": {"path": str(base), "weight": 0.5},
                   "experts": [{"path": str(path), "weight": expert_share} for path in experts]}
        (stage / "opd-init.json").write_text(json.dumps(sources, indent=2) + "\n")
        stage.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--expert", required=True, action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.base.resolve(), [path.resolve() for path in args.expert], args.output.resolve())


if __name__ == "__main__":
    main()
