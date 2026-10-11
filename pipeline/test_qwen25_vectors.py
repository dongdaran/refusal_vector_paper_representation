"""Extract candidate vectors and save baseline responses for 10 examples.

Run from the repository root: python -m pipeline.test_qwen25_vectors
"""

import argparse
import json
import random
from pathlib import Path

from dataset.load_dataset import load_dataset_split


MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--num-samples", type=int, default=10,
                        help="Total examples, split equally between harmful and harmless")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--generation-batch-size", type=int, default=1)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("pipeline/runs/Qwen2.5-7B-Instruct/smoke_vectors"))
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate sampling without loading or downloading the model")
    args = parser.parse_args()
    if args.num_samples < 2 or args.num_samples % 2:
        parser.error("--num-samples must be an even integer >= 2")
    if args.max_new_tokens < 1 or args.generation_batch_size < 1:
        parser.error("generation token limit and batch size must be >= 1")
    return args


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    samples = {}
    sampled_rows = {}
    for group in ("harmful", "harmless"):
        rows = load_dataset_split(group, "train")
        if len(rows) < args.num_samples // 2:
            raise ValueError(f"Not enough {group} training examples")
        sampled_rows[group] = rng.sample(rows, args.num_samples // 2)
        samples[group] = [row["instruction"] for row in sampled_rows[group]]
    print(f"Model: {args.model_id}, revision: {args.revision}")
    print(f"Examples: {len(samples['harmful'])} harmful + {len(samples['harmless'])} harmless")
    if args.dry_run:
        print("Sampling validated; model was not loaded.")
        return

    import torch
    from pipeline.model_utils.model_factory import construct_model_base
    from pipeline.submodules.generate_directions import generate_directions

    model_base = construct_model_base(args.model_id, revision=args.revision)
    model_base.model.config.use_cache = False
    # Use precisely the same extraction entrypoint and positions as run_pipeline.py.
    with torch.inference_mode():
        vectors = generate_directions(
            model_base, samples["harmful"], samples["harmless"],
            artifact_dir=str(args.output_dir),
        ).cpu()
    positions = list(range(-len(model_base.eoi_toks), 0))
    metadata = {
        "model_id": args.model_id, "revision": args.revision,
        "num_samples": args.num_samples, "seed": args.seed, "split": "train",
        "positions": positions, "shape": list(vectors.shape),
        "definition": "harmful mean minus harmless mean; transformer block inputs",
        "samples": samples,
        "model_dtype": str(model_base.model.dtype),
        "generation": {"intervention": "baseline", "do_sample": False,
                       "max_new_tokens": args.max_new_tokens,
                       "batch_size": args.generation_batch_size},
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"Saved {args.output_dir / 'mean_diffs.pt'}; shape={tuple(vectors.shape)}")

    completions = []
    model_base.model.config.use_cache = True
    with torch.inference_mode():
        for group, rows in sampled_rows.items():
            responses = model_base.generate_completions(
                rows, batch_size=args.generation_batch_size,
                max_new_tokens=args.max_new_tokens,
            )
            for response in responses:
                response["harmtype"] = group
                response["intervention"] = "baseline"
            completions.extend(responses)
    if len(completions) != args.num_samples:
        raise RuntimeError(f"Expected {args.num_samples} responses, got {len(completions)}")
    completion_path = args.output_dir / "completions.json"
    completion_path.write_text(
        json.dumps(completions, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"Saved {completion_path}; responses={len(completions)}")


if __name__ == "__main__":
    main()
