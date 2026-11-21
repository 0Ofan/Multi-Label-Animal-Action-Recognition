"""Unified entry; defaults to configuration preflight, not training."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .safety import OUT, enable_guard
from .config import load_config, config_hash, cohort_hash
from .engine import Trainer


def main(model=None):
    enable_guard()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=OUT / f"configs/{model}_seed42_bce.json" if model else None, required=model is None)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--execute-training", action="store_true", help="Explicitly start formal training after human review; omitted = preflight only")
    args = parser.parse_args()
    config = load_config(args.config)
    if model and config["model"] != model:
        raise ValueError("Entry and configuration model disagree")
    directory = args.output_dir or OUT / "runs" / f"{config['model']}_{config['loss']['name']}_seed{config['seed']}"
    if not args.execute_training:
        print(json.dumps(dict(mode="preflight_only", training_started=False, model=config["model"], seed=config["seed"], loss=config["loss"]["name"],
                              config_SHA256=config_hash(config), shared_cohort_SHA256=cohort_hash(config), output_dir=str(directory),
                              max_epochs=config["training"]["max_epochs"], microbatch=config["data"]["batch_size"],
                              accumulation=config["data"]["accumulation_steps"], Test_used=False), ensure_ascii=False, indent=2))
        return
    trainer = Trainer(config, directory)
    trainer.run(resume=args.resume, execute_training=True)


if __name__ == "__main__":
    main()
