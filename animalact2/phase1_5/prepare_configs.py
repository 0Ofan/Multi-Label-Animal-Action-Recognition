"""Write the shared first-round plans, never run training."""
import copy
from .config import default_config, cohort_hash
from .safety import OUT, enable_guard, write_json


def main():
    enable_guard()
    hashes = []
    for name in ("model1", "model2", "model3"):
        config = default_config(name, 42)
        config["training"].update(max_epochs=50,patience=10)
        path = OUT / f"configs/{name}_seed42_bce.json"
        if path.exists():
            raise FileExistsError("Preserve prepared configuration")
        write_json(path, config)
        hashes.append(cohort_hash(config))
    assert len(set(hashes)) == 1
    reference = default_config("model3", 42)
    reference["training"].update(max_epochs=50,patience=10)
    reference["loss"]["name"] = "asl"
    write_json(OUT / "configs/asl_reference_not_first_round.json", reference)
    write_json(OUT / "configs/experiment_matrix.json", dict(first_round=[f"{n}_seed42_bce.json" for n in ("model1", "model2", "model3")],
               shared_cohort_SHA256=hashes[0], seed=42, future_replication_seeds_not_scheduled=[123,2026],
               ASL_reference="asl_reference_not_first_round.json", automatic_execution=False, Test_use=False))
    print("Prepared three matched BCE plans and one inactive ASL reference; no training started")


if __name__ == "__main__":
    main()
