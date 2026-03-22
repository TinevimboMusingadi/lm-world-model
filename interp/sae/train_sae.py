try:
    from sae_lens import SAETrainingRunner, LanguageModelSAERunnerConfig
except ImportError:
    pass
import torch

def train_sae_on_layer(
    model_path: str,
    activation_store_path: str,
    target_layer: int,
    d_model: int,
    expansion_factor: int = 16,
    n_steps: int = 10_000,
    output_dir: str = "interp/results/sae_features/"
) -> None:
    try:
        cfg = LanguageModelSAERunnerConfig(
            model_name=model_path,
            hook_name=f"blocks.{target_layer}.hook_resid_post",
            hook_layer=target_layer,
            d_in=d_model,
            expansion_factor=expansion_factor,
            activation_fn="topk",
            topk=32,                   # 32 active features per token
            n_training_steps=n_steps,
            train_batch_size_tokens=4096,
            log_to_wandb=True,
            wandb_project="lm-world-model-sae",
            checkpoint_path=output_dir,
            dtype="float32",
        )
        runner = SAETrainingRunner(cfg)
        sae = runner.run()
        return sae
    except NameError:
        print("sae_lens not installed")
