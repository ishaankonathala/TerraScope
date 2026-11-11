# Model Checkpoints Directory

Trained model weights and checkpoint metadata are saved under `models/checkpoints/`.

## Checkpoint Convention

- `<model_name>_<dataset>_best.pt` — Checkpoint achieving best validation metric (F1 / IoU).
- `<model_name>_<dataset>_latest.pt` — Most recent epoch checkpoint for resuming training.
- Associated `experiments/results/` JSON/CSV logs record full training hyperparameters, random seed, hardware device, parameter count, training duration, and evaluation provenance (`pretrained_backbone_untrained_head` vs `fine_tuned` vs `fully_trained`).

Models without saved fine-tuned checkpoints can still be instantiated for architecture verification or backbone inspection, and the UI/API explicitly distinguishes trained checkpoints from untrained/backbone-only states.

