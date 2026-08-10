# Test fixtures

**Only synthetic, non-clinical data belongs here.** Never commit real reports, patient text,
datasets, or model weights. Unit tests build their tiny vocab/config, dummy weights, and fake
backends in-memory or under `tmp_path`. `real_checkpoint_golden.json` contains only the deterministic
graph expected for the invented sentence used by the pinned-model CI integration test.
