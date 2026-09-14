# third_party

Optional notes for external FoundationPose source trees. Do **not** vendor the full upstream repository here.

Set the path in `config/project.yaml`:

```yaml
foundationpose:
  repository: ../FoundationPose   # or your local clone
  weights_dir: data/foundationpose/weights
  demo_data_dir: data/foundationpose/demo_data
```

Large assets live in this project (gitignored):

```text
../data/foundationpose/weights/
../data/foundationpose/demo_data/
```

Link them into the upstream tree for official `run_demo.py`:

```powershell
python scripts\link_foundationpose_assets.py
```
