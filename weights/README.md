# weights/

Drop your trained checkpoints here (git-ignored — see root `.gitignore`, `*.pt` / `*.pth` are excluded):

- `ohrc_best.pt` — OHRC-trained YOLOv11 + ResNet101/timm backbone
- `tmc2_best.pt` — TMC-2-trained YOLOv11 + ResNet101/timm backbone

Both checkpoints require the patched Ultralytics build from
[PR #19609](https://github.com/ultralytics/ultralytics/pull/19609) to load —
stock `ultralytics` will fail to deserialize the timm backbone module. See
the root `README.md` for install steps.
