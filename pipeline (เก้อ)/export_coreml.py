import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from classify_trucks import COREML_PATH, MODEL, PRETRAINED  # noqa: E402


def export_yolo(weights, out, imgsz):
    from ultralytics import YOLO
    made = Path(YOLO(weights).export(format="coreml", imgsz=imgsz, nms=False, half=True))
    if made.resolve() != Path(out).resolve():
        shutil.rmtree(out, ignore_errors=True)
        shutil.move(str(made), out)
    print("wrote", out)


def export_clip(out, pretrained=PRETRAINED):
    import coremltools as ct
    import open_clip
    import torch

    class ImageEncoder(torch.nn.Module):
        def __init__(self, visual):
            super().__init__()
            self.visual = visual

        def forward(self, x):
            f = self.visual(x)
            return f / f.norm(dim=-1, keepdim=True)

    model = open_clip.create_model_and_transforms(MODEL, pretrained=pretrained)[0].eval()
    x = torch.zeros(1, 3, 224, 224)
    traced = torch.jit.trace(ImageEncoder(model.visual).eval(), x)
    ml = ct.convert(traced, inputs=[ct.TensorType(name="x", shape=x.shape)], outputs=[ct.TensorType(name="emb")],
                    convert_to="mlprogram", compute_precision=ct.precision.FLOAT16,
                    minimum_deployment_target=ct.target.macOS13)
    shutil.rmtree(out, ignore_errors=True)
    ml.save(out)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="*", default=["yolo", "clip"], help="yolo, clip")
    ap.add_argument("--weights", default="models/yolo11s.pt")
    ap.add_argument("--yolo-out", default="models/yolo11s_nonms.mlpackage")
    ap.add_argument("--clip-out", default=COREML_PATH)
    ap.add_argument("--imgsz", type=int, default=640)
    a = ap.parse_args()
    if set(a.which) - {"yolo", "clip"}:
        ap.error(f"unknown model {sorted(set(a.which) - {'yolo', 'clip'})}, choose yolo and/or clip")
    if "yolo" in a.which:
        export_yolo(a.weights, a.yolo_out, a.imgsz)
    if "clip" in a.which:
        export_clip(a.clip_out)


if __name__ == "__main__":
    main()
