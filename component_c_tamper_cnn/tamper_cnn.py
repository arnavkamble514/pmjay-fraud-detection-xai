from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

HERE = Path(__file__).parent
WEIGHTS = HERE / "tamper_cnn.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TamperDetector:
    def __init__(self, weights_path=WEIGHTS):
        if not Path(weights_path).exists():
            raise FileNotFoundError(
                f"{weights_path} not found — run train.py first."
            )
        ckpt = torch.load(weights_path, map_location=DEVICE)

        self.classes = ckpt.get("classes", ["authentic", "tampered"])
        self.img_size = ckpt.get("img_size", 224)

        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
        model.load_state_dict(ckpt["state_dict"])
        self.model = model.to(DEVICE).eval()

        # last conv block — what Grad-CAM reads activations from
        self.target_layer = self.model.layer4[-1]

        self.tf = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

    def _tensor(self, image_path):
        img = Image.open(image_path).convert("RGB")
        return self.tf(img).unsqueeze(0).to(DEVICE), img

    def score_tamper(self, image_path):
        """
        Returns {"tamper_probability": 0-1, "predicted_class": str}.

        tamper_probability is the number that becomes a column in the
        Stage 2 fused feature table.
        """
        x, _ = self._tensor(image_path)
        with torch.no_grad():
            probs = torch.softmax(self.model(x), dim=1)[0]
        idx = int(probs.argmax())
        return {
            "tamper_probability": float(probs[1]),
            "predicted_class": self.classes[idx],
        }

    def gradcam_heatmap(self, image_path, out_path=None, alpha=0.45):
        """
        Grad-CAM overlay showing WHERE the model thinks tampering is.

        This is the auditor-facing explanation from your synopsis —
        the red regions on the document in the dashboard mockup.
        Returns (overlay_bgr, stats_dict).
        """
        x, pil_img = self._tensor(image_path)

        activations, gradients = {}, {}

        def fwd_hook(_m, _i, out):
            activations["value"] = out.detach()

        def bwd_hook(_m, _gi, gout):
            gradients["value"] = gout[0].detach()

        h1 = self.target_layer.register_forward_hook(fwd_hook)
        h2 = self.target_layer.register_full_backward_hook(bwd_hook)

        logits = self.model(x)
        self.model.zero_grad()
        logits[0, 1].backward()   # class 1 = tampered

        h1.remove()
        h2.remove()

        acts = activations["value"][0]
        grads = gradients["value"][0]
        weights = grads.mean(dim=(1, 2), keepdim=True)

        cam = F.relu((weights * acts).sum(dim=0)).cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()

        orig = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        cam_resized = cv2.resize(cam, (orig.shape[1], orig.shape[0]))
        heat = cv2.applyColorMap(np.uint8(255 * cam_resized),
                                 cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(orig, 1 - alpha, heat, alpha, 0)

        if out_path:
            cv2.imwrite(str(out_path), overlay)

        stats = {
            "heatmap_max_intensity": float(cam_resized.max()),
            "suspicious_area_pct": float((cam_resized > 0.5).mean()),
        }
        return overlay, stats


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python tamper_cnn.py <image_path> [output_overlay.png]")
        sys.exit(1)

    det = TamperDetector()
    print(det.score_tamper(sys.argv[1]))

    out = sys.argv[2] if len(sys.argv) > 2 else "gradcam_overlay.png"
    _, stats = det.gradcam_heatmap(sys.argv[1], out_path=out)
    print(f"Heatmap stats: {stats}")
    print(f"Overlay saved to {out}")
