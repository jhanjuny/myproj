import cv2

class MultiViewCapture:
    def __init__(self, path, views: dict):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {path}")
        self.views = views  # { "top": {"crop":[x,y,w,h]}, "bottom": {...} }

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Cannot read frame from source")

        out = {}
        H, W = frame.shape[:2]
        for vid, vcfg in self.views.items():
            x, y, w, h = vcfg["crop"]
            x = max(0, min(x, W-1))
            y = max(0, min(y, H-1))
            w = max(1, min(w, W - x))
            h = max(1, min(h, H - y))
            out[vid] = frame[y:y+h, x:x+w].copy()
        return out

    def release(self):
        self.cap.release()
