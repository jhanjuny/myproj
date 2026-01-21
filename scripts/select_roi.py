import sys
import cv2

path = sys.argv[1]
img = cv2.imread(path)
if img is None:
    raise RuntimeError(f"cannot read: {path}")

r = cv2.selectROI("select ROI (drag mouse, press ENTER)", img, fromCenter=False, showCrosshair=True)
print("rect:", [int(v) for v in r])  # [x,y,w,h]
cv2.destroyAllWindows()
