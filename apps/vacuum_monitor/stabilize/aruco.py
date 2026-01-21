import cv2
import numpy as np

class ArucoStabilizer:
    def __init__(self, dict_name, marker_ids, output_size):
        self.dict = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, dict_name)
        )
        self.marker_ids = marker_ids
        self.output_size = tuple(output_size)

    def stabilize(self, frame):
        corners, ids, _ = cv2.aruco.detectMarkers(frame, self.dict)
        if ids is None:
            return None

        id_to_corner = {
            int(i): c[0] for c, i in zip(corners, ids.flatten())
            if int(i) in self.marker_ids
        }

        if len(id_to_corner) != 4:
            return None

        src = np.array([id_to_corner[i].mean(axis=0) for i in self.marker_ids],
                       dtype=np.float32)

        w, h = self.output_size
        dst = np.array([[0,0], [w,0], [w,h], [0,h]], dtype=np.float32)

        H = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(frame, H, (w, h))
