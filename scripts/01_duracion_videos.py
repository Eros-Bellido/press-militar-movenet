import cv2, glob, os, csv

VIDEOS = os.path.join("data", "videos")
os.makedirs("outputs", exist_ok=True)

with open(os.path.join("outputs", "frames.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["archivo", "fotogramas", "fps", "ancho", "alto"])
    for v in sorted(glob.glob(os.path.join(VIDEOS, "**", "*.*"), recursive=True)):
        c = cv2.VideoCapture(v)
        w.writerow([os.path.relpath(v, VIDEOS),
                    int(c.get(cv2.CAP_PROP_FRAME_COUNT)),
                    round(c.get(cv2.CAP_PROP_FPS), 2),
                    int(c.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))])
        c.release()

print("listo")