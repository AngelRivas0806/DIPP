import matplotlib.pyplot as plt
from PIL import Image

img = Image.open("THOR_MAGNI/maps3/1205_SC3_map.png").convert("RGB")

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(img)
ax.set_title("Da clic en esquina inferior izquierda y luego esquina superior derecha del workspace")
pts = plt.ginput(n=2, timeout=0)

print("Puntos:", pts)

(u1, v1), (u2, v2) = pts

U_MIN = min(u1, u2)
U_MAX = max(u1, u2)
V_MIN = min(v1, v2)
V_MAX = max(v1, v2)

print("U_MIN =", U_MIN)
print("U_MAX =", U_MAX)
print("V_MIN =", V_MIN)
print("V_MAX =", V_MAX)