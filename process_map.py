import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os


WIDTH_M = 18.8
HEIGHT_M = 8.6

# Estos valores los tienes que medir/clickear en la imagen
# U_MIN = 5.15375259032362
# U_MAX = 1780.6676341497948
# V_MIN = 691.9201665529711
# V_MAX = 1714.2773930727349
U_MIN = 2.493506493506402
U_MAX = 1787.883116883117
V_MIN = 936.4480519480519
V_MAX = 1708.590909090909
# Cambia según tu sistema:
# "center" -> x [-9.4, 9.4], y [-4.3, 4.3]
# "corner" -> x [0, 18.8], y [0, 8.6]
ORIGIN_MODE = "center"

# Si se ve volteado verticalmente, cambia a False
FLIP_Y = True


def pixel_to_world(u, v):
    """
    Convierte pixel a coordenada mundo usando SOLO el rectángulo útil del workspace.
    Origen en el centro del escenario.
    """

    x = ((u - U_MIN) / (U_MAX - U_MIN)) * WIDTH_M - WIDTH_M / 2.0

    # flip_y=True porque en imagen v crece hacia abajo
    y = (1.0 - ((v - V_MIN) / (V_MAX - V_MIN))) * HEIGHT_M - HEIGHT_M / 2.0

    return x, y


def save_polygons(polygons, output_txt):
    os.makedirs(os.path.dirname(output_txt), exist_ok=True)

    with open(output_txt, "w") as f:
        for poly in polygons:
            for x, y in poly:
                f.write(f"{x:.6f} {y:.6f}\n")
            f.write("\n\n")

    print(f"[SAVE] Guardados {len(polygons)} polígonos en {output_txt}")


def draw_polygon_from_clicks(image_path, output_txt):
    img = Image.open(image_path).convert("RGB")
    W_px, H_px = img.size

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(img)
    ax.set_title(
        "Clic en vértices. Enter para cerrar polígono. "
        "Cierra ventana o Ctrl+C para terminar."
    )
    ax.axis("on")

    all_polygons = []

    # Si ya existe un archivo previo, pregunta/puedes cargarlo después si quieres.
    if os.path.exists(output_txt):
        print(f"[WARN] Ya existe {output_txt}. Se sobrescribirá al guardar.")

    try:
        while True:
            print("\nDa clic en los vértices de un polígono.")
            print("Presiona Enter cuando termines ese polígono.")
            print("Cierra la ventana o Ctrl+C para finalizar.")

            try:
                pts = plt.ginput(n=-1, timeout=0)
            except Exception as e:
                print("[INFO] Se cerró o falló la ventana de matplotlib:", repr(e))
                break

            if len(pts) == 0:
                print("[INFO] No se recibieron puntos. Terminando.")
                break

            poly_world = []

            for u, v in pts:
                x, y = pixel_to_world(u, v)
                poly_world.append([x, y])

            # cerrar polígono
            if np.linalg.norm(np.array(poly_world[0]) - np.array(poly_world[-1])) > 1e-6:
                poly_world.append(poly_world[0])

            poly_world = np.array(poly_world, dtype=np.float32)
            all_polygons.append(poly_world)

            # Dibujar encima
            poly_px = np.array(pts + [pts[0]], dtype=np.float32)
            ax.plot(poly_px[:, 0], poly_px[:, 1], linewidth=2)
            fig.canvas.draw_idle()
            plt.pause(0.01)

            print(f"Polígono agregado con {len(poly_world)} puntos.")

            # Guardar inmediatamente después de cada polígono
            save_polygons(all_polygons, output_txt)

    except KeyboardInterrupt:
        print("\n[INFO] Interrumpido con Ctrl+C.")

    finally:
        # Guardado final por seguridad
        if len(all_polygons) > 0:
            save_polygons(all_polygons, output_txt)
        else:
            print("[WARN] No se guardó nada porque no hay polígonos.")

        plt.close(fig)


if __name__ == "__main__":
    draw_polygon_from_clicks(
        image_path="THOR_MAGNI/maps3/1205_SC3_map.png",
        output_txt="mapa/THOR_MAGNI_120522_SC3/obstacles.txt",
    )