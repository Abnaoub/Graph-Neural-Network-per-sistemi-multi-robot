#!/usr/bin/env python3
"""Genera il mondo Webots del magazzino a partire da core/warehouse.py (nessuna coordinata a mano).

Il mondo contiene:
- pavimento, muri, piazzole, scaffali e pilastri (stessa geometria del simulatore 2D);
- N robot (DEF AGV_0 ... AGV_{N-1}):
    --robot olonomo  (default) base cilindrica r 0,28 m con fisica, velocita' (vx, vy) imposta
                     dal supervisore con setVelocity: la policy viene eseguita come nel 2D;
    --robot pioneer  Pioneer 3-DX con Receiver e controller agv_controller (variante storica);
- N marcatori colorati dei goal (DEF GOAL_i), solo visivi;
- le linee del grafo di comunicazione (DEF EDGE_COORD) aggiornate dal supervisore;
- un robot "gnn_supervisor" (Supervisor) che esegue core/controller.py.

Uso:
    python scripts/genera_mondo_webots.py --robots 6 --policy gnn --episodes 20
Il file viene scritto in webots/worlds/warehouse_gnn.wbt.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import warehouse  # noqa: E402
from core.environment import make_scenario  # noqa: E402

COLORS = [(0.90, 0.10, 0.10), (0.10, 0.45, 0.95), (0.95, 0.75, 0.05), (0.55, 0.20, 0.85),
          (0.05, 0.75, 0.75), (0.95, 0.45, 0.70), (0.45, 0.30, 0.10), (0.20, 0.70, 0.25)]
AGV_HEIGHT = 0.22           # altezza del cilindro del robot olonomo
AGV_MASS = 20.0             # kg
AGV_COLOR = "0.36 0.55 0.93"    # tutti i robot dello stesso colore: si distinguono dalla targa "AMR-i"
GOAL_COLOR = "0.25 0.25 0.25"


def label(i: int) -> str:
    """Sigla mostrata su robot e goal (i robot restano AGV_i / agv_i nel codice, indice da 0)."""
    return f"AMR-{i + 1}"


def write_label_textures(n: int, worlds_dir: Path) -> bool:
    """Crea webots/worlds/textures/amr_<i>.png con la scritta 'AMR-i' (serve Pillow, incluso in matplotlib)."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("Pillow non disponibile: targhe senza testo")
        return False
    tex = worlds_dir / "textures"
    tex.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 96)
    except OSError:
        try:
            font = ImageFont.truetype("arialbd.ttf", 96)
        except OSError:
            font = ImageFont.load_default()
    for i in range(n):
        img = Image.new("RGB", (512, 256), "white")
        draw = ImageDraw.Draw(img)
        text = label(i)
        box = draw.textbbox((0, 0), text, font=font)
        draw.text(((512 - (box[2] - box[0])) / 2 - box[0], (256 - (box[3] - box[1])) / 2 - box[1]), text, fill="black", font=font)
        img.save(tex / f"amr_{i + 1}.png")
    return True


def plate(i: int, x: float, y: float, z: float, w: float, h: float, textured: bool) -> str:
    """Targa rettangolare con la scritta AMR-i (texture) appoggiata orizzontalmente."""
    if textured:
        app = (f'PBRAppearance {{ baseColorMap ImageTexture {{ url [ "textures/amr_{i + 1}.png" ] }} '
               f"roughness 1 metalness 0 }}")
    else:
        app = appearance("1 1 1")
    return (f"Pose {{ translation {x:.3f} {y:.3f} {z:.3f} children [ Shape {{ appearance {app} "
            f"geometry Box {{ size {w:g} {h:g} 0.006 }} }} ] }}")


def color(i: int) -> str:
    r, g, b = COLORS[i % len(COLORS)]
    return f"{r} {g} {b}"


def appearance(rgb: str, transparency: float = 0.0) -> str:
    extra = f" transparency {transparency}" if transparency else ""
    return f"PBRAppearance {{ baseColor {rgb} roughness 1 metalness 0{extra} }}"


def pose(x: float, y: float, z: float, rgb: str, geometry: str, rot: str = "", rough: float = 0.8,
         metal: float = 0.0) -> str:
    r = f" rotation {rot}" if rot else ""
    return (f"Pose {{ translation {x:.3f} {y:.3f} {z:.3f}{r} children [ Shape {{ appearance PBRAppearance "
            f"{{ baseColor {rgb} roughness {rough} metalness {metal} }} geometry {geometry} }} ] }}")


RACK_H = 1.2
UPRIGHT = "0.12 0.28 0.62"
BEAM = "0.95 0.45 0.05"
BOARD = "0.72 0.74 0.76"
CARTON = ("0.66 0.49 0.30", "0.74 0.57 0.36", "0.58 0.42 0.26")


def rack(name: str, x: float, y: float, k: int) -> str:
    """Scaffale industriale: stesso ingombro (0,4 x 0,6 m) del simulatore 2D."""
    sx, sy = warehouse.SHELF_SIZE[0], warehouse.SHELF_SIZE[1]
    hx, hy = sx / 2 - 0.02, sy / 2 - 0.02
    parts = []
    for px in (-hx, hx):
        for py in (-hy, hy):
            parts.append(pose(px, py, RACK_H / 2, UPRIGHT, f"Box {{ size 0.04 0.04 {RACK_H:g} }}", rough=0.4, metal=0.6))
    for lvl, z in enumerate((0.08, 0.48, 0.88)):
        parts.append(pose(0, 0, z, BOARD, f"Box {{ size {sx:g} {sy:g} 0.025 }}", rough=0.5, metal=0.5))
        for px in (-hx, hx):
            parts.append(pose(px, 0, z + 0.03, BEAM, f"Box {{ size 0.03 {sy:g} 0.05 }}", rough=0.5, metal=0.3))
        seed = (k * 7 + lvl * 3) % 5
        if seed == 4:
            continue
        boxes = [(-0.14, 0.24), (0.14, 0.22)] if seed % 2 else [(0.0, 0.30)]
        for j, (oy, bh) in enumerate(boxes):
            parts.append(pose(0, oy, z + 0.0125 + bh / 2, CARTON[(seed + j) % 3],
                              f"Box {{ size {sx - 0.08:.2f} {0.24 if len(boxes) > 1 else 0.44:.2f} {bh:.2f} }}", rough=0.9))
    body = " ".join(parts)
    return (f'Solid {{ translation {x:g} {y:g} 0 name "{name}" children [ {body} ] '
            f"boundingObject Pose {{ translation 0 0 {RACK_H / 2:g} children [ Box {{ size {sx:g} {sy:g} {RACK_H:g} }} ] }} }}")


def lane_markings() -> list[str]:
    """Linea gialla tratteggiata sulla fila centrale e frecce bianche del senso di marcia nei passaggi."""
    out = []
    x0, x1 = warehouse.SHELF_ZONE_X
    y = warehouse.LANE_SPLIT_Y
    half = warehouse.SHELF_SIZE[0] / 2 + 0.25
    x = x0
    while x < x1 - 0.2:
        if all(abs(x + 0.2 - cx) > half for cx in warehouse.SHELF_COLUMNS):
            out.append(pose(x + 0.2, y, 0.004, "1 0.85 0", "Box { size 0.4 0.05 0.002 }", rough=0.6))
        x += 0.7
    cols = sorted(warehouse.SHELF_COLUMNS)
    gaps = [(a + b) / 2 for a, b in zip(cols[:-1], cols[1:])]
    for _, lo, hi, _, lane in warehouse.passages():
        ay = (lo + hi) / 2
        direction = 1 if lane == "destra" else -1
        for gx in gaps:
            out.append(pose(gx - 0.1 * direction, ay, 0.004, "0.95 0.95 0.95", "Box { size 0.45 0.06 0.002 }", rough=0.6))
            for s in (1, -1):
                ang = s * 2.5 if direction > 0 else s * 0.64
                out.append(pose(gx + 0.12 * direction, ay + s * 0.06, 0.004, "0.95 0.95 0.95",
                                "Box { size 0.2 0.06 0.002 }", rot=f"0 0 1 {ang:.3f}", rough=0.6))
    return out


def zone_markings() -> list[str]:
    """Contorno tratteggiato delle zone critiche (incroci a prenotazione), solo visivo."""
    out = []
    for _, cx, cy, w, h in warehouse.critical_zones():
        for sx, sy, px, py in ((w, 0.03, 0, h / 2), (w, 0.03, 0, -h / 2), (0.03, h, w / 2, 0), (0.03, h, -w / 2, 0)):
            out.append(pose(cx + px, cy + py, 0.003, "0.85 0.15 0.15", f"Box {{ size {sx:g} {sy:g} 0.002 }}", rough=0.6))
    return out


def holonomic_agv(i: int, x: float, y: float, yaw: float, textured: bool = True) -> list[str]:
    """AGV olonomo: cilindro con fisica, ruote mecanum decorative, materiale 'agv' senza attrito."""
    r = warehouse.ROBOT_RADIUS
    wheels = []
    for wx, wy in ((0.2, 0.2), (-0.2, 0.2), (0.2, -0.2), (-0.2, -0.2)):
        wheels.append(f"Pose {{ translation {wx} {wy} {-AGV_HEIGHT / 2 + 0.05} rotation 1 0 0 1.5708 children [ Shape {{ "
                      f"appearance PBRAppearance {{ baseColor 0.15 0.15 0.15 roughness 0.9 metalness 0 }} "
                      f"geometry Cylinder {{ height 0.06 radius 0.05 }} }} ] }}")
    return [
        f"DEF AGV_{i} Robot {{",
        f"  translation {x:.3f} {y:.3f} {AGV_HEIGHT / 2 + 0.001:.3f}",
        f"  rotation 0 0 1 {yaw:.4f}",
        f'  name "agv_{i}"',
        '  controller "<none>"',
        '  contactMaterial "agv"',
        "  children [",
        f"    Shape {{ appearance PBRAppearance {{ baseColor {AGV_COLOR} roughness 0.6 metalness 0.2 }} "
        f"geometry Cylinder {{ height {AGV_HEIGHT:g} radius {r:g} }} }}",
        "    " + plate(i, 0.0, 0.0, AGV_HEIGHT / 2 + 0.004, 0.36, 0.18, textured),
        f"    Pose {{ translation {r - 0.06:.3f} 0 {AGV_HEIGHT / 2 + 0.005:.3f} children [ Shape {{ appearance "
        f"{appearance('0.1 0.1 0.1')} geometry Box {{ size 0.08 0.04 0.01 }} }} ] }}",
        "    " + " ".join(wheels),
        "  ]",
        f"  boundingObject Cylinder {{ height {AGV_HEIGHT:g} radius {r:g} }}",
        f"  physics Physics {{ mass {AGV_MASS:g} centerOfMass [ 0 0 -0.08 ] }}",
        "}",
    ]


def pioneer_agv(i: int, x: float, y: float, yaw: float, textured: bool = True) -> list[str]:
    return [
        f"DEF AGV_{i} Pioneer3dx {{",
        f"  translation {x:.3f} {y:.3f} 0.0975",
        f"  rotation 0 0 1 {yaw:.4f}",
        f'  name "agv_{i}"',
        '  controller "agv_controller"',
        "  extensionSlot [",
        "    Receiver { channel 1 }",
        "    " + plate(i, 0.0, 0.0, 0.25, 0.3, 0.15, textured),
        "  ]",
        "}",
    ]


def build_world(robots: int, seed: int, controller_args: list[str], robot: str = "olonomo",
                lanes: bool = True, edges: bool = True, worlds_dir: Path | None = None) -> str:
    warehouse.ONE_WAY = lanes
    scenario = make_scenario(robots, seed, "warehouse")
    textured = write_label_textures(robots, Path(worlds_dir) if worlds_dir else ROOT / "webots" / "worlds")
    w, h = warehouse.ARENA_W, warehouse.ARENA_H
    out = [
        "#VRML_SIM R2025a utf8",
        "",
        'EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/objects/backgrounds/protos/TexturedBackground.proto"',
        'EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/objects/backgrounds/protos/TexturedBackgroundLight.proto"',
        'EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/objects/floors/protos/RectangleArena.proto"',
        'EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/appearances/protos/RoughConcrete.proto"',
    ]
    if robot == "pioneer":
        out.append('EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/robots/adept/pioneer3/protos/Pioneer3dx.proto"')
    out += [
        "",
        "# FILE GENERATO da scripts/genera_mondo_webots.py: per cambiare il magazzino",
        "# modifica core/warehouse.py e rigenera il mondo.",
        "WorldInfo {",
        '  info [ "Magazzino automatizzato: navigazione cooperativa con GNN" ]',
        '  title "Warehouse GNN"',
        "  basicTimeStep 16",
        "  contactProperties [",
        "    # l'AGV olonomo riceve la velocita' direttamente dal supervisore: senza attrito col",
        "    # pavimento la velocita' eseguita e' esattamente quella della policy (come nel 2D)",
        '    ContactProperties { material1 "agv" material2 "default" coulombFriction [ 0 ] bounce 0 }',
        '    ContactProperties { material1 "agv" material2 "agv" coulombFriction [ 0 ] bounce 0 }',
        "  ]",
        "}",
        "Viewpoint {",
        "  orientation -0.4294 0.4178 0.8006 1.7934",
        "  position -1.0 -16.5 23.0",
        "}",
        'TexturedBackground { texture "factory" }',
        'TexturedBackgroundLight { texture "factory" }',
        "RectangleArena {",
        f"  floorSize {w:g} {h:g}",
        "  floorTileSize 2 2",
        "  floorAppearance RoughConcrete { colorOverride 0.85 0.85 0.85 }",
        f"  wallThickness {warehouse.WALL_T:g}",
        "  wallHeight 0.8",
        "}",
    ]
    for name, x, y, sx, sy, rgb, _ in warehouse.STATIONS:
        out.append(f"DEF {name} Pose {{ translation {x:g} {y:g} 0.005 children [ Shape {{ appearance "
                   f"{appearance(' '.join(f'{c:g}' for c in rgb), 0.3)} geometry Box {{ size {sx:g} {sy:g} 0.01 }} }} ] }}")
    for (x, y) in warehouse.SLOTS_LEFT + warehouse.SLOTS_RIGHT:
        out.append(f"Pose {{ translation {x:g} {y:g} 0.006 children [ Shape {{ appearance "
                   f"{appearance('0.3 0.3 0.3', 0.5)} geometry Cylinder {{ height 0.004 radius 0.24 }} }} ] }}")
    for k, (name, x, y) in enumerate(warehouse.shelves()):
        out.append(rack(name, x, y, k))
    if lanes:
        out.append(f"DEF CORSIE Group {{ children [ {' '.join(lane_markings())} ] }}")
    out.append(f"DEF ZONE_CRITICHE Group {{ children [ {' '.join(zone_markings())} ] }}")
    for k, (x, y, r) in enumerate(warehouse.PILLARS, start=1):
        out.append(f'DEF PILLAR_{k} Solid {{ translation {x:g} {y:g} 0.6 children [ Shape {{ appearance '
                   f'PBRAppearance {{ baseColor 1 0.8 0 roughness 0.6 metalness 0 }} geometry Cylinder '
                   f'{{ height 1.2 radius {r:g} }} }} ] name "pillar_{k}" boundingObject Cylinder {{ height 1.2 radius {r:g} }} }}')
    for i, (gx, gy) in enumerate(scenario.goals):
        # goal: anello grigio + targa "AMR-i" (il supervisore sposta il nodo GOAL_i a ogni episodio)
        out.append(f"DEF GOAL_{i} Pose {{ translation {gx:.3f} {gy:.3f} 0.012 children [ "
                   f"Shape {{ appearance {appearance(GOAL_COLOR, 0.4)} geometry Cylinder {{ height 0.01 radius 0.24 }} }} "
                   + plate(i, 0.0, 0.0, 0.008, 0.4, 0.2, textured) + " ] }")
    for i, ((x, y), (gx, gy)) in enumerate(zip(scenario.starts, scenario.goals)):
        yaw = math.atan2(gy - y, gx - x)
        out += holonomic_agv(i, x, y, yaw, textured) if robot == "olonomo" else pioneer_agv(i, x, y, yaw, textured)
    if edges:
        pairs = robots * (robots - 1) // 2
        points = " ".join(["0 0 -1"] * (2 * pairs))
        index = " ".join(f"{2 * k} {2 * k + 1} -1" for k in range(pairs))
        out.append("DEF GRAPH_EDGES Shape { appearance Appearance { material Material { diffuseColor 0 0.8 1 "
                   "emissiveColor 0 0.8 1 } } geometry IndexedLineSet { coord DEF EDGE_COORD Coordinate { "
                   f"point [ {points} ] }} coordIndex [ {index} ] }} castShadows FALSE }}")
    args = " ".join(f'"{a}"' for a in controller_args)
    out += [
        "Robot {",
        "  translation 0 0 -1",
        '  name "gnn_supervisor"',
        '  controller "gnn_supervisor"',
        f"  controllerArgs [ {args} ]",
        "  supervisor TRUE",
        "  children [ Emitter { channel 1 } ]",
        "}",
        "",
    ]
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robots", type=int, default=6, help=f"fino a {2 * len(warehouse.SLOTS_LEFT)}")
    parser.add_argument("--seed", type=int, default=1000, help="scenario del primo episodio")
    parser.add_argument("--policy", choices=["gnn", "mlp", "esperto"], default="gnn")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-time", type=float, default=90.0, help="secondi simulati per episodio")
    parser.add_argument("--robot", choices=["olonomo", "pioneer"], default="olonomo")
    parser.add_argument("--senza-corsie", action="store_true", help="disattiva le corsie a senso unico")
    parser.add_argument("--senza-zone", action="store_true", help="disattiva la prenotazione delle zone critiche")
    parser.add_argument("--senza-repulsione", action="store_true", help="disattiva l'evitamento degli scaffali")
    parser.add_argument("--senza-retromarcia", action="store_true", help="disattiva la retromarcia di emergenza")
    parser.add_argument("--filtro", choices=["emergenza", "simmetrico", "nessuno"], default="emergenza")
    parser.add_argument("--scaglionato", type=float, default=0.0, help="secondi tra una partenza e la successiva")
    parser.add_argument("--senza-archi", action="store_true", help="non disegnare gli archi del grafo")
    parser.add_argument("--output", default=str(ROOT / "webots" / "worlds" / "warehouse_gnn.wbt"))
    args = parser.parse_args()
    controller_args = ["--policy", args.policy, "--seed", str(args.seed), "--episodes", str(args.episodes),
                       "--max-time", str(args.max_time), "--robot", args.robot, "--filtro", args.filtro]
    for flag in ("senza_corsie", "senza_zone", "senza_repulsione", "senza_retromarcia"):
        if getattr(args, flag):
            controller_args.append("--" + flag.replace("_", "-"))
    if args.scaglionato > 0:
        controller_args += ["--scaglionato", str(args.scaglionato)]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_world(args.robots, args.seed, controller_args, robot=args.robot,
                                  lanes=not args.senza_corsie, edges=not args.senza_archi,
                                  worlds_dir=output.parent), encoding="utf-8")
    print(f"scritto {output} con {args.robots} robot ({args.robot}), policy={args.policy}")


if __name__ == "__main__":
    main()
