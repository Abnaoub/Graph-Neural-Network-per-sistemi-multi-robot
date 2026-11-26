"""Controller Webots "agv_controller": esegue i comandi delle ruote ricevuti via radio.

Ogni Pioneer 3-DX si chiama "agv_<indice>". Il supervisore invia a tutti un
messaggio JSON {"ep": episodio, "w": [[sx, dx], ...]} e ogni robot prende la
coppia di velocita' (rad/s) con il proprio indice. Se per 0,5 s non arriva
nessun comando il robot si ferma (sicurezza).

Non serve numpy: questo controller usa solo la libreria standard.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

DEBUG_LOG = Path(__file__).resolve().parents[3] / "webots_debug.log"


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} [agv_controller] {msg}"
    print(line, flush=True)
    try:
        with DEBUG_LOG.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except OSError:
        pass

COMMAND_TIMEOUT = 0.5   # s


def parse_command(message: str, index: int) -> tuple[float, float] | None:
    """Estrae le velocita' (sinistra, destra) di questo robot dal messaggio."""
    try:
        wheels = json.loads(message)["w"]
        left, right = wheels[index]
        return float(left), float(right)
    except (ValueError, KeyError, IndexError, TypeError):
        return None


def main(robot=None) -> None:
    if robot is None:
        log("creo il Robot")
        from controller import Robot
        robot = Robot()
    log(f"avvio {robot.getName()}")
    timestep = int(robot.getBasicTimeStep())
    index = int(robot.getName().split("_")[-1])
    left = robot.getDevice("left wheel")
    right = robot.getDevice("right wheel")
    for motor in (left, right):
        motor.setPosition(float("inf"))     # controllo in velocita'
        motor.setVelocity(0.0)
    receiver = robot.getDevice("receiver")
    receiver.enable(timestep)
    last_command = robot.getTime()
    log(f"{robot.getName()} pronto (motori e ricevitore attivi)")
    first = True
    while robot.step(timestep) != -1:
        while receiver.getQueueLength() > 0:
            if hasattr(receiver, "getString"):
                message = receiver.getString()
            else:                            # versioni di Webots precedenti alla R2023b
                message = receiver.getData().decode("utf-8")
            receiver.nextPacket()
            command = parse_command(message, index)
            if command is not None:
                if first:
                    log(f"{robot.getName()}: primo comando ricevuto {command}")
                    first = False
                left.setVelocity(command[0])
                right.setVelocity(command[1])
                last_command = robot.getTime()
        if robot.getTime() - last_command > COMMAND_TIMEOUT:
            left.setVelocity(0.0)
            right.setVelocity(0.0)


if __name__ == "__main__":
    main()
