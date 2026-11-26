"""Finto Supervisor Webots per eseguire gnn_supervisor.py senza Webots (test end-to-end).

Integra la cinematica del robot olonomo (posizione += velocita' * dt) e blocca il moto
contro scaffali e muri con core.warehouse.resolve_motion, come il simulatore 2D.
"""
from __future__ import annotations

import math

import numpy as np

from core import warehouse


class _Field:
    def __init__(self, value):
        self.value = list(value)

    def setSFVec3f(self, v):
        self.value = list(v)

    def setSFRotation(self, v):
        self.value = list(v)

    def setMFVec3f(self, index, v):
        pass


class _Node:
    def __init__(self, world, x=0.0, y=0.0):
        self.world = world
        self.fields = {"translation": _Field([x, y, 0.111]), "rotation": _Field([0, 0, 1, 0]), "point": _Field([])}
        self.velocity = [0.0] * 6

    def getField(self, name):
        return self.fields[name]

    def getPosition(self):
        return list(self.fields["translation"].value)

    def getOrientation(self):
        a = self.fields["rotation"].value[3]
        return [math.cos(a), -math.sin(a), 0, math.sin(a), math.cos(a), 0, 0, 0, 1]

    def getVelocity(self):
        return list(self.velocity)

    def setVelocity(self, v):
        self.velocity = list(v)

    def resetPhysics(self):
        self.velocity = [0.0] * 6


class _Emitter:
    def send(self, payload):
        pass


class FakeSupervisor:
    SIMULATION_MODE_PAUSE = 0

    def __init__(self, n: int, timestep: int = 16):
        self.timestep = timestep
        self.time = 0.0
        self.robots = [_Node(self) for _ in range(n)]
        self.goals = [_Node(self) for _ in range(n)]
        self.edge = _Node(self)
        self.obstacles = warehouse.warehouse_obstacles()

    def getBasicTimeStep(self):
        return self.timestep

    def getDevice(self, name):
        return _Emitter()

    def getFromDef(self, name):
        if name.startswith("AGV_"):
            k = int(name[4:])
            return self.robots[k] if k < len(self.robots) else None
        if name.startswith("GOAL_"):
            k = int(name[5:])
            return self.goals[k] if k < len(self.goals) else None
        if name == "EDGE_COORD":
            return self.edge
        return None

    def getTime(self):
        return self.time

    def step(self, ms):
        dt = ms / 1000.0
        old = np.asarray([r.getPosition()[:2] for r in self.robots], dtype=np.float32)
        vel = np.asarray([r.velocity[:2] for r in self.robots], dtype=np.float32)
        new, _ = warehouse.resolve_motion(old, old + dt * vel, self.obstacles)
        for r, p in zip(self.robots, new):
            r.fields["translation"].value[:2] = [float(p[0]), float(p[1])]
        self.time += dt
        return 0

    def simulationSetMode(self, mode):
        pass
