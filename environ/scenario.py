from typing import Generator, Tuple, List

import matplotlib.pyplot as plt
import numpy as np

from environ.components import Vehicle, Zone, Human
from environ.utils import Spot


class Scenario:
    def __init__(self) -> None:
        self.vehicles: list[Vehicle] = []
        self.humans: list[Human] = []
        self.__tick = 0.1
        self.__elapsed = 0

    @property
    def tick(self) -> float:
        return self.__tick

    @property
    def elapsed(self) -> float:
        return self.__elapsed

    def step(self, zones: Generator[Zone, None, None]) -> Generator[Tuple[List[float], float, bool], None, None]:
        self.__elapsed += 1

        # everything steps forward, and collect base reward of all vehicles
        rewards = []
        for vehicle, zone in zip(self.vehicles, zones):
            # offline vehicles do not have reward
            if vehicle.offline:
                rewards.append(0.0)
                continue

            # base reward is the distance it moved in this step
            # further adjustments will be applied to it
            moved = vehicle.move(zone)
            rewards.append(moved * vehicle.priority)

        for human in self.humans:
            human.move()

        # all observations are locked here, back population of offline flags is a late update
        offline = []
        for i, vehicle in enumerate(self.vehicles):
            # early yield if this vehicle is already offline
            if vehicle.offline:
                dim = 4 + 8 * (len(self.vehicles) - 1) + 4 * len(self.humans)
                yield [0.0] * dim, rewards[i], True
                continue

            obs = [*vehicle.direction.t, vehicle.speed]
            reward = rewards[i]
            # occurs when vehicle safely finished its journey
            done = False
            # occurs when one vehicle hits a human or another vehicle
            crashed = False
            # a small punishment/warning before bad things happened
            warning = 0.0

            if vehicle.odometer >= 20:
                offline.append(i)
                reward = 5.0 * vehicle.priority
                done = True

            # observe other vehicles
            for other in self.vehicles[:i] + self.vehicles[i + 1:]:
                if other.offline:
                    obs.extend([0.0] * 8)
                    continue

                displacement = other.position - vehicle.position
                info = *displacement.t, *other.direction.t, other.speed, other.priority
                obs.extend(info)

                if warning or crashed:
                    continue

                distance = displacement.magnitude
                # too close
                if distance < 1:
                    offline.append(i)
                    crashed = True
                # encourage vehicles with high priority to move, and the others to wait
                elif distance < 3 and vehicle.priority < other.priority and vehicle.speed > 1:
                    warning = vehicle.speed / vehicle.v
                # encourage conservative actions when the surrounding gets complicated
                # cumulated with observation of humans
                elif distance < 10 and vehicle.speed > 5 and other.speed > 5:
                    reward *= 0.9 + distance * 0.01

            # observe all humans
            for human in self.humans:
                displacement = human.position - vehicle.position
                feared = displacement.magnitude < 3 and vehicle.speed > 1
                info = *displacement.t, feared
                obs.extend(info)

                if warning or crashed:
                    continue

                distance = displacement.magnitude
                # too close
                if distance < 1:
                    offline.append(i)
                    crashed = True
                # avoid human getting feared, and also a warning of hitting a human
                elif distance < 3 and vehicle.speed > 1:
                    warning = vehicle.speed / vehicle.v
                # encourage conservative actions when the surrounding gets complicated
                # cumulated with observation of vehicles
                elif distance < 10 and vehicle.speed > 5:
                    reward *= 0.9 + distance * 0.01

            if crashed:
                yield obs, -5.0, True
            elif warning:
                yield obs, -warning, False
            else:
                yield obs, reward, done

        # late update offline vehicles
        for i in offline:
            self.vehicles[i].offline = True

    def reset(self, vehicles=7, humans=6) -> Generator[List[float], None, None]:
        self.__elapsed = 0

        self.vehicles.clear()
        for _ in range(vehicles):
            start = Spot.normal(10, 1)
            priority = np.random.uniform(1, 3)
            vehicle = Vehicle(14, 7, priority, self.tick)
            vehicle.position = start
            vehicle.direction = -start.normalized
            vehicle.speed = np.random.normal(7, 1)
            self.vehicles.append(vehicle)

        self.humans.clear()
        for _ in range(humans):
            human = Human(1, self.tick)
            human.position = Spot.uniform(0, 7)
            human.direction = Spot.uniform(0, 1)
            self.humans.append(human)

        for i, vehicle in enumerate(self.vehicles):
            obs = [*vehicle.direction.t, vehicle.speed]
            for other in self.vehicles[:i] + self.vehicles[i + 1:]:
                displacement = other.position - vehicle.position
                info = *displacement.t, *other.direction.t, other.speed, other.priority
                obs.extend(info)
            for human in self.humans:
                displacement = human.position - vehicle.position
                feared = displacement.magnitude < 3 and vehicle.speed > 1
                info = *displacement.t, feared
                obs.extend(info)
            yield obs

    def render(self, vehicle=True, human=True) -> None:
        ax = plt.subplot(projection='3d')
        ax.set_xlim3d(-10, 10)
        ax.set_ylim3d(-10, 10)
        ax.set_zlim3d(-10, 10)
        ax.scatter(0, 0, 0, c='black')

        var = np.tanh(self.elapsed / 60)

        if human:
            for human in self.humans:
                ax.scatter(*human.position.t, c=[[0.5, 0.5, 0.5]], s=2)

        if vehicle:
            for vehicle in self.vehicles:
                ax.scatter(*vehicle.position.t, c=[[var, 0.0, 0.0]], s=2)

    @staticmethod
    def demo():
        def helper():
            while True:
                c = (0.7, 0.7)
                yield Zone(c, c, c)

        scenario = Scenario()

        for _ in scenario.reset():
            pass
        scenario.render()

        for _ in range(50):
            zones = helper()
            dones = []
            for _, _, done in scenario.step(zones):
                dones.append(done)
            scenario.render()

            if all(dones):
                break

        plt.show()
