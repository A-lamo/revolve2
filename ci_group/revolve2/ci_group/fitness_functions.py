"""Standard fitness functions for modular robots."""

from copy import deepcopy
import math
import numpy as np
from revolve2.modular_robot_simulation import ModularRobotSimulationState

# def xy_displacement(
#     begin_state: ModularRobotSimulationState, end_state: ModularRobotSimulationState
# ) -> float:
#     """
#     Calculate the distance traveled on the xy-plane by a single modular robot.

#     :param begin_state: Begin state of the robot.
#     :param end_state: End state of the robot.
#     :returns: The calculated fitness.
#     """
#     begin_position = begin_state.get_pose().position
#     end_position = end_state.get_pose().position
#     return math.sqrt(
#         (begin_position.x - end_position.x) ** 2
#         + (begin_position.y - end_position.y) ** 2
#    )


def xy_displacement(x_distance: float, y_distance: float) -> float:
    """
    Goal:
        Calculate the distance traveled on the xy-plane by a single modular robot.
    -------------------------------------------------------------------------------------------
    Input:
        x_distance: The distance traveled in the x direction.
        y_distance: The distance traveled in the y direction.
    -------------------------------------------------------------------------------------------
    Output:
        The calculated fitness.
    """
    # Calculate the distance
    return math.sqrt((x_distance ** 2) + (y_distance ** 2))


def x_speed_Miras2021(x_distance: float, simulation_time = float) -> float:
    """Goal:
        Calculate the fitness for speed in x direction for a single modular robot according to 
            Miras (2021).
    -------------------------------------------------------------------------------------------
    Input:
        x_distance: The distance traveled in the x direction.
        simulation_time: The time of the simulation.
    -------------------------------------------------------------------------------------------
    Output:
        The calculated fitness.
    """
    # Begin and end Position

    # Calculate the speed in x direction
    vx = float((x_distance / simulation_time) * 100)
    if vx > 0:
        return vx
    elif vx == 0:
        return -0.1
    else:
        return vx / 10
    
def x_efficiency(xbest: float, eexp: float, simulation_time: float) -> float:
    """Goal:
        Calculate the efficiency of a robot for locomotion in x direction.
    -------------------------------------------------------------------------------------------
    Input:
        xbest: The furthest distance traveled in the x direction.
        eexp: The energy expended.
        simulation_time: The time of the simulation.
    -------------------------------------------------------------------------------------------
    Output:
        The calculated fitness.
    """
    def food(xbest, bmet):
        # Get food
        if xbest <= 0:
            return 0
        else:
            food = (xbest / 0.05) * (80 * bmet)
            return food
    
    def scale_EEXP(eexp, bmet):
        return eexp / 346 * (80 * bmet)
    
    # Get baseline metabolism
    bmet = 80
    battery = -bmet * simulation_time + food(xbest, bmet) - scale_EEXP(eexp, bmet)
    
    return battery


# ARON'S ADDITION
def get_x(state: ModularRobotSimulationState) -> float:
    """
    Get the x position of a single modular robot.

    :param state: State of the robot.
    :returns: The x position.
    """
    return state.get_pose().position.x
# ARON'S ADDITION
def get_y(state: ModularRobotSimulationState) -> float:
    """
    Get the x position of a single modular robot.

    :param state: State of the robot.
    :returns: The x position.
    """
    return state.get_pose().position.y