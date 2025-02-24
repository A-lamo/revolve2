import numpy as np
import numpy.typing as npt
import math
from ..._modular_robot_control_interface import ModularRobotControlInterface
from ...body.base import ActiveHinge
from ...sensor_state._modular_robot_sensor_state import ModularRobotSensorState
from .._brain_instance import BrainInstance


class BrainInstanceFish(BrainInstance):

    _initial_state: npt.NDArray[np.float_]
    _output_mapping: list[tuple[int, ActiveHinge]]
    _weight_matrix: npt.NDArray[np.float_]
    _A: npt.NDArray[np.float_]
    _W: npt.NDArray[np.float_]
    _LAMBDA: np.float_


    def __init__(
        self,
        initial_state: npt.NDArray[np.float_],
        output_mapping: list[tuple[int, ActiveHinge]],
        weight_matrix: npt.NDArray[np.float_],
        A: npt.NDArray[np.float_],
        W: npt.NDArray[np.float_],
        LAMBDA: np.float_
    ) -> None:
        """
        Initialize this object.

        :param initial_state: The initial state of the neural network.
        :param weight_matrix: The weight matrix used during integration.
        :param output_mapping: Marks neurons as controller outputs and map them to the correct active hinge.
        """

        self._state = initial_state
        self._output_mapping = output_mapping
        self._weight_matrix = weight_matrix
        self._A = A
        self._W = W
        self._LAMBDA = LAMBDA

    @staticmethod
    def _rk45(
        state: npt.NDArray[np.float_], A: npt.NDArray[np.float_], dt: float
    ) -> npt.NDArray[np.float_]:
        A1: npt.NDArray[np.float_] = np.matmul(A, state)
        A2: npt.NDArray[np.float_] = np.matmul(A, (state + dt / 2 * A1))
        A3: npt.NDArray[np.float_] = np.matmul(A, (state + dt / 2 * A2))
        A4: npt.NDArray[np.float_] = np.matmul(A, (state + dt * A3))
        return state + dt / 6 * (A1 + 2 * (A2 + A3) + A4)

    @staticmethod
    def _step(
        state: npt.NDArray[np.float_],
        A: npt.NDArray[np.float_],
        W: npt.NDArray[np.float_],
        LAMBDA: np.float_,
        s: np.float_

        ) -> npt.NDArray[np.float_]:

        X = state[:len(state) // 2]
        Y = state[len(state) // 2:]
        
        if len(X) == 0:
            return state
        
        if len(X) == 1:
            x = X[0]
            y = Y[0]
            w = W[0]
            dx = -w*y + x*(A[0]**2 - x**2 - y**2) + LAMBDA*s
            dy =  w*x + y*(A[0]**2 - x**2 - y**2) - LAMBDA*s
            X[0] = x + dx
            Y[0] = y + dy
            return np.append(X, Y)

        for i in range(len(X)):
            
            x = X[i]
            y = Y[i]
            w = W[i]

            if i == 0:
                s = s
                dx = -w*y + x*(A[i]**2 - x**2 - y**2) + LAMBDA*s
                dy =  w*x + y*(A[i]**2 - x**2 - y**2) - LAMBDA*s
            
            elif i == len(X) - 1:
                s = s
                dx = -w*y + x*(A[i]**2 - x**2 - y**2) + LAMBDA*s
                dy =  w*x + y*(A[i]**2 - x**2 - y**2) - LAMBDA*s

            else:
                s = s
                dx = -w*y + x*(A[i]**2 - x**2 - y**2) + LAMBDA*s
                dy =  w*x + y*(A[i]**2 - x**2 - y**2) - LAMBDA*s

            X[i] = x + dx
            Y[i] = y + dy
        
        return np.append(X, Y)  
    

    def control(
        self,
        dt: float,
        sensor_state: ModularRobotSensorState,
        control_interface: ModularRobotControlInterface,
    ) -> None:

        # print('ACTIVE HINGE SENSOR STATE ANGULAR RATE: ',sensor_state.get_imu_sensor_state(ActiveHinge.sensor).specific_force)
        s_x = sensor_state.get_imu_sensor_state(ActiveHinge.sensor).specific_force[0] 
        s_y = sensor_state.get_imu_sensor_state(ActiveHinge.sensor).specific_force[1] 
        # get angle of specific force vector
        s = math.atan2(s_y, s_x)
        s *= (s_x + s_y)/20
        # print(s)

        # old_s = (s_x + s_y) / 20

        # Step the state forward in time.
        self._state = BrainInstanceFish._step(
            state=self._state, A=self._A, W=self._W, LAMBDA=self._LAMBDA, s=0
        )   

        # Integrate ODE to obtain new state.
        self._state = self._rk45(self._state, self._weight_matrix, dt) 

        # Set active hinge targets to match newly calculated state.
        for i, (state_index, active_hinge) in enumerate(self._output_mapping):
            # print('STATE-INDEX : ', state_index)
            control_interface.set_active_hinge_target(
                active_hinge, math.tanh(float(self._state[state_index]))*active_hinge.range
            )             
            # print("STATE : ", self._state)