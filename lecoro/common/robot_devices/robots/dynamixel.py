from abc import ABC, abstractmethod
import numpy as np

from lecoro.common.robot_devices.motors.dynamixel import (
    DynamixelMotorsBus,
    CalibrationMode,
    TorqueMode,
    DYNAMIXEL_OPERATION_MODE,
    INTERBOTIX_MOTORMODELS
)
from lecoro.common.robot_devices.robots.dynamixel_calibration import convert_degrees_to_steps, compute_nearest_rounded_position, apply_drive_mode
from lecoro.common.robot_devices.utils import RobotDeviceAlreadyConnectedError, RobotDeviceNotConnectedError

URL_TEMPLATE = (
    "https://raw.githubusercontent.com/huggingface/lerobot/main/media/{robot}/{arm}_{position}.webp"
)

# The following positions are provided in nominal degree range ]-180, +180[
# For more info on these constants, see comments in the code where they get used.
ZERO_POSITION_DEGREE = 0
ROTATED_POSITION_DEGREE = 90

def require_connection(method):
    """Decorator to check if the device is connected before executing the method."""
    def wrapper(self, *args, **kwargs):
        if not getattr(self, "is_connected", False):
            raise RobotDeviceNotConnectedError(f"{self.__class__.__name__} is not connected.")
        return method(self, *args, **kwargs)
    return wrapper


class DynamixelActuator:
    def __init__(self, port, motors, mock=False, moving_time=2.0, accel_time=0.3):
        self.bus = DynamixelMotorsBus(port, motors, mock)
        self.moving_time = moving_time
        self.accel_time = accel_time

    @require_connection
    def set_presets(self, robot_type):
        # dynamixel servos handle shadows automatically
        if "shoulder_shadow" in self.joint_names:
            shoulder_idx = self.bus.read("ID", "shoulder")
            self.bus.write("Secondary_ID", shoulder_idx, "shoulder_shadow")

        if "elbow_shadow" in self.joint_names:
            elbow_idx = self.bus.read("ID", "elbow")
            self.bus.write("Secondary_ID", elbow_idx, "elbow_shadow")

        # set the drive mode to time-based profile to set moving time via velocity profiles
        drive_mode = self.bus.read('Drive_Mode')
        for i in range(len(self.joint_names)):
            drive_mode[i] |= 1 << 2  # set third bit to enable time-based profiles
        self.bus.write('Drive_Mode', drive_mode)

        if robot_type == 'follower':
            self.bus.write("Velocity_Limit", 131)

        all_joints_except_gripper = [name for name in self.joint_names if name != "gripper"]
        if len(all_joints_except_gripper) > 0:
            # 4 corresponds to Extended Position on Aloha motors
            self.set_op_mode('extended_position', joint_names=all_joints_except_gripper)

        # Use 'position control current based' for follower gripper to be limited by the limit of the current.
        # It can grasp an object without forcing too much even tho,
        # it's goal position is a complete grasp (both gripper fingers are ordered to join and reach a touch).
        # 5 corresponds to Current Controlled Position on Aloha gripper follower "xm430-w350"
        if robot_type == 'follower' and 'gripper' in self.joint_names and self.bus.motors['gripper'][1] != 'xc430-w150':
            self.set_op_mode('current_based_position', joint_names=['gripper'])
            #self.bus.write('Current_Limit', motor_names=['gripper'], values=[500.0])

        # set profiles after setting operation modes
        self.set_trajectory_time(moving_time=self.moving_time, accel_time=self.accel_time)

    @property
    def joint_names(self):
        return self.bus.motor_names

    @property
    def motor_models(self):
        return self.bus.motor_models

    @property
    def is_connected(self):
        return self.bus.is_connected

    def connect(self):
        self.bus.connect()

    @require_connection
    def disconnect(self):
        self.bus.disconnect()

    def run_arm_calibration(self, robot_type, arm_name, arm_type):
        # arm_type must be in {"follower", "leader"}

        """This function ensures that a neural network trained on data collected on a given robot
        can work on another robot. For instance before calibration, setting a same goal position
        for each motor of two different robots will get two very different positions. But after calibration,
        the two robots will move to the same position.To this end, this function computes the homing offset
        and the drive mode for each motor of a given robot.

        Homing offset is used to shift the motor position to a ]-2048, +2048[ nominal range (when the motor uses 2048 steps
        to complete a half a turn). This range is set around an arbitrary "zero position" corresponding to all motor positions
        being 0. During the calibration process, you will need to manually move the robot to this "zero position".

        Drive mode is used to invert the rotation direction of the motor. This is useful when some motors have been assembled
        in the opposite orientation for some robots. During the calibration process, you will need to manually move the robot
        to the "rotated position".

        After calibration, the homing offsets and drive modes are stored in a cache.

        Example of usage:
        ```python
        run_arm_calibration(arm, "koch", "left", "follower")
        ```
        """
        if (self.read_register("Torque_Enable") != TorqueMode.DISABLED.value).any():
            raise ValueError("To run calibration, the torque must be disabled on all motors.")

        print(f"\nRunning calibration of {robot_type} {arm_name} {arm_type}...")

        print("\nMove arm to zero position")
        print("See: " + URL_TEMPLATE.format(robot=robot_type, arm=arm_type, position="zero"))
        input("Press Enter to continue...")

        # We arbitrarily chose our zero target position to be a straight horizontal position with gripper upwards and closed.
        # It is easy to identify and all motors are in a "quarter turn" position. Once calibration is done, this position will
        # correspond to every motor angle being 0. If you set all 0 as Goal Position, the arm will move in this position.
        zero_target_pos = convert_degrees_to_steps(ZERO_POSITION_DEGREE, self.motor_models)

        # Compute homing offset so that `present_position + homing_offset ~= target_position`.
        zero_pos = self.get_joint_positions(apply_calibration=False)
        zero_nearest_pos = compute_nearest_rounded_position(zero_pos, self.motor_models)
        homing_offset = zero_target_pos - zero_nearest_pos

        # The rotated target position corresponds to a rotation of a quarter turn from the zero position.
        # This allows to identify the rotation direction of each motor.
        # For instance, if the motor rotates 90 degree, and its value is -90 after applying the homing offset, then we know its rotation direction
        # is inverted. However, for the calibration being successful, we need everyone to follow the same target position.
        # Sometimes, there is only one possible rotation direction. For instance, if the gripper is closed, there is only one direction which
        # corresponds to opening the gripper. When the rotation direction is ambiguous, we arbitrarely rotate clockwise from the point of view
        # of the previous motor in the kinetic chain.
        print("\nMove arm to rotated target position")
        print("See: " + URL_TEMPLATE.format(robot=robot_type, arm=arm_type, position="rotated"))
        input("Press Enter to continue...")

        rotated_target_pos = convert_degrees_to_steps(ROTATED_POSITION_DEGREE, self.motor_models)

        # Find drive mode by rotating each motor by a quarter of a turn.
        # Drive mode indicates if the motor rotation direction should be inverted (=1) or not (=0).
        rotated_pos = self.get_joint_positions(apply_calibration=False)
        drive_mode = (rotated_pos < zero_pos).astype(np.int32)

        # Re-compute homing offset to take into account drive mode
        rotated_drived_pos = apply_drive_mode(rotated_pos, drive_mode)
        rotated_nearest_pos = compute_nearest_rounded_position(rotated_drived_pos, self.motor_models)
        homing_offset = rotated_target_pos - rotated_nearest_pos

        print("\nMove arm to rest position")
        print("See: " + URL_TEMPLATE.format(robot=robot_type, arm=arm_type, position="rest"))
        input("Press Enter to continue...")
        print()

        # Joints with rotational motions are expressed in degrees in nominal range of [-180, 180]
        calib_mode = [CalibrationMode.DEGREE.name] * len(self.motor_models)

        if "gripper" in self.joint_names:
            # Joints with linear motions (like gripper of Aloha) are experessed in nominal range of [0, 100]
            calib_idx = self.joint_names.index("gripper")
            calib_mode[calib_idx] = CalibrationMode.LINEAR.name

        calib_data = {
            "homing_offset": homing_offset.tolist(),
            "drive_mode": drive_mode.tolist(),
            "start_pos": zero_pos.tolist(),
            "end_pos": rotated_pos.tolist(),
            "calib_mode": calib_mode,
            "motor_names": self.joint_names,
        }
        return calib_data

    def set_calibration(self, calibration):
        self.bus.set_calibration(calibration)

    def apply_calibration(self, values, joint_names=None):
        if joint_names is None:
            joint_names = self.joint_names

        assert len(values) == len(joint_names), \
            f"DynamixelManipulator({self.bus.port}): number of values ({len(values)}) must match number of joints ({len(joint_names)})"

        return self.bus.apply_calibration(values, joint_names)

    def revert_calibration(self, values, joint_names=None):
        if joint_names is None:
            joint_names = self.joint_names

        assert len(values) == len(joint_names), \
            f"DynamixelManipulator({self.bus.port}): number of values ({len(values)}) must match number of joints ({len(joint_names)})"

        return self.bus.revert_calibration(values, joint_names)

    @require_connection
    def get_joint_positions(self, apply_calibration=True):
        if not apply_calibration:
            _calibration = self.bus.calibration
            self.bus.set_calibration(None)

        joint_positions = np.array(self.bus.read("Present_Position"))

        if not apply_calibration:
            self.bus.set_calibration(_calibration)
        return joint_positions

    @require_connection
    def get_single_joint_position(self, joint_name, apply_calibration=True):
        if not apply_calibration:
            _calibration = self.bus.calibration
            self.bus.set_calibration(None)

        joint_position = np.array(self.bus.read("Present_Position", motor_names=[joint_name]))

        if not apply_calibration:
            self.bus.set_calibration(_calibration)
        return joint_position[0]

    @require_connection
    def set_joint_positions(self, joint_positions, moving_time=None, accel_time=None, apply_calibration=True, blocking=True):
        self.set_trajectory_time(moving_time, accel_time)
        if not apply_calibration:
            _calibration = self.bus.calibration
            self.bus.set_calibration(None)

        self.bus.write("Goal_Position", joint_positions)

        if not apply_calibration:
            self.bus.set_calibration(_calibration)

        if blocking:
            while False:
                # todo: check moving status bit
                pass

    @require_connection
    def set_single_joint_position(self, joint_name, position, moving_time=None, accel_time=None, apply_calibration=True, blocking=True):
        self.set_trajectory_time(moving_time, accel_time)
        if not apply_calibration:
            _calibration = self.bus.calibration
            self.bus.set_calibration(None)

        self.bus.write("Goal_Position", position, motor_names=[joint_name])

        if not apply_calibration:
            self.bus.set_calibration(_calibration)

        if blocking:
            while False:
                # todo: check moving status bit
                pass

    @require_connection
    def set_trajectory_time(self, moving_time=None, accel_time=None):
        if moving_time is not None:
            self.moving_time = moving_time
            self.bus.write("Profile_Velocity", int(moving_time * 1000))
        if accel_time is not None:
            self.accel_time = accel_time
            self.bus.write("Profile_Acceleration", int(accel_time * 1000))

    @require_connection
    def set_op_mode(self, op_mode, joint_names=None):
        if op_mode not in DYNAMIXEL_OPERATION_MODE.keys():
            raise ValueError(f"DynamixelManipulator({self.bus.port}): Unknown operation mode {op_mode}!")

        if joint_names is None:
            joint_names = self.joint_names
        self.bus.write("Operating_Mode", DYNAMIXEL_OPERATION_MODE[op_mode], joint_names)

    @require_connection
    def set_pid_gains(self, p_gain=None, i_gain=None, d_gain=None, joint_names=None):
        if joint_names is None:
            joint_names = self.joint_names

        if p_gain is not None:
            self.bus.write("Position_P_Gain", p_gain, joint_names)
        if i_gain is not None:
            self.bus.write("Position_I_Gain", i_gain, joint_names)
        if d_gain is not None:
            self.bus.write("Position_D_Gain", d_gain, joint_names)

    @require_connection
    def torque_on(self, joint_names=None):
        if joint_names is None:
            joint_names = self.joint_names
        self.bus.write("Torque_Enable", TorqueMode.ENABLED.value, joint_names)

    @require_connection
    def torque_off(self, joint_names=None):
        if joint_names is None:
            joint_names = self.joint_names
        self.bus.write("Torque_Enable", TorqueMode.DISABLED.value, joint_names)

    @require_connection
    def read_register(self, register_name, joint_names=None):
        if joint_names is None:
            joint_names = self.joint_names
        return self.bus.read(register_name, joint_names)


class InterbotixActuator(DynamixelActuator):
    ARM_GROUP = set(['waist', 'shoulder', 'elbow', 'forearm_roll', 'wrist_angle', 'wrist_rotate'])
    def __init__(self,
                 robot_name,
                 robot_model,
                 init_node,
                 moving_time=2.0,
                 accel_time=0.3,
                 gripper_pressure=0.5,
                 gripper_pressure_lower_limit=150,
                 gripper_pressure_upper_limit=350):
        from interbotix_xs_modules.arm import InterbotixManipulatorXS
        from interbotix_xs_msgs.msg import JointSingleCommand

        self.manipulator = None
        self.gripper_command = JointSingleCommand(name="gripper")
        self._manipulator_cls = InterbotixManipulatorXS
        self._manipulator_kwargs = {
            "robot_name": robot_name,
            "robot_model": robot_model,
            "init_node": init_node,
            "moving_time": moving_time,
            "accel_time": accel_time,
            "gripper_pressure": gripper_pressure,
            "gripper_pressure_lower_limit": gripper_pressure_lower_limit,
            "gripper_pressure_upper_limit": gripper_pressure_upper_limit
        }
        self.bus = DynamixelMotorsBus("", list(self.ARM_GROUP) + ["gripper"], mock=True)
        self._is_connected = False

    ### interface

    @property
    @require_connection
    def joint_names(self):
        return self.manipulator.group_info.joint_names + ["gripper"]

    @property
    @require_connection
    def joint_names(self):
        return INTERBOTIX_MOTORMODELS[self.robot_model]

    @property
    def is_connected(self):
        return self._is_connected

    def connect(self):
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                f"InterbotixManipulator({self._manipulator_kwargs['robot_name']}) is already connected. Do not call `manipulator.connect()` twice."
            )

        self.manipulator = self._manipulator_cls(group_name="arm", gripper_name="gripper", **self._manipulator_kwargs)
        self._is_connected = True

    @require_connection
    def disconnect(self):
        del self.manipulator
        self._is_connected = False

    def set_calibration(self, calibration):
        self.bus.set_calibration(calibration)

    def apply_calibration(self, values, joint_names=None):
        if joint_names is None:
            joint_names = self.joint_names

        assert len(values) == len(joint_names), \
            f"DynamixelManipulator({self.bus.port}): number of values ({len(values)}) must match number of joints ({len(joint_names)})"

        return self.bus.apply_calibration(self, values, joint_names)

    def revert_calibration(self, values, joint_names=None):
        if joint_names is None:
            joint_names = self.joint_names

        assert len(values) == len(joint_names), \
            f"DynamixelManipulator({self.bus.port}): number of values ({len(values)}) must match number of joints ({len(joint_names)})"

        return self.bus.revert_calibration(self, values, joint_names)

    @require_connection
    def get_joint_positions(self, apply_calibration=True):
        joint_positions = self.manipulator.dxl.robot_get_joint_states().position
        if apply_calibration:
            joint_positions = self.apply_calibration(joint_positions, self.joint_names)
        return joint_positions

    @require_connection
    def get_single_joint_position(self, joint_name, apply_calibration=True):
        joint_position = self.manipulator.dxl.robot_get_single_joint_state(joint_name).position
        if apply_calibration:
            joint_position = self.apply_calibration(np.array([joint_position]), [joint_name])[0]
        return joint_position

    @require_connection
    def set_joint_positions(self, joint_positions, moving_time=None, accel_time=None, apply_calibration=True, blocking=True):
        if apply_calibration:
            joint_positions = self.revert_calibration(joint_positions)

        self.gripper_command.cmd = joint_positions[-1]
        self.manipulator.gripper.core.pub_single.publish(self.gripper_command)

        arm_positions = joint_positions[:-1]
        self.manipulator.arm.set_joint_positions(arm_positions, moving_time, accel_time, blocking)

    @require_connection
    def set_single_joint_position(self, joint_name, position, moving_time=None, accel_time=None, apply_calibration=True, blocking=True):
        if apply_calibration:
            position = self.revert_calibration([position], [joint_name])[0]

        if joint_name == "gripper":
            self.gripper_command.cmd = position
            self.manipulator.gripper.core.pub_single.publish(self.gripper_command)
        else:
            self.manipulator.arm.set_single_joint_position(joint_name, position, moving_time, accel_time, blocking)

    @require_connection
    def set_trajectory_time(self, moving_time=None, accel_time=None):
        self.manipulator.arm.set_trajectory_time(moving_time, accel_time)

    @require_connection
    def set_op_mode(self, op_mode, joint_names=None):
        if op_mode not in DYNAMIXEL_OPERATION_MODE.keys():
            raise ValueError(f"InterbotixManipulator({self._manipulator_kwargs['robot_name']}): Unknown operation mode {op_mode}!")
        use_group_cmd, remaining_joint_names = self._use_group_cmd(joint_names)

        if use_group_cmd:
            self.manipulator.dxl.robot_set_operating_modes("group", "arm", op_mode)
        for joint_name in remaining_joint_names:
            self.manipulator.dxl.robot_set_operating_modes("single", joint_name, op_mode)

    @abstractmethod
    def set_pid_gains(self, p_gain=None, i_gain=None, d_gain=None, joint_names=None):
        use_group_cmd, remaining_joint_names = self._use_group_cmd(joint_names)

        register_names = ['Position_P_Gain', 'Position_I_Gain', 'Position_D_Gain']
        for register, gain in zip(register_names, [p_gain, i_gain, d_gain]):
            if gain is None:
                continue

            if use_group_cmd:
                self.manipulator.dxl.robot_set_motor_registers("group", "arm", register, gain)
            for joint_name in remaining_joint_names:
                self.manipulator.dxl.robot_set_operating_modes("single", joint_name, register, gain)

    @require_connection
    def torque_on(self, joint_names=None):
        use_group_cmd, remaining_joint_names = self._use_group_cmd(joint_names)

        if use_group_cmd:
            self.manipulator.dxl.robot_torque_enable("group", "arm", True)
        for joint_name in remaining_joint_names:
            self.manipulator.dxl.robot_torque_enable("single", joint_name, True)

    @require_connection
    def torque_off(self, joint_names=None):
        use_group_cmd, remaining_joint_names = self._use_group_cmd(joint_names)

        if use_group_cmd:
            self.manipulator.dxl.robot_torque_enable("group", "arm", False)
        for joint_name in remaining_joint_names:
            self.manipulator.dxl.robot_torque_enable("single", joint_name, False)

    @require_connection
    def read_register(self, register_name, joint_names=None):
        use_group_cmd, remaining_joint_names = self._use_group_cmd(joint_names)

        if use_group_cmd:
            self.manipulator.dxl.robot_get_motor_registers("group", "arm", register_name)
        for joint_name in remaining_joint_names:
            self.manipulator.dxl.robot_get_motor_registers("single", joint_name, register_name)

    ### ee control
    @require_connection
    def set_ee_pose(self, x=0, y=0, z=0, roll=0, pitch=0, yaw=None, custom_guess=None, execute=True, moving_time=None, accel_time=None, blocking=True):
        return self.manipulator.arm.set_ee_pos(x, y, z, roll, pitch, yaw, custom_guess, execute, moving_time, accel_time, blocking)

    @require_connection
    def set_ee_cartesian_trajectory(self, x=0, y=0, z=0, roll=0, pitch=0, yaw=0, moving_time=None, wp_moving_time=0.2, wp_accel_time=0.1, wp_period=0.05):
        return self.manipulator.arm.set_ee_pos_trajectory(x, y, z, roll, pitch, yaw, moving_time, wp_moving_time, wp_accel_time, wp_period)

    @require_connection
    def get_ee_pose(self):
        return self.manipulator.arm.get_ee_pose()

    ### utilities
    def _use_group_cmd(self, joint_names=None):
        # if joint_names is a proper subset of all joints in the arm,
        # returns (true, remaining_joint_names)
        # otherwise returns (false, joint_names)
        if joint_names is None:
            joint_names = self.joint_names

        _joint_names = set(joint_names)
        if _joint_names.issubset(self.ARM_GROUP):
            return True, list(_joint_names.difference(self.ARM_GROUP))
        else:
            return False, joint_names

