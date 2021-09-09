import time

from transitions import Machine
import numpy as np

import rospy

from sensor_msgs.msg import JointState

STATES = ['capture', 'grasping']


class FSM():
    """Our intent is to continually check the robot at high rates to determine which state the robot is currently in."""

    def __init__(self,
                 capture_joint_positions,
                 rate = 15,
                 states = STATES):
        """Instantiate FSM.
        Args:
            states: List of all possible finite states for the mobile robot.
            rate: Rate in [Hz] at which we want to update the FSM.
        """

        # Init update rate.
        self.rate = rate

        # init Finite State Machine.
        self.FSM = Machine(model=self, states=states, initial='capture')
        # Add our user defined transitions.
        self.FSM.add_transition(trigger='grasping_object', source='capture', dest='grasping')
        self.FSM.add_transition(trigger='returning_to_capture', source='grasping', dest='capture')

        # init user defined default positions.
        self.tol = 0.075
        self.capture_joint_positions = capture_joint_positions

        # init subscriber for joint positions.
        self.joint_positions = np.zeros(shape=7)
        self.joint_positions_sub = rospy.Subscriber("/wam/joint_states", JointState, self.joint_positions_callback)

    def joint_positions_callback(self, joint_positions_msg):
        # TODO: create a buffer of joint_positions.
        self.joint_positions = np.array(joint_positions_msg.position)
        # print("joint_positions: {}".format(self.joint_positions))

    def is_joint_position_within_tol(self, array1, array2, tol):
        diff = np.sum(np.abs(array1) - np.abs(array2))
        return diff < tol

    def update_robot_state(self):
        """Function to continually update the robot's state."""

        while True:
            # throttle updates for FSM.
            time.sleep(1/self.rate)

            # init current state.
            state = self.state


            # check if we are home or capture postions.
            is_capture = True if \
                self.is_joint_position_within_tol(self.joint_positions, self.capture_joint_positions, self.tol) \
                else False

            if state == 'capture' and not is_capture:
                self.grasping_object()

            if state == 'grasping' and is_capture:
                self.returning_to_capture()

            if state != self.state:
                rospy.loginfo("--------> Robot's State: {}".format(self.state))


