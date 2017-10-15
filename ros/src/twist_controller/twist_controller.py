from yaw_controller import YawController
from pid import  PID
import rospy

from ros.src.twist_controller.lowpass import LowPassFilter

MAX_TORQUE = 375  # NM this is from the specification for LINCOLN MKZ found online
MAX_THROTTLE_PERCENTAGE = 0.5

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle, decel_limit, accel_limit,
                 mass, wheel_radius, fuel_capacity, break_deadband):

        self.vehicle_mass = mass + GAS_DENSITY*fuel_capacity
        self.wheel_radius = wheel_radius
        self.break_deadband = break_deadband

        self.steering_controller = YawController(wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle)
        self.speed_pid = PID(1.0, 0.0001, 0.5, mn=decel_limit, mx=accel_limit)


    def control(self, cur_linear, cur_ang, target_linear, target_ang, time):
        speed_error = target_linear.x - cur_linear.x
        acc = self.speed_pid.step(speed_error, time)


        #  http://www.asawicki.info/Mirror/Car%20Physics%20for%20Games/Car%20Physics%20for%20Games.html
        # torque = force * wheel_radius = mass * acceleration * wheel_radius
        torque = abs(acc) * self.vehicle_mass * self.wheel_radius

        break_torque = 0.0
        if acc < 0.0:
            throttle = 0.0
            if torque > self.break_deadband:
                break_torque = torque
        else:
            throttle = min(torque / MAX_TORQUE, MAX_THROTTLE_PERCENTAGE)

        steer = self.steering_controller.get_steering(target_linear.x, target_ang.z, cur_linear.x)

        rospy.logdebug_throttle(2, 'cur {}, target {}, throttle {}, break {}, steer {}'.format(
            cur_linear.x, target_linear.x, throttle, break_torque, steer))

        return throttle, break_torque, steer

    def reset(self):
        self.speed_pid.reset()
