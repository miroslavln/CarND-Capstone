from yaw_controller import YawController
from pid import  PID
import rospy
from lowpass import LowPassFilter

MAX_TORQUE = 375  # NM this is from the specification for LINCOLN MKZ found online
MAX_THROTTLE_PERCENTAGE = 0.8

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle, decel_limit, accel_limit,
                 mass, wheel_radius, fuel_capacity, brake_deadband):

        self.vehicle_mass = mass + GAS_DENSITY*fuel_capacity
        self.wheel_radius = wheel_radius
        self.brake_deadband = brake_deadband

        self.steering_controller = YawController(wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle)
        self.acc_pid = PID(kp=1.0, ki=0.004, kd=0.1, mn=decel_limit, mx=accel_limit)

    def control(self, cur_linear, cur_ang, target_linear, target_ang, dt):
        speed_error = target_linear.x - cur_linear.x

        brake = 0.0
        throttle = 0.0

        acc = self.acc_pid.step(speed_error, dt)

        if speed_error < 0.0:
            #  http://www.asawicki.info/Mirror/Car%20Physics%20for%20Games/Car%20Physics%20for%20Games.html
            # torque = force * wheel_radius = mass * acceleration * wheel_radius
            torque = abs(acc) * self.vehicle_mass * self.wheel_radius
            brake = torque
        else:
            throttle = acc

        if throttle < 0.01:
            throttle = 0.0

        steer = self.steering_controller.get_steering(target_linear.x, target_ang.z, cur_linear.x)

        rospy.logdebug_throttle(2, 'cur {}, target {}, throttle {}, brake {}, steer {}'.format(
            cur_linear.x, target_linear.x, throttle, brake, steer))

        return throttle, brake, steer

    def reset(self):
        self.acc_pid.reset()
