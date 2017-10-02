from yaw_controller import YawController
from pid import  PID
import rospy

MAX_SPEED_MPS = 44

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle, decel_limit, accel_limit,
                 mass, wheel_radius):

        self.vehicle_mass = mass
        self.wheel_radius = wheel_radius

        self.steering_controller = YawController(wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle)
        self.speed_pid = PID(5.0, 0.0001, 0.5, mn=decel_limit, mx=accel_limit)


    def control(self, cur_linear, cur_ang, target_linear, target_ang, time):
        speed_error = target_linear.x - cur_linear.x
        acc = self.speed_pid.step(speed_error, time)

        steer = self.steering_controller.get_steering(target_linear.x, target_ang.z, cur_linear.x)

        brk = 0
        if acc < 0.0:
            brk = -acc
            throttle = 0.0

            #http://www.asawicki.info/Mirror/Car%20Physics%20for%20Games/Car%20Physics%20for%20Games.html
            brk = brk * self.vehicle_mass * self.wheel_radius
        else:
            throttle = (cur_linear.x + acc) / MAX_SPEED_MPS

        rospy.logdebug_throttle(2, 'cur {}, target {}, throttle {}, break {}, steer {}'.format(
            cur_linear.x, target_linear.x, throttle, brk, steer))

        return acc, brk, steer

    def reset(self):
        self.speed_pid.reset()
