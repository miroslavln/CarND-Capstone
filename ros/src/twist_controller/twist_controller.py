from yaw_controller import YawController
from pid import  PID
import rospy

MAX_TORQUE = 375  # NM this is from the specification for LINCOLN MKZ found online


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

        #  http://www.asawicki.info/Mirror/Car%20Physics%20for%20Games/Car%20Physics%20for%20Games.html
        # torque = force * wheel_radius = mass * acceleration * wheel_radius
        torque = abs(acc) * self.vehicle_mass * self.wheel_radius

        brk = 0.0
        if acc < 0.0:
            throttle = 0.0
            brk = torque
        else:
            throttle = min(torque / MAX_TORQUE, 0.8)  # convert to throttle percentage without going over 80%

        rospy.logdebug_throttle(2, 'cur {}, target {}, throttle {}, break {}, steer {}'.format(
            cur_linear.x, target_linear.x, throttle, brk, steer))

        return throttle, brk, steer

    def reset(self):
        self.speed_pid.reset()
