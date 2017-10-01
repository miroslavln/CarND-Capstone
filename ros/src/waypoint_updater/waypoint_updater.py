#!/usr/bin/env python
import copy

import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int32
from styx_msgs.msg import Lane, Waypoint

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200  # Number of waypoints we will publish. You can change this number

STOP_LINE_DISTANCE = 30.0
DECELERATION = 0.5  # m/s^2


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater', log_level=rospy.DEBUG)

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        self.base_waypoints = None
        self.pose = None
        self.light_wp = -1

        self.run()

    def get_distance(self, a, b):
        return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

    def get_closest_waypoint_index(self):
        closest = float('inf')
        index = 0
        for i in range(1, len(self.base_waypoints)):
            dist = self.get_distance(self.pose.position, self.base_waypoints[i].pose.pose.position)
            if dist < closest:
                closest = dist
                index = i

        return index

    def get_next_waypoint_index(self):
        index = self.get_closest_waypoint_index()

        waypoint = self.base_waypoints[index].pose.pose
        q = waypoint.orientation

        yaw = math.asin(2 * q.x * q.y + 2 * q.z * q.w)

        heading = math.atan2(waypoint.position.y - self.pose.position.y,
                             waypoint.position.x - self.pose.position.x)

        angle = abs(heading - yaw)
        if angle > math.pi / 4.0:
            index += 1

        return index

    def run(self):
        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            self.publish()
            rate.sleep()

    def publish(self):
        if not self.base_waypoints or not self.pose:
            return

        index = self.get_next_waypoint_index()
        waypoints = self.get_final_waypoints(index)

        waypoints = self.adjust_velocity(waypoints, index)
        final_waypoints_msg = Lane()
        final_waypoints_msg.waypoints = waypoints

        self.final_waypoints_pub.publish(final_waypoints_msg)

        rospy.logdebug_throttle(1, 'Position {}'.format(index))

    def adjust_velocity(self, waypoints, index):
        if self.light_wp < 0 or self.light_wp - index >= len(waypoints):
            return waypoints

        light = waypoints[self.light_wp - index]

        for i, waypoint in enumerate(waypoints):
            dist = self.get_distance(waypoint.pose.pose.position, light.pose.pose.position)
            dist -= STOP_LINE_DISTANCE

            if dist < -STOP_LINE_DISTANCE / 2:
                continue

            if dist < 0.0:
                velocity = 0.0
            else:
                # https://www.johannes-strommer.com/diverses/pages-in-english/stopping-distance-acceleration-speed/
                velocity = math.sqrt(2 * DECELERATION * dist)
                velocity = max(0.0, velocity)
                velocity = min(velocity, self.get_waypoint_velocity(waypoint))

            velocity = max(0.0, velocity)
            self.set_waypoint_velocity(waypoints, i, velocity)

        return waypoints

    def get_final_waypoints(self, index):
        #return copy.deepcopy(self.base_waypoints[index:index+LOOKAHEAD_WPS])
        res = []
        for i in range(LOOKAHEAD_WPS):
            wp = self.base_waypoints[(index + i) % len(self.base_waypoints)]
            res.append(copy.deepcopy(wp))
        return res


    def pose_cb(self, msg):
        self.pose = msg.pose

    def waypoints_cb(self, waypoints):
        self.base_waypoints = waypoints.waypoints

    def traffic_cb(self, msg):
        self.light_wp = msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)
        for i in range(wp1, wp2 + 1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
