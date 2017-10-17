from styx_msgs.msg import TrafficLight
import rospy
import numpy as np
import tensorflow as tf
import os
import cv2

class TLClassifier(object):
    def __init__(self, model_path):
        od_graph_def = tf.GraphDef()

        dir = os.path.dirname(__file__)
        file_name = os.path.join(dir, model_path)
        rospy.loginfo('Using inference model: {}'.format(file_name))
        with tf.gfile.GFile(file_name, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            self.graph = tf.import_graph_def(od_graph_def, name='')

        self.sess = tf.Session(graph=self.graph)
        self.image_tensor = self.sess.graph.get_tensor_by_name('image_tensor:0')
        self.detection_scores = self.sess.graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.sess.graph.get_tensor_by_name('detection_classes:0')
        self.detection_boxes = self.sess.graph.get_tensor_by_name('detection_boxes:0')

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.expand_dims(image, axis=0)
        scores, classes, boxes = self.sess.run([self.detection_scores, self.detection_classes, self.detection_boxes],
                                       feed_dict={self.image_tensor: image})
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)
        boxes = np.squeeze(boxes)

        index = np.argmax(scores)
        if scores[index] > 0.5:
            return self.convert_to_traffic_color(classes[index])

        return TrafficLight.UNKNOWN

    def convert_to_traffic_color(self, id):
        if id == 1:
            return TrafficLight.GREEN
        if id == 2:
            return TrafficLight.RED
        if id == 3:
            return TrafficLight.YELLOW

        return TrafficLight.UNKNOWN
