from styx_msgs.msg import TrafficLight
from keras.models import load_model
import cv2
import numpy as np
import tensorflow as tf

graph = []
class TLClassifier(object):
    def __init__(self):
        self.model = load_model('light_classification/model.h5')
        graph.append(tf.get_default_graph())

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        global graph
        with graph[0].as_default():
            image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
            transformed_image_array = image[None, :, :, :]
            light = self.model.predict(transformed_image_array, batch_size=1).squeeze()

            cls = np.argmax(light)
            if cls == 1:
                return TrafficLight.RED

            return TrafficLight.UNKNOWN
