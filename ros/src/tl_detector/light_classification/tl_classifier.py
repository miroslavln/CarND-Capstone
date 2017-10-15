#from styx_msgs.msg import TrafficLight
import numpy as np
import tensorflow as tf

real_model = 'frozen_models/real_data/frozen_inference_graph.pb'
sim_model = 'frozen_models/sim_data/frozen_inference_graph.pb'


class TLClassifier(object):
    def __init__(self):
        graph = tf.Graph()
        od_graph_def = tf.GraphDef()

        with tf.gfile.GFile(sim_model, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            graph = tf.import_graph_def(od_graph_def, name='')

        self.sess = tf.Session(graph=graph)

    def convert_traffic_color(self, id):
        if id == 1:
            return TrafficLight.GREEN
        if id == 2:
            return TrafficLight.RED
        if id == 3:
            return TrafficLight.YELLOW

        return TrafficLight.UNKNOWN

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        image_tensor = self.sess.graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = self.sess.graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = self.sess.graph.get_tensor_by_name('detection_scores:0')
        detection_classes = self.sess.graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.sess.graph.get_tensor_by_name('num_detections:0')

        image = np.expand_dims(image, axis=0)
        boxes, scores, classes, num = self.sess.run([detection_boxes, detection_scores,
                                                detection_classes, num_detections],
                                               feed_dict={image_tensor: image})
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)

        index = np.argmax(scores)
        if scores[index] > 0.5:
            return classes[index]

        return -1
        return TrafficLight.UNKNOWN
