#from styx_msgs.msg import TrafficLight
from keras.models import model_from_json
import cv2
import numpy as np

class TLClassifier(object):
    def __init__(self):
        self.model = self.load_model('model.json')

    def load_model(self, path):
        with open(path, 'r') as jfile:
            model_json = jfile.read()
            model = model_from_json(model_json)
        model.compile("adam", "categorical_crossentropy")
        weights_file = path.replace('json', 'h5')
        model.load_weights(weights_file)
        return model

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
        transformed_image_array = image[None, :, :, :]
        light = self.model.predict(transformed_image_array, batch_size=1).squeeze()

        return np.argmax(light)
        #if light == 1:
        #    return TrafficLight.RED

        #return TrafficLight.UNKNOWN
