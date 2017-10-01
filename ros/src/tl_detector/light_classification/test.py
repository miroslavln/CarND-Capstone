from tl_classifier import TLClassifier
import cv2

classfier = TLClassifier()
test_image_red = cv2.imread('../images/red/img0.jpg')
print(classfier.get_classification(test_image_red))

test_image_green = cv2.imread('../images/green/img281.jpg')
print(classfier.get_classification(test_image_green))
