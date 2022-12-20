from torchvision import transforms
import cv2
import numpy as np
from facenet_pytorch import MTCNN


class LandmarksTransform(object):
    def __call__(self, path):
        mtcnn = MTCNN(image_size=224, post_process=False, margin=50)
        frame = cv2.imread(path)

        frame_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Resize([224, 224]),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        assert frame is not None
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, probs = mtcnn.detect(frame)
        face = mtcnn.extract(frame, boxes, save_path=None)
        face = face.permute(1, 2, 0).numpy()
        face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
        face = face.astype(np.uint8)
        frame = frame_transform(face)
        assert frame.shape == (3, 224, 224)

        return frame.float()

