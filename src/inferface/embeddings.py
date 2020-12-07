import logging
import os
from tqdm import tqdm
from PIL import Image
from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1
from facenet_pytorch.models.mtcnn import MTCNN
from torchvision.transforms import transforms

from inferface.utils import walk_dir_in_batches

image_ext = [".jpeg", ".jpg"]
_logger = logging.getLogger(__name__)


def create_face_embeddings(image_dir):
    _logger.info(f"Starting creation of face embeddings in directory {image_dir}")

    # If required, create a face detection pipeline using MTCNN:
    mtcnn = MTCNN()

    # Create an inception resnet (in eval mode):
    resnet = InceptionResnetV1(pretrained='vggface2').eval()

    img_paths_and_embeddings = list()
    img_paths_and_embeddings.append(['Image Path', 'Embedding'])
    no_faces_found = list()
    no_faces_found.append(['Image Path'])

    pbar = tqdm(total=len(os.listdir(image_dir)))
    batch_size = 128
    for file_name_batch in walk_dir_in_batches(image_dir, batch_size=batch_size):
        for file_name in file_name_batch:
            # Calculate embedding
            if file_name.endswith(tuple(image_ext)):
                img_path = os.path.join(image_dir, file_name)
                _logger.debug(img_path)

                img_embedding = get_embedding(img_path, mtcnn, resnet)
                if img_embedding is None:
                    no_faces_found.append(img_path)
                else:
                    img_paths_and_embeddings.append([img_path, img_embedding.detach().cpu().numpy()])
                pbar.update(1)
    pbar.close()
    _logger.info(f"Starting creation of face embeddings in directory {image_dir}")
    return img_paths_and_embeddings, no_faces_found


def get_embedding(img_path, mtcnn, resnet):
    img = Image.open(img_path)
    # Get cropped and prewhitened image tensor
    boxes, _ = mtcnn.detect(img)
    if boxes is not None:  # if no bounding box of a face can be found
        img_cropped = mtcnn(img)
        # pil_to_tensor = transforms.ToTensor()(img_cropped)
        pil_to_tensor = transforms.Resize((160, 160))(img_cropped).unsqueeze(0)
        # Calculate embedding (unsqueeze to add batch dimension)
        img_embedding = resnet(pil_to_tensor)
        return img_embedding
    return None

