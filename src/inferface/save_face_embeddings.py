import os
import csv

from PIL import Image
from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1
from facenet_pytorch.models.mtcnn import MTCNN
from torchvision.transforms import transforms

image_ext = [".jpeg", ".jpg"]


def create_face_embeddings(image_dir, crops_dir=None):
    # If required, create a face detection pipeline using MTCNN:
    mtcnn = MTCNN()

    # Create an inception resnet (in eval mode):
    resnet = InceptionResnetV1(pretrained='vggface2').eval()

    img_paths_and_embeddings = list()
    img_paths_and_embeddings.append(['Image Path', 'Embedding'])

    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith(tuple(image_ext)):
                img_path = os.path.join(root, file)
                print(img_path)

                img = Image.open(img_path)

                pil_to_tensor = transforms.ToTensor()(img)
                pil_to_tensor = transforms.Resize((160, 160))(pil_to_tensor).unsqueeze(0)

                # Calculate embedding (unsqueeze to add batch dimension)
                img_embedding = resnet(pil_to_tensor)

                img_paths_and_embeddings.append([img_path, img_embedding.detach().cpu().numpy()])

    return img_paths_and_embeddings


def write_face_embeddings_to_csv(img_paths_and_embeddings, embedding_csv):
    with open(embedding_csv, 'w', newline='') as csv_file:
        embedding_writer = csv.writer(csv_file, delimiter=',', quoting=csv.QUOTE_ALL)
        embedding_writer.writerows(img_paths_and_embeddings)
