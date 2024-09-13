import torch
from facenet_pytorch import MTCNN
import cv2
from PIL import Image
import matplotlib.pyplot as plt

def main():
    # Check if a GPU is available and if not, use a CPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Print the device being used
    if device.type == 'cuda':
        print("Using GPU")
    else:
        print("Using CPU")

    # Create an instance of the MTCNN face detector
    mtcnn = MTCNN(keep_all=True, device=device)

    # Load an image using OpenCV
    # Replace './input/faces.png' with the path to your image file
    image_path = './input/faces.png'
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert the image to PIL format
    pil_image = Image.fromarray(image)

    # Use MTCNN to detect faces in the image
    boxes, _ = mtcnn.detect(pil_image)

    # Draw bounding boxes and label each face
    fig, ax = plt.subplots()
    ax.imshow(image)

    if boxes is not None:
        for i, box in enumerate(boxes):
            ax.text(box[0], box[1], str(i+1), fontsize=12, color='cyan')
            rect = plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, color='magenta', linewidth=2)
            ax.add_patch(rect)

        # Print the number of faces detected
        print(f"Number of faces detected: {len(boxes)}")
    else:
        print("No faces detected.")

    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
