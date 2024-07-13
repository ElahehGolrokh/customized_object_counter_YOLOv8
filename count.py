# Source code: https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/object_counting.ipynb

import argparse
import os
import cv2
from ultralytics import YOLO
from src.custom_counter import ObjectCounter


parser = argparse.ArgumentParser(description='Generate data and model paths')

# Defining the parser arguments
parser.add_argument('-o',
                    '--outputs',
                    default='outputs',
                    help='path to output dir')
parser.add_argument('-i',
                    '--inputs',
                    default='inputs',
                    help='path to input dir')
parser.add_argument('-m',
                    '--models',
                    default='models',
                    help='path to saved models dir')
parser.add_argument('-on',
                    '--output_name',
                    default='output.avi',
                    help='name of the output file for saving with .avi ext')
parser.add_argument('-in',
                    '--input_name',
                    default=None,
                    help='name of the input video file')
parser.add_argument('-mn',
                    '--model_name',
                    default='yolov8n.pt',
                    help='name of the object detection saved model')

args = parser.parse_args()

# Extract tha arguments values
output_dir = args.outputs
input_dir = args.inputs
model_dir = args.models
output_name = args.output_name
input_name = args.input_name
model_name = args.model_name


def main(input_dir: str = input_dir,
         output_dir: str = output_dir,
         model_dir: str = model_dir,
         input_name: str = input_name,
         output_name: str = output_name,
         model_name: str = model_name):

    try:
        model_path = os.path.join(model_dir, model_name)
        input_path = os.path.join(input_dir, input_name)
        output_path = os.path.join(output_dir, output_name)

        # Load the pre-trained YOLOv8 model
        model = YOLO(model_path)

        # Open the video file
        cap = cv2.VideoCapture(input_path)
    except Exception:
        print("Error reading video file or model")

    else:
        # Get video properties: width, height, and frames per second (fps)
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH,
                                               cv2.CAP_PROP_FRAME_HEIGHT,
                                               cv2.CAP_PROP_FPS))

        # Define points for a line or region of interest in the video frame
        line_points = [(20, 200), (w, 200)]  # Predefined Line coordinates

        # Specify classes to count, for example: person (0) and car (2)
        classes_to_count = [0, 2]  # Class IDs for person and car

        # Initialize the video writer to save the output video
        video_writer = cv2.VideoWriter(output_path,
                                       cv2.VideoWriter_fourcc(*"mp4v"),
                                       fps, (w, h))

        # Initialize the Object Counter
        counter = ObjectCounter(
            view_img=True,  # Display the image during processing
            reg_pts=line_points,  # Region of interest points
            classes_names=model.names,  # Class names from the YOLO model
            draw_tracks=True,  # Draw tracking lines for objects
            line_thickness=2,  # Thickness of the lines drawn
        )

        # Process video frames in a loop
        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                print("Video frame is empty or video processing has been successfully completed.")
                break

            # Perform object tracking on the current frame
            tracks = model.track(im0,
                                 persist=True,
                                 show=False,
                                 classes=classes_to_count)

            # Use the Object Counter to count objects in the frame and get the annotated image
            im0 = counter.start_counting(im0, tracks)
            # Write the annotated frame to the output video
            video_writer.write(im0)

        # Release the video capture and writer objects
        cap.release()
        video_writer.release()

        # Close all OpenCV windows
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
