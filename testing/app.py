import cv2
from ultralytics import solutions

cap = cv2.VideoCapture("1462_CH04_20250607210159_211703.mp4") # video_source=0 to use a webcam
assert cap.isOpened(), "Error reading video file"

region_points = [[262, 1069], [1009, 579], [1386, 730], [1233, 1067]]    # rectangle region

# Video writer
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("object_counting_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Desired display resolution
display_width = 1280  # Example width
display_height = 720 # Example height

# Initialize object counter object
counter = solutions.ObjectCounter(
    show=False,  # Set to False to handle display manually
    region=region_points,  # pass region points
    model="best.pt",  # model="yolo11n-obb.pt" for object counting with OBB model.
    classes=[0],  # count specific classes i.e. person and car with COCO pretrained model.
    tracker="bytetrack.yaml",  # choose trackers i.e "bytetrack.yaml"
)

# Process video
while cap.isOpened():
    success, im0 = cap.read()

    if not success:
        print("Video frame is empty or processing is complete.")
        break

    results = counter(im0)

    # Get the annotated frame from results
    annotated_frame = results.plot_im

    # Write the original resolution annotated frame to the output file
    video_writer.write(annotated_frame)

    # Resize the annotated frame for display only
    display_frame = cv2.resize(annotated_frame, (display_width, display_height))

    # Display the resized frame
    cv2.imshow("Object Counting Display", display_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()