# Pizza counting project

Stage 1: Data preparation as yolo format:

- Optional: Snapshot video then labeling on roboflow.
- Collect more annotated data on roboflow, pick proper data, e.g. data with model (so we can test our data first), with tags: pizza, object detection, images>2000.
- Combine all or mixed some to have various models.

Stage 2: Train and fine-tuning:

- Pick pretrained model yolo, the larger the better result.
- Train our datasets to have more models to test.
- Increase epochs, imgsz, batch, etc.
- Run models to pick the best.

## Challenges

Due to hardware limitation (I'm running RTX 3060, 32GB RAM), it takes long and sometime face OutofMemoryError.
To improve, with good conditions, I'd try fine-tuning params or run larger pretrained model. However, I think the most importance is data, the corner of the cam get the data, some is so bright, or hard to see, sometime is covered by obstacles.

## Installation

```bash
git clone https://github.com/long3888/pizzaCounter.git
pip install requirements.txt
```

## Usage

```bash
cd testing
# Place the video test here, e.g. 1462_CH04_20250607210159_211703.mp4
python app.py
```

In app.py:

```python
# Video path
cap = cv2.VideoCapture("1462_CH04_20250607210159_211703.mp4") # video_source=0 to use a webcam

# Whenever change the video path, find the region on https://polygonzone.roboflow.com/
region_points = [[262, 1069], [1009, 579], [1386, 730], [1233, 1067]]    # rectangle region
```
