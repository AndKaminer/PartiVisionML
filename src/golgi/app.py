import os
import base64
import json
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # For headless/server environments
import matplotlib.pyplot as plt
import requests
import tempfile
import time

from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from dash_canvas import DashCanvas
from dash.exceptions import PreventUpdate
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import roboflow
import cv2

from golgi import settings
from golgi.inference import InferencePipeline
from golgi.annotation import AnnotatedImage

#############################################
# 1) Check / Download Model from Hugging Face
#############################################

MODELS_FOLDER = os.path.join(os.getcwd(), "models")
LOCAL_MODEL_PATH = os.path.join(MODELS_FOLDER, settings.soft_get_setting("model_name"))
INFERENCE_PIPELINE = None
CANVAS_WIDTH = 400
CANVAS_HEIGHT = 400

def ensure_model_exists():
    """
    Use local `models/sep13.pt` if present. Otherwise, download from Hugging Face.
    """
    if os.path.isfile(LOCAL_MODEL_PATH):
        print(f"Using existing local model at {LOCAL_MODEL_PATH}.")
        return LOCAL_MODEL_PATH
    else:
        print(f"Local model not found at {LOCAL_MODEL_PATH}. Downloading from HuggingFace...")
        os.makedirs(MODELS_FOLDER, exist_ok=True)
        downloaded_file = hf_hub_download(
repo_id=settings.soft_get_setting("huggingface_repo_id"),
            filename=settings.soft_get_setting("model_name"),
token=settings.soft_get_setting("huggingface_token"),
            local_dir=MODELS_FOLDER
        )
        # Move downloaded file to "models/sep13.pt"
        print(f"Model downloaded and stored at {LOCAL_MODEL_PATH}.")
        return LOCAL_MODEL_PATH

MODEL_PATH = ensure_model_exists()


#############################################
# 2) Background Subtraction & 6-plot Tracking
#############################################
def background_subtraction_with_6plots(
    input_video,
    output_video,
    plot_output_video,
    model_path=None,    # currently not used for real inference, but loaded if desired
    scaling_factor_brightness=1.0,
    denoise=True,
    frame_rate_override=None
):
    """
    Median-based background subtraction + largest contour detection -> area/perimeter/etc.
    Writes 2 AVIs if requested, returns list of dict for CSV.
    """

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {input_video}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
    cap.release()
    if len(frames) == 0:
        raise ValueError("No frames found in the input video.")

    arr = np.array(frames)
    bg = np.median(arr, axis=0).astype(np.uint8)
    bg = cv2.medianBlur(bg, 5)

    gmin, gmax = float('inf'), float('-inf')
    float_bg = bg.astype(np.float32)
    for f in arr:
        diff = f.astype(np.float32) - float_bg
        mn, mx = diff.min(), diff.max()
        gmin = min(gmin, mn)
        gmax = max(gmax, mx)

    h, w = arr[0].shape
    cap2 = cv2.VideoCapture(input_video)
    fps = cap2.get(cv2.CAP_PROP_FPS)
    cap2.release()
    if fps <= 0:
        fps = 25
    if frame_rate_override and frame_rate_override > 0:
        fps = frame_rate_override

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_vid = None
    out_plt = None
    if output_video:
        out_vid = cv2.VideoWriter(output_video, fourcc, fps, (w, h), True)
    if plot_output_video:
        out_plt = cv2.VideoWriter(plot_output_video, fourcc, fps, (1600, 400), True)

    kernel = np.ones((3,3), np.uint8)

    # Lists for time-series data
    area_vals, perimeter_vals, height_vals = [], [], []
    velocity_vals, acceleration_vals, circularity_vals = [], [], []
    y_position_vals, time_vals = [], []

    prev_cx, prev_cy, prev_v = None, None, 0

    plt.ioff()
    fig, axs = plt.subplots(1, 6, figsize=(16, 3), dpi=100, facecolor='black')
    for ax in axs:
        ax.set_facecolor("black")

    def fig_to_bgr():
        fig.canvas.draw()
        ww, hh = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(hh, ww, 4)
        return cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)

    time_per_frame = 1.0 / fps

    for i, f in enumerate(arr):
        diff = f.astype(np.float32) - float_bg
        if gmax == gmin:
            norm = np.zeros_like(diff, dtype=np.uint8)
        else:
            norm = (diff - gmin)/(gmax - gmin)*255.0*scaling_factor_brightness
        norm = np.clip(norm, 0, 255).astype(np.uint8)

        if denoise:
            norm = cv2.morphologyEx(norm, cv2.MORPH_OPEN, kernel)
            norm = cv2.morphologyEx(norm, cv2.MORPH_CLOSE, kernel)

        # threshold + largest contour
        _, th = cv2.threshold(norm, 20, 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area, perimeter, bh, v, a, circ, cy_val = 0, 0, 0, 0, 0, 0, 0

        if cnts:
            c = max(cnts, key=cv2.contourArea)
            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)
            x, y, bw, bh = cv2.boundingRect(c)
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
            else:
                cx, cy = 0, 0

            if prev_cx is not None:
                v = ((cx - prev_cx)**2 + (cy - prev_cy)**2)**0.5
                a = v - prev_v
            prev_cx, prev_cy, prev_v = cx, cy, v
            cy_val = cy
            if perimeter > 0:
                circ = 4.0 * np.pi * area / (perimeter * perimeter)

        area_vals.append(area)
        perimeter_vals.append(perimeter)
        height_vals.append(bh)
        velocity_vals.append(v)
        acceleration_vals.append(a)
        circularity_vals.append(circ)
        y_position_vals.append(cy_val)
        time_vals.append(i*time_per_frame)

        if out_vid is not None:
            bgr_frame = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
            out_vid.write(bgr_frame)

        if out_plt is not None:
            x_vals = range(len(area_vals))
            for ax in axs:
                ax.clear()
                ax.set_facecolor("black")
                ax.tick_params(axis='x', colors='white')
                ax.tick_params(axis='y', colors='white')
                for spine in ax.spines.values():
                    spine.set_color('white')

            axs[0].plot(x_vals, area_vals, color='white')
            axs[0].set_title("Area", color='white')
            axs[1].plot(x_vals, perimeter_vals, color='white')
            axs[1].set_title("Perimeter", color='white')
            axs[2].plot(x_vals, height_vals, color='white')
            axs[2].set_title("Height", color='white')
            axs[3].plot(x_vals, velocity_vals, color='white')
            axs[3].set_title("Velocity", color='white')
            axs[4].plot(x_vals, acceleration_vals, color='white')
            axs[4].set_title("Acceleration", color='white')
            axs[5].plot(x_vals, circularity_vals, color='white')
            axs[5].set_title("Circularity", color='white')

            plot_bgr = fig_to_bgr()
            plot_bgr = cv2.resize(plot_bgr, (1600, 400))
            out_plt.write(plot_bgr)

    if out_vid is not None:
        out_vid.release()
    if out_plt is not None:
        out_plt.release()
    plt.close(fig)

    results = []
    for i in range(len(time_vals)):
        row = {
            "time": time_vals[i],
            "area": area_vals[i],
            "perimeter": perimeter_vals[i],
            "height": height_vals[i],
            "velocity": velocity_vals[i],
            "acceleration": acceleration_vals[i],
            "circularity": circularity_vals[i],
            "y_position": y_position_vals[i],
        }
        results.append(row)
    return results


def track_video(video_path, model, framerate, window_width, scaling_factor, um_per_pixel, output_folder, avi, csv):
    global INFERENCE_PIPELINE

    ip = InferencePipeline(
            model=model,
            framerate=framerate,
            window_width=window_width,
            scaling_factor=scaling_factor,
            um_per_pixel=um_per_pixel,
            output_folder=output_folder)
        
    INFERENCE_PIPELINE = ip

    ip.process_video(video_path, avi=avi, csv=csv)

    INFERENCE_PIPELINE = None


def run_tracking_on_folder(folder_path, output_types, frame_rate):
    """
    For each .avi/.mp4 in folder, run background subtraction + tracking
    and optionally output CSV, AVI videos.
    """

    processed_files = []
    if not os.path.isdir(folder_path):
        return processed_files

    output_folder = os.path.join(folder_path, "output")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    model = YOLO(MODEL_PATH)

    avi = "AVI"in output_types
    csv = "CSV" in output_types

    for file_name in os.listdir(folder_path):
        ext = file_name.lower()
        if not(ext.endswith('.avi') or ext.endswith('.mp4')):
            continue
        input_path = os.path.join(folder_path, file_name)

        track_video(input_path,
                    model,
                    frame_rate,
                    settings.soft_get_setting("window_width"),
                    settings.soft_get_setting("scaling_factor"),
                    settings.soft_get_setting("um_per_pixel"),
                    output_folder,
                    avi,
                    csv)
        
        processed_files.append(file_name)
    return processed_files


#############################################
# 3) Roboflow Annotation Upload
#############################################
def upload_annotation_to_roboflow(api_key, workspace, project, image_bgr, shapes, frame_index, window_width=150):
    """
    Upload bounding boxes for the frame (converted to .jpg in memory) to 
    the Roboflow dataset "sep13" with your API key.
    """

    contours = []
    scale = 1
    for s in shapes:
        print(s)
        if s.get("type") == "path":
            contours.append(dash_canvas_to_opencv(s, scale))
        elif s.get("type") == "image": # image always comes first
            scale = s["scaleX"]

    mask = np.zeros(image_bgr.shape, np.uint8)
    cv2.drawContours(mask, contours, -1, (255), -1)

    left_bound = float("inf")
    right_bound = float("-inf")

    for ctr in contours:
        x, _, w, _ = cv2.boundingRect(ctr)
        left_bound = min(left_bound, x)
        right_bound = max(right_bound, x + w)

    left_bound = max(0, left_bound - window_width // 2)
    right_bound = min(image_bgr.shape[1], right_bound + window_width // 2)

    image_bgr = image_bgr[:, left_bound:right_bound + 1]
    mask = mask[:, left_bound:right_bound + 1]


    rf = roboflow.Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project)

    image_path, annotation_path, temp_dir = temp_construct_roboflow_annotation(image_bgr, mask)

    project.single_upload(image_path=image_path,
                                annotation_path=annotation_path,
                                batch_name="batch")


def temp_construct_roboflow_annotation(image, mask):
    temp_dir = tempfile.TemporaryDirectory()
    img_path = os.path.join(temp_dir.name, "defaultfilename.png")
    annotation_path = img_path + "-annotation.coco.json"

    current_time = time.localtime()

    resize_constant = 3
    
    # info
    year = current_time.tm_year
    version = "1.0"
    description = "not found"
    contributor = "not found"
    url = "not found"
    date_created = f"{current_time.tm_mday}-{current_time.tm_mon}-{current_time.tm_year}"

    info = {
        "year": year,
        "version": version,
        "description": description,
        "contributor": contributor,
        "url": url,
        "date_created": date_created
    }

    licenses = [{
            "id": 1,
            "url": "not found",
            "name": "not found"
            }]

    categories = [{
        "id": 0,
        "name": "Cell",
        "supercategory": "none"
        }]

    images = [{
        "id": 0,
        "license": 1,
        "file_name": img_path,
        "height": image.shape[0],
        "width": image.shape[1],
        "date_captured": date_created
        }]

    annotations = []

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    left_bound = float("inf")
    right_bound = float("-inf")

    for id, ctr in enumerate(contours):
        annotation = {}
        contour = ctr.flatten().tolist()
        contour_pairs = [(contour[i], contour[i+1]) for i in range(0, len(contour), 2)]
        segmentation = [int(coord) for pair in contour_pairs for coord in pair]

        area = cv2.contourArea(ctr)
        bbox = [ int(x) for x in cv2.boundingRect(ctr) ]
        left_bound = min(left_bound, bbox[0])
        right_bound = max(right_bound, bbox[0] + bbox[2])

        annotation["segmentation"] = segmentation
        annotation["area"] = area
        annotation["bbox"] = bbox
        annotation["image_id"] = 0
        annotation["category_id"] = 0
        annotation["id"] = id
        annotation["iscrowd"] = 0

        annotations.append(annotation)


    coco_json = {
            "info" : info,
            "licenses" : licenses,
            "categories" : categories,
            "images" : images,
            "annotations" : annotations
            }

    cv2.imwrite(img_path, image)

    with open(annotation_path, "w") as f:
        f.write(json.dumps(coco_json))

    return img_path, annotation_path, temp_dir
    


def dash_canvas_to_opencv(path_object, scale):
    path = path_object.get("path", [])
    points = []

    for curve in path:
        if curve[0] == "M" or curve[0] == "L":
            points.append([curve[1] / scale, curve[2] / scale])
        elif curve[0] == "Q":
            points.append([curve[1] / scale, curve[2] / scale])
            points.append([curve[3] / scale, curve[4] / scale])
        else:
            return []

    ctr = np.array(points).reshape((-1, 1, 2)).astype(np.int32)

    return ctr




#############################################
# 4) Dash App
#############################################
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], update_title=None)

app.title = "Cell Tracking Dashboard"

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H2("Cell Tracking Interface", className="text-primary mt-3"),
            html.Hr()
        ], width=12)
    ]),

    # Huggingface section
    dbc.Row([
        dbc.Col([
            html.H5("Huggingface API Details", className="mt-4"),
            dbc.Input(id="huggingface-rep-id", placeholder="Enter Huggingface Rep ID", type="text", className="mb-3",
                      value=settings.soft_get_setting("huggingface_repo_id")),
            dbc.Input(id="huggingface-token", placeholder="Enter Huggingface Token", type="text", className="mb-3",
                      value=settings.soft_get_setting("huggingface_token")),
            dbc.Button("Submit Huggingface", id="huggingface-submit", color="primary")
        ], width=6),
        
        # Roboflow section
        dbc.Col([
            html.H5("Roboflow API Details", className="mt-4"),
            dbc.Input(id="roboflow-model-name", placeholder="Enter Roboflow Model Name", type="text", className="mb-3"),
            dbc.Input(id="roboflow-api-key", placeholder="Enter Roboflow API Key", type="text", className="mb-3",
                      value=settings.soft_get_setting("roboflow_api_key")),
            dbc.Input(id="roboflow-workspace", placeholder="Enter Roboflow Workspace", type="text", className="mb-3",
                      value=settings.soft_get_setting("roboflow_workspace_name")),
            dbc.Input(id="roboflow-project-id", placeholder="Enter Roboflow Project ID", type="text", className="mb-3",
                      value=settings.soft_get_setting("roboflow_project_name")),
            dbc.Input(id="roboflow-version", placeholder="Enter Roboflow Version", type="text", className="mb-3",
                      value=settings.soft_get_setting("roboflow_version_number")),
            dbc.Button("Submit Roboflow", id="roboflow-submit", color="primary")
        ], width=6),
    ]),

    # Message after submission
    dbc.Row([
        dbc.Col([
            html.Div(id="submit-message", className="mt-3", style={"color": "green", "fontWeight": "bold"})
        ], width=12)
    ]),


    # TRAINING SECTION
    dbc.Card([
        dbc.CardHeader(html.H4("Training", className="text-white bg-primary mb-0")),
        dbc.CardBody([
            html.Div([
                html.Label("Upload Training Video", className="fw-bold"),
                dcc.Upload(
                    id='upload-training-video',
                    children=html.Div([
                        'Drag and Drop or ',
                        html.A('Select a Video File')
                    ]),
                    style={
                        'width': '100%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '10px'
                    },
                    multiple=False
                ),
                html.Div(id="train-video-status", className="text-info mb-3"),
            ]),

            dbc.Row([
                dbc.Col([
                    html.Button("Auto-Detect Particles", id="btn-auto-detect", n_clicks=0, className="btn btn-success"),
                    html.Div(id="autodetect-status", className="text-success mt-2")
                ], width=4),
            ], className="mb-3"),

            dbc.Row([
                dbc.Col([
                    html.Label("Frame Index:", className="fw-bold"),
                    dcc.Input(id="frame-index-input", type="number", value=0, min=0, className="form-control d-inline-block me-2", style={"width":"80px"}),
                    html.Button("Go To Frame", id="btn-goto-frame", n_clicks=0, className="btn btn-secondary"),
                    html.Div(id="frame-count-display", className="text-info mt-2"),
                ], width=4),
                dbc.Col([
                    html.Div(id="video-frame-display", style={"textAlign":"center"}),
                ], width=8),
            ], className="mb-3"),

            html.Label("Annotation Tool:", className="fw-bold"),
            html.Div([
                DashCanvas(
                    id='canvas-annotation',
                    width=CANVAS_WIDTH,
                    height=CANVAS_HEIGHT,
                    lineWidth=2,
                    goButtonTitle='Done',
                    tool='pencil'
                ),
            ], className="border border-secondary mb-2", style={"display":"inline-block"}),

            html.Button("Save Annotation", id="btn-save-annotation", n_clicks=0, className="btn btn-warning"),
            html.Div(id="save-annotation-status", className="text-success mt-2"),
        ])
    ], className="my-3"),

    # TRACKING SECTION
    dbc.Card([
        dbc.CardHeader(html.H4("Tracking", className="text-white bg-primary mb-0")),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Folder Path for Videos", className="fw-bold"),
                    dcc.Input(id="tracking-folder-path", type="text", placeholder=r"C:\my_videos", className="form-control",
                              value=settings.soft_get_setting("video_folder"))
                ], width=6),
                dbc.Col([
                    html.Label("Frame Rate Override (optional)", className="fw-bold"),
                    dcc.Input(id="tracking-frame-rate", type="number", placeholder="25", className="form-control",
                              value=settings.soft_get_setting("framerate") )
                ], width=3),
            ], className="mb-3"),

            dbc.Row([
                dbc.Col([
                    html.Label("Export Options:", className="fw-bold"),
                    dbc.Checklist(
                        options=[
                            {"label": "CSV", "value": "CSV"},
                            {"label": "AVI", "value": "AVI"},
                        ],
                        value=["CSV", "AVI"],
                        id="export-options",
                        inline=True,
                        className="mb-2"
                    )
                ], width=3),
                dbc.Col([
                    html.Button("Run Tracking", id="btn-run-tracking", n_clicks=0, className="btn btn-danger")
                ], width=2),
                dbc.Col([
                    dbc.Progress(id="tracking-progress", value=0, striped=True, animated=True, style={"height":"30px"}),
                    dcc.Interval(id="progress-interval", n_intervals=0, interval=500)
                ], width=7),
            ]),

            html.Div(id="tracking-status", className="text-info mt-2"),
        ])
    ], className="my-3"),

    dcc.Store(id='training-frames', data=[]),
    dcc.Store(id='autodetected-bboxes', data={}),
], fluid=True)


#############################################
# 5) Callbacks
#############################################

# -- TRAINING VIDEO UPLOAD --
@app.callback(
    Output("training-frames", "data"),
    Output("train-video-status", "children"),
    Output("canvas-annotation", "width"),
    Output("canvas-annotation", "height"),
    Input("upload-training-video", "contents"),
    prevent_initial_call=True
)
def on_training_video_upload(contents):
    if not contents:
        raise PreventUpdate
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    temp_filename = 'temp_training_video.avi'
    with open(temp_filename, 'wb') as f:
        f.write(decoded)

    cap = cv2.VideoCapture(temp_filename)
    frames_data = []
    success = True
    width = height = 0
    while success:
        success, frame = cap.read()
        if not success:
            break
        height, width = frame.shape[0], frame.shape[1]
        # Convert to jpg base64
        _, buffer = cv2.imencode('.jpg', frame)
        
        b64_frame = base64.b64encode(buffer).decode('utf-8')
        frames_data.append(b64_frame)

    cap.release()
    os.remove(temp_filename)
    status = f"Uploaded video with {len(frames_data)} frames."
    return frames_data, status, width, height


# -- AUTO-DETECT PARTICLES (naive) --
@app.callback(
    Output("autodetected-bboxes", "data"),
    Output("autodetect-status", "children"),
    Input("btn-auto-detect", "n_clicks"),
    State("training-frames", "data"),
    prevent_initial_call=True
)
def auto_detect_particles(n_clicks, frames):
    if not frames:
        raise PreventUpdate

    # Quick approach: do median background of all frames, get largest contour per frame
    autodetected = {}
    gray_frames = []
    for f_b64 in frames:
        dec = base64.b64decode(f_b64)
        arr = np.frombuffer(dec, np.uint8)
        im = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        gray_frames.append(gray)

    arr = np.array(gray_frames)
    bg = np.median(arr, axis=0).astype(np.uint8)
    bg = cv2.medianBlur(bg, 5)
    float_bg = bg.astype(np.float32)

    gmin, gmax = float('inf'), float('-inf')
    for g in arr:
        df = g.astype(np.float32) - float_bg
        mn, mx = df.min(), df.max()
        gmin = min(gmin, mn)
        gmax = max(gmax, mx)

    kernel = np.ones((3,3), np.uint8)

    for i, gf in enumerate(arr):
        df = gf.astype(np.float32) - float_bg
        if gmax != gmin:
            norm = (df-gmin)/(gmax-gmin)*255
        else:
            norm = np.zeros_like(df)
        norm = np.clip(norm, 0, 255).astype(np.uint8)
        norm = cv2.morphologyEx(norm, cv2.MORPH_OPEN, kernel)
        norm = cv2.morphologyEx(norm, cv2.MORPH_CLOSE, kernel)
        _, th = cv2.threshold(norm, 20, 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            autodetected[i] = [(x, y, w, h)]
        else:
            autodetected[i] = []

    msg = f"Auto-detected bounding boxes for {len(frames)} frames."
    return autodetected, msg


# -- GO TO FRAME & DISPLAY --
@app.callback(
    Output("frame-count-display", "children"),
    Output("video-frame-display", "children"),
    Output("canvas-annotation", "image_content"),
    Output("canvas-annotation", "json_data"),
    Input("btn-goto-frame", "n_clicks"),
    State("frame-index-input", "value"),
    State("training-frames", "data"),
    prevent_initial_call=True
)
def goto_frame(n_clicks, frame_idx, frames):
    if not frames or frame_idx < 0 or frame_idx >= len(frames):
        raise PreventUpdate

    count_text = f"Frame {frame_idx} of {len(frames)-1}"
    frame_b64 = frames[frame_idx]

    # DashCanvas expects "image_content" as a data URI for annotations
    canvas_content = "data:image/jpg;base64," + frame_b64

    # For "json_data", dash-canvas expects a string (JSON). We'll pass an empty "objects" list
    return (
        count_text,
        # Remove the large image display here
        None,
        canvas_content,
        json.dumps({"objects": []})  # Empty objects for annotation tool
    )


# -- SAVE ANNOTATION to Roboflow --
@app.callback(
    Output("save-annotation-status", "children"),
    Input("btn-save-annotation", "n_clicks"),
    State("roboflow-api-key", "value"),
    State("roboflow-workspace", "value"),
    State("roboflow-project-id", "value"),
    State("frame-index-input", "value"),
    State("training-frames", "data"),
    State("canvas-annotation", "json_data"),
    prevent_initial_call=True
)
def save_annotation(n_clicks, api_key, workspace, project, frame_idx, frames, annotation_str):
    if not frames or frame_idx<0 or frame_idx>=len(frames):
        raise PreventUpdate
    frame_b64 = frames[frame_idx]
    dec = base64.b64decode(frame_b64)
    arr = np.frombuffer(dec, np.uint8)
    frame_bgr = cv2.imdecode(arr, 0)

    # annotation_str is a JSON string
    try:
        ann_data = json.loads(annotation_str)
    except:
        ann_data = {"objects":[]}

    shapes = ann_data.get("objects", [])
    upload_annotation_to_roboflow(api_key, workspace, project, frame_bgr, shapes, frame_idx, settings.soft_get_setting("window_width"))
    return f"Annotation for frame {frame_idx} uploaded to Roboflow successfully."


# -- RUN TRACKING ON A FOLDER --
@app.callback(
    Output("tracking-progress", "value", allow_duplicate=True),
    Output("tracking-progress", "label", allow_duplicate=True),
    Output("tracking-status", "children"),
    Input("btn-run-tracking", "n_clicks"),
    State("tracking-folder-path", "value"),
    State("export-options", "value"),
    State("tracking-frame-rate", "value"),
    prevent_initial_call=True
)
def run_tracking(n_clicks, folder_path, export_values, frame_rate):
    if not folder_path:
        raise PreventUpdate

    processed = run_tracking_on_folder(folder_path, export_values, frame_rate)
    if not processed:
        return (0, "0%", "No videos processed. Check folder path or no .avi/.mp4 found.")

    return (100, "100%", f"Processing complete: {processed}")


# -- UPDATE PROGRESS BAR --
@app.callback(
    Output("tracking-progress", "value"),
    Output("tracking-progress", "label"),
    Input("progress-interval", "n_intervals"),
    Input("tracking-progress", "value"),
    Input("tracking-progress", "label"))
def update_progress(n, value, label):
    if INFERENCE_PIPELINE == None:
        return value, label

    progress = int(INFERENCE_PIPELINE.progress * 100)

    return progress, f"{progress}%" if progress >= 5 else ""



def main():
    app.run_server(debug=True)

if __name__ == "__main__":
    main()
