import os
import base64
import json
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import tempfile
import time
import csv
import re
import datetime
import requests

from dash import Dash, dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
from dash import dcc
from dash.exceptions import PreventUpdate
import huggingface_hub
from ultralytics import YOLO
import roboflow
import cv2
import plotly.graph_objects as go
from tqdm import tqdm

from partivision import settings
from partivision.inference import InferencePipeline
from partivision.annotation import AnnotatedImage
from partivision.date_utils import parse_model_date, hasdate

#############################################
# 1) Check / Download Model from Hugging Face
#############################################


class BasicCallback():
    def __init__(self):
        self.value = 0


MODELS_FOLDER = os.path.join(os.getcwd(), "models")
LOCAL_MODEL_PATH = os.path.join(MODELS_FOLDER, settings.soft_get_setting("model_name", "NOTFOUND"))
INFERENCE_PIPELINE = None
CANVAS_WIDTH = 400
CANVAS_HEIGHT = 400
MODEL_FOUND = False
MODEL_PATH = None
MODEL_DOWNLOAD_PROGRESS = BasicCallback()


def get_most_recent_model(repo_id, token):
    
    files = [ f for f in huggingface_hub.list_repo_files(repo_id=repo_id, token=token) if (f.endswith(".pt")
                                                                                           and hasdate(f)) ]

    files.sort(key=parse_model_date)

    return files[-1]

def ensure_model_exists(repo_id, token):
    global MODEL_FOUND
    global MODEL_PATH

    if MODEL_FOUND:
        return MODEL_PATH

    if LOCAL_MODEL_PATH.endswith(".pt") and os.path.isfile(LOCAL_MODEL_PATH):
        print(f"Using existing local model at {LOCAL_MODEL_PATH}.")
        MODEL_FOUND = True
        MODEL_PATH = LOCAL_MODEL_PATH
    else:
        print(f"Local model not found at {LOCAL_MODEL_PATH}. Downloading most recent model from HuggingFace...")
        os.makedirs(MODELS_FOLDER, exist_ok=True)
        try:
            most_recent_model = get_most_recent_model(repo_id, token)
        except Exception as e:
            print(e)
            raise Exception("Couldn't find a model to download. Check your huggingface information.")

       # downloaded_file = huggingface_hub.hf_hub_download(
       #         repo_id=repo_id,
       #         filename=most_recent_model,
       #         token=token,
       #         local_dir=MODELS_FOLDER)

        downloaded_file = custom_hub_download(repo_id=repo_id,
                                             filename=most_recent_model,
                                             token=token,
                                             local_dir=MODELS_FOLDER,
                                             callback=MODEL_DOWNLOAD_PROGRESS)

        MODEL_FOUND = True
        MODEL_PATH = os.path.join(MODELS_FOLDER, os.path.basename(most_recent_model))

        print(f"Model downloaded and stored at {MODEL_PATH}")

    return MODEL_PATH


#############################################
# 2) Video Tracking
#############################################
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


def run_tracking_on_folder(folder_path, output_types, frame_rate, um_per_pixel, repo_id, token):
    processed_files = []
    if not os.path.isdir(folder_path):
        return processed_files

    output_folder = os.path.join(folder_path, "output")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    model = YOLO(ensure_model_exists(repo_id, token))

    avi = "AVI" in output_types
    csv = "CSV" in output_types

    for file_name in os.listdir(folder_path):
        ext = file_name.lower()
        if not (ext.endswith('.avi') or ext.endswith('.mp4')):
            continue
        input_path = os.path.join(folder_path, file_name)

        track_video(
            input_path,
            model,
            frame_rate,
            settings.soft_get_setting("window_width"),
            settings.soft_get_setting("scaling_factor"),
            um_per_pixel,  
            output_folder,
            avi,
            csv
        )
        
        processed_files.append(file_name)
    return processed_files



#############################################
# 3) Roboflow Annotation Upload
#############################################
def upload_annotation_to_roboflow(api_key, workspace, project, image_bgr, shapes, frame_index, window_width):
    if not shapes or not isinstance(shapes, list):
        return "Error: No annotation data found."

    contours = []
    scale = 1  
    img_height, img_width = image_bgr.shape[:2]

    # Convert Dash annotation to OpenCV contours
    for s in shapes:
        if s.get("type") == "path":
            contour = dash_canvas_to_opencv(s, scale, img_height)
            if contour is not None and len(contour) > 0:
                contours.append(contour)

    if not contours:
        return "Error: No annotations detected."


    contours = [np.array(c, dtype=np.int32).reshape((-1, 1, 2)) for c in contours if len(c) > 0]


    left_bound = float("inf")
    right_bound = float("-inf")
    for ctr in contours:
        x, _, w, _ = cv2.boundingRect(ctr)
        left_bound = min(left_bound, x)
        right_bound = max(right_bound, x + w)
    left_bound = int(max(0, left_bound - window_width // 2))
    right_bound = int(min(img_width, right_bound + window_width // 2))


    cropped_image = image_bgr[:, left_bound:right_bound + 1]
    translated_contours = [ctr - [left_bound, 0] for ctr in contours]


    cropped_mask = np.zeros(cropped_image.shape[:2], np.uint8)
    cv2.drawContours(cropped_mask, translated_contours, -1, (255), -1)

    debug_overlay = np.ascontiguousarray(cropped_image.copy())
    cv2.drawContours(debug_overlay, translated_contours, -1, (0, 255, 0), 2)


    image_path, annotation_path, temp_dir = temp_construct_roboflow_annotation(cropped_image, cropped_mask)

    with open(annotation_path, "r") as json_file:
        json_content = json_file.read()

    try:
        rf = roboflow.Roboflow(api_key=api_key)
        project = rf.workspace(workspace).project(project)
        project.single_upload(image_path=image_path, annotation_path=annotation_path, batch_name="batch")
        return f"Annotation for frame {frame_index} uploaded to Roboflow successfully."
    except Exception as e:
        print(f"Upload error: {e}")
        return "Error: Upload failed. Check API credentials and network."



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

    # Ensure mask is grayscale
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    left_bound = float("inf")
    right_bound = float("-inf")

    for id, ctr in enumerate(contours):
        annotation = {}
        contour = ctr.flatten().tolist()
        contour_pairs = [(contour[i], contour[i+1]) for i in range(0, len(contour), 2)]
        segmentation = [float(coord) for pair in contour_pairs for coord in pair]

        area = cv2.contourArea(ctr)
        bbox = [ int(x) for x in cv2.boundingRect(ctr) ]
        left_bound = min(left_bound, bbox[0])
        right_bound = max(right_bound, bbox[0] + bbox[2])

        annotation["segmentation"] = [segmentation]
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
    


def dash_canvas_to_opencv(path_object, scale, img_height):
    """
    Converts Dash Canvas-drawn SVG path string into OpenCV-compatible contour format.
    Handles coordinate system and scaling mismatches.
    """
    if not path_object or "path" not in path_object:
        return []

    path_str = path_object["path"]
    path_commands = re.findall(r'[MLZ]|\d+\.\d+', path_str)

    if not path_commands:
        return []

    points = []
    i = 0
    while i < len(path_commands):
        cmd = path_commands[i]
        if cmd in ["M", "L"]:
            try:
                x = float(path_commands[i + 1]) / scale
                y = img_height - (float(path_commands[i + 2]) / scale) 
                points.append([x, y])
                i += 3
            except (ValueError, IndexError):
                print(f"Warning: Invalid coordinate values at {cmd}")
                i += 1
        elif cmd == "Z":
            if points:
                points.append(points[0])
            i += 1
        else:
            i += 1

    return np.array(points).reshape((-1, 1, 2)).astype(np.int32)





#############################################
# 4) Dash App
#############################################
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], update_title=None)

app.title = "Particle Tracking Dashboard"

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H2("Particle Tracking Interface", className="text-black mt-3"),
            html.Hr()
        ], width=12)
    ]),

    # Huggingface section
    dbc.Row([
        dbc.Col([
            html.H5("Huggingface Details", className="mt-4"),
            dbc.Input(id="huggingface-rep-id", placeholder="Enter Huggingface Rep ID", type="text", className="mb-3",
                      value=settings.soft_get_setting("huggingface_repo_id")),
            dbc.Input(id="huggingface-token", placeholder="Enter Huggingface Token", type="text", className="mb-3",
                      value=settings.soft_get_setting("huggingface_token")),
        ], width=6),
        
        # Roboflow section
        dbc.Col([
            html.H5("Roboflow Details", className="mt-4"),
            dbc.Input(id="roboflow-api-key", placeholder="Enter Roboflow API Key", type="text", className="mb-3",
                      value=settings.soft_get_setting("roboflow_api_key")),
            dbc.Input(id="roboflow-workspace", placeholder="Enter Roboflow Workspace", type="text", className="mb-3",
                      value=settings.soft_get_setting("roboflow_workspace_name")),
            dbc.Input(id="roboflow-project-id", placeholder="Enter Roboflow Project ID", type="text", className="mb-3",
                      value=settings.soft_get_setting("roboflow_project_name")),
            dbc.Input(id="roboflow-version", placeholder="Enter Roboflow Version", type="text", className="mb-3",
                      value=settings.soft_get_setting("roboflow_version_number")),
        ], width=6),
    ]),

    # Message after submission
    dbc.Row([
        dbc.Col([
            html.Div(id="submit-message", className="mt-3", style={"color": "green", "fontWeight": "bold"})
        ], width=12)
    ]),

    # MODEL STATUS SECTION
    dbc.Row([
        dbc.Col([
            html.H5("Model Status", className="mt-4"),
            dbc.Col([
                dbc.Progress(id="download-progress", value=0, striped=True, animated=True, style={"height":"30px"}, label="Download not started"),
                dcc.Interval(id="download-interval", n_intervals=0, interval=500)
            ], width=7),
        ])
    ]),


    # TRAINING SECTION
    dbc.Card([
        dbc.CardHeader(html.H4("Training", className="text-black mb-0")),
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
                html.Div(id="train-video-status", className="text-secondary mb-3"),
            ]),

            dbc.Row([
                dbc.Col([
                    dcc.Slider(
                        id="frame-slider",
                        min=0,
                        max=100,  
                        step=1,
                        value=0,
                        marks=None,
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                ], width=12),
            ], className="mb-3"),



            dbc.Row([
                dbc.Col([
                    html.Button(
                        "Identify Undetectable Frames", 
                        id="btn-auto-detect", 
                        n_clicks=0, 
                        className="btn btn-danger"
                    ),
                    html.Div(id="autodetect-status", className="text-secondary mt-2")
                ], width=12)
            ], className="mb-3"),


            dbc.Row([
                dbc.Col([
                    html.Label("Frames with No Detected Contours", className="fw-bold"),
                    dcc.Dropdown(
                        id="no-contour-dropdown",
                        options=[], 
                        placeholder="Select a frame with no contours"
                    )
                ], width=12)
            ], className="mb-3"),

            dbc.Row([
                dbc.Col([
                    html.Label("Annotation Tool", className="fw-bold"),
                ], width=12)
            ], className="mb-2"),  


            # Annotation Tool 
            dbc.Row([
                dbc.Col([
                    dcc.Graph(
                        id="annotation-graph",
                        config={
                            "modeBarButtonsToAdd": [
                                "drawclosedpath",  
                                "eraseshape", 
                                "lasso2d",  
                                "pan2d",  
                                "zoomIn2d", 
                                "zoomOut2d"                               
                            ],
                            "modeBarButtonsToRemove": ["resetScale2d","zoom2d","autoscale2d"],
                            "scrollZoom": True, 
                            "displaylogo": False  
                            
                        },
                        style={"width": "100%", "height": "250px", "border": "1px solid gray"},
                        figure=go.Figure().update_layout(
                            autosize=True,
                            xaxis=dict(visible=False, showgrid=False, zeroline=False),
                            yaxis=dict(visible=False, showgrid=False, zeroline=False),
                            margin=dict(l=0, r=0, t=0, b=0),
                            dragmode="drawclosedpath", 
                            plot_bgcolor="white",
                            paper_bgcolor="white",
                            newshape=dict(fillcolor="red", opacity=0.5, line=dict(color="red", width=3)),
                        )
                    )
                ], width=12)
            ], className="mb-3"),


            dbc.Row([
                dbc.Col([
                    html.Button("Save Annotation", id="btn-save-annotation", n_clicks=0, className="btn btn-warning"),
                ], width="auto")
            ], className="mt-3"),  

            dbc.Row([
                dbc.Col([
                    html.Div(id="save-annotation-status", className="text-secondary mt-2"),
    ])
]),




        ])
    ], className="my-3"),

    # TRACKING SECTION
    dbc.Card([
        dbc.CardHeader(html.H4("Tracking", className="text-black mb-0")),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Folder Path for Videos", className="fw-bold"),
                    dcc.Input(
                        id="tracking-folder-path",
                        type="text",
                        placeholder=r"Enter Video Folder Path",
                        className="form-control",
                        value=settings.soft_get_setting("video_folder")
                    )
                ], width=6),
                dbc.Col([
                    html.Label("Frame Rate (μs/frame)", className="fw-bold"),
                    dcc.Input(
                        id="tracking-frame-rate",
                        type="number",
                        placeholder="Enter Frametrate",
                        className="form-control",
                        value=settings.soft_get_setting("framerate")
                    )
                ], width=3),
                dbc.Col([
                    html.Label("Pixel Size (μm/pixel)", className="fw-bold"),
                    dcc.Input(
                        id="tracking-um-per-pixel",
                        type="number",
                        placeholder="Enter Scaling Factor",
                        className="form-control",
                        value=settings.soft_get_setting("um_per_pixel")
                    )
                ], width=3),
            ], className="mb-3"),

            dbc.Row([
                dbc.Col([
                    html.Label("Export Options:", className="fw-bold me-3"),
                ], width="auto", className="d-flex align-items-center"),
                dbc.Col([
                    dbc.Checklist(
                        options=[
                            {"label": "CSV", "value": "CSV"},
                            {"label": "AVI", "value": "AVI"},
                        ],
                        value=["CSV", "AVI"],
                        id="export-options",
                        inline=True,
                        className="ms-2"
                    )
                ], width="auto", className="d-flex align-items-center"),
            ], className="mb-3"),

            dbc.Row([
                dbc.Col([
                    html.Button(
                        "Run Tracking",
                        id="btn-run-tracking",
                        n_clicks=0,
                        className="btn btn-success px-4"
                    ),
                ], width="auto", className="d-flex align-items-center"),
                dbc.Col([
                    dbc.Progress(id="tracking-progress", value=0, striped=True, animated=True, style={"height":"30px"}),
                    dcc.Interval(id="progress-interval", n_intervals=0, interval=500)
                ], width=7),
            ], className="mb-3 d-flex align-items-center"),

            html.Div(id="tracking-status", className="text-secondary mt-2"),
        ])
], className="my-3"),



dcc.Store(id='training-frames', data=[]),
dcc.Store(id='autodetected-bboxes', data={}),
dcc.Store(id='processed-frames', data=[]),
dcc.Store(id='uploaded-video-path', data=""),
dcc.Store(id='no-contour-indices', data=[]),
], fluid=True)



#############################################
# 5) Callbacks
#############################################

# -- TRAINING VIDEO UPLOAD --
@app.callback(
    Output("training-frames", "data"),
    Output("train-video-status", "children"),
    Output("frame-slider", "max"),   
    Output("uploaded-video-path", "data"),
    Input("upload-training-video", "contents"),
    prevent_initial_call=True
)
def on_training_video_upload(contents):
    if not contents:
        raise PreventUpdate

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    temp_file = tempfile.NamedTemporaryFile(suffix=".avi", delete=False)
    temp_file.write(decoded)
    temp_file.close()
    temp_video_path = temp_file.name

    cap = cv2.VideoCapture(temp_video_path)
    frames_data = []
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        _, buffer = cv2.imencode('.jpg', frame)
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
        frames_data.append(frame_b64)
    cap.release()

    if not frames_data:
        return [], "Error: No frames extracted from video.", 0, ""

    status = f"Uploaded video with {len(frames_data)} frames."
    return frames_data, status, len(frames_data) - 1, temp_video_path



@app.callback(
    Output("processed-frames", "data"),
    Output("no-contour-indices", "data"),
    Output("autodetect-status", "children"),
    Input("btn-auto-detect", "n_clicks"),
    State("training-frames", "data"),
    State("huggingface-rep-id", "value"),
    State("huggingface-token", "value"),
    State("tracking-um-per-pixel", "value")
    prevent_initial_call=True
)
def run_full_inference(n_clicks, frames, repo_id, token, um_per_pixel):
    if not frames:
        raise PreventUpdate

    # Save the training frames to a temporary video file
    temp_video_path = os.path.join(tempfile.gettempdir(), "temp_auto_detect.avi")
    first_frame = cv2.imdecode(np.frombuffer(base64.b64decode(frames[0]), np.uint8), cv2.IMREAD_COLOR)
    height, width, _ = first_frame.shape
    fps = 25  # adjust if needed
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out_video = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
    for f_b64 in frames:
        decoded = base64.b64decode(f_b64)
        np_arr = np.frombuffer(decoded, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        out_video.write(img)
    out_video.release()


    output_folder = tempfile.mkdtemp()

    pipeline = InferencePipeline(
        model=YOLO(ensure_model_exists(repo_id, token)),
        framerate=fps,
        window_width=settings.soft_get_setting("window_width"),
        scaling_factor=settings.soft_get_setting("scaling_factor"),
        um_per_pixel=um_per_pixel,
        output_folder=output_folder
    )

    pipeline.process_video(temp_video_path, avi=True, csv=True, include_plots=False)


    processed_video_path = os.path.join(output_folder, "temp_auto_detect-analysis.avi")
    processed_frames = []
    cap = cv2.VideoCapture(processed_video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        _, buffer = cv2.imencode('.jpg', frame)
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
        processed_frames.append(frame_b64)
    cap.release()


    csv_path = os.path.join(output_folder, "temp_auto_detect-analysis.csv")
    no_contour_indices = []
    try:
        with open(csv_path, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for idx, row in enumerate(reader):
                try:
                    area = float(row.get("area", 0))
                except Exception:
                    area = 0
                if area == 0:
                    no_contour_indices.append(idx)
    except Exception as e:
        no_contour_indices = []

    os.remove(temp_video_path)
    return processed_frames, no_contour_indices, "Detection complete and video updated with tracked contours."



# -- GO TO FRAME & DISPLAY --
@app.callback(
    Output("annotation-graph", "figure"),
    Output("frame-slider", "value"),
    Output("autodetected-bboxes", "data"),
    Input("frame-slider", "value"),
    Input("no-contour-dropdown", "value"),
    State("training-frames", "data"),
    State("processed-frames", "data"),
    State("annotation-graph", "relayoutData"),
    State("no-contour-indices", "data"),
    prevent_initial_call=True
)
def update_annotation_display(slider_value, dropdown_value, training_frames, processed_frames, relayout_data, no_contour_indices):
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if trigger_id == "frame-slider":
        frame_idx = slider_value
    elif trigger_id == "no-contour-dropdown" and dropdown_value is not None:
        frame_idx = dropdown_value
    else:
        frame_idx = slider_value

    frame_idx = max(0, min(len(training_frames) - 1, frame_idx))
    if not training_frames:
        return go.Figure(), frame_idx, {}
    

    if no_contour_indices and frame_idx in no_contour_indices:
        frame_b64 = training_frames[frame_idx]
    else:
        if processed_frames and len(processed_frames) > 0:
            frame_b64 = processed_frames[frame_idx]
        else:
            frame_b64 = training_frames[frame_idx]

    decoded = base64.b64decode(frame_b64)
    np_arr = np.frombuffer(decoded, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    

    fig = go.Figure()
    fig.add_layout_image(
        dict(
            source=f"data:image/jpg;base64,{frame_b64}",
            x=0,
            y=img.shape[0],
            xref="x",
            yref="y",
            sizex=img.shape[1],
            sizey=img.shape[0],
            xanchor="left",
            yanchor="top",
            layer="below"
        )
    )

    if relayout_data and "xaxis.range" in relayout_data and "yaxis.range" in relayout_data:
        x_range = relayout_data["xaxis.range"]
        y_range = relayout_data["yaxis.range"]
    else:
        x_range = [0, img.shape[1]]
        y_range = [0, img.shape[0]]

    fig.update_layout(
        autosize=True,
        xaxis=dict(visible=False, showgrid=False, zeroline=False, range=x_range, scaleanchor="y"),
        yaxis=dict(visible=False, showgrid=False, zeroline=False, range=y_range, scaleanchor="x"),
        margin=dict(l=0, r=0, t=0, b=0),
        dragmode="pan",
        plot_bgcolor="white",
        paper_bgcolor="white"
    )


    return fig, frame_idx, {}




@app.callback(
    Output("no-contour-dropdown", "options"),
    Input("no-contour-indices", "data"),
    prevent_initial_call=True
)
def update_no_contour_dropdown_from_inference(no_contour_indices):
    if no_contour_indices is None:
        raise PreventUpdate
    options = [{"label": f"Frame {i}", "value": i} for i in no_contour_indices]
    return options




# -- SAVE ANNOTATION to Roboflow --
@app.callback(
    Output("save-annotation-status", "children"),
    Input("btn-save-annotation", "n_clicks"),
    State("roboflow-api-key", "value"),
    State("roboflow-workspace", "value"),
    State("roboflow-project-id", "value"),
    State("frame-slider", "value"),
    State("training-frames", "data"),
    State("annotation-graph", "figure"),  
    prevent_initial_call=True
)
def save_annotation(n_clicks, api_key, workspace, project, frame_idx, frames, figure):
    if not frames or frame_idx < 0 or frame_idx >= len(frames):
        raise PreventUpdate
    

    shapes = figure.get("layout", {}).get("shapes", None)
    if not shapes:
        return "Error: No annotation data found."

    frame_b64 = frames[frame_idx]
    dec = base64.b64decode(frame_b64)
    np_arr = np.frombuffer(dec, np.uint8)
    frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    upload_result = upload_annotation_to_roboflow(api_key, workspace, project, frame_bgr, shapes, frame_idx, settings.soft_get_setting("window_width"))
    return upload_result



# -- RUN TRACKING ON A FOLDER --
@app.callback(
    Output("tracking-progress", "value", allow_duplicate=True),
    Output("tracking-progress", "label", allow_duplicate=True),
    Output("tracking-status", "children"),
    Input("btn-run-tracking", "n_clicks"),
    State("tracking-folder-path", "value"),
    State("export-options", "value"),
    State("tracking-frame-rate", "value"),
    State("tracking-um-per-pixel", "value"), 
    State("huggingface-rep-id", "value"),
    State("huggingface-token", "value"),
    prevent_initial_call=True
)
def run_tracking(n_clicks, folder_path, export_values, frame_rate, um_per_pixel, repo_id, token):
    if not folder_path:
        raise PreventUpdate

    processed = run_tracking_on_folder(folder_path, export_values, frame_rate, um_per_pixel, repo_id, token)
    if not processed:
        return 0, "", "No videos processed. Check folder path or no .avi/.mp4 found."

    return 0, "100%", "All videos processed!"


@app.callback(
    Output("tracking-progress", "value"),
    Output("tracking-progress", "label"),
    Input("progress-interval", "n_intervals"),
    Input("tracking-progress", "value"),
    Input("tracking-progress", "label"))
def update_inference_progress(n, value, label):
    if INFERENCE_PIPELINE == None:
        return value, label

    progress = int(INFERENCE_PIPELINE.progress * 100)

    return progress, f"{progress}%" if progress >= 5 else ""

@app.callback(
        Output("download-progress", "value"),
        Output("download-progress", "label"),
        Input("download-interval", "n_intervals"),
        Input("download-progress", "value"),
        Input("download-progress", "value"))
def update_download_progress(n, value, label):

    progress = int(MODEL_DOWNLOAD_PROGRESS.value * 100)

    return progress, f"{progress}%" if progress >= 5 else ""



def custom_hub_download(repo_id, filename, token=None, local_dir=None, callback=None, revision=None):
    url = huggingface_hub.hf_hub_url(repo_id=repo_id, filename=filename, revision=revision)

    headers = {"Authorization": f"Bearer {token}"} if token else {}

    response = requests.head(url, headers=headers, allow_redirects=True)
    response.raise_for_status()  # Ensure the request was successful
    file_size = int(response.headers.get('content-length', 0))

    if local_dir is None:
        local_dir = "."

    local_file_path = f"{local_dir}/{filename}"
    
    if os.path.exists(local_file_path):
        callback.value = 1
        return 1

    total = 0

    with requests.get(url, headers=headers, stream=True) as r, open(local_file_path, 'wb') as f, tqdm(
        desc=filename,
        total=file_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))
            total += len(chunk) / file_size
            callback.value = total

    return local_file_path


def main():
    app.run(debug=True)

if __name__ == "__main__":
    main()
