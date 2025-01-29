import os
import glob
import uuid
import time
import math
import base64
import cv2
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from roboflow import Roboflow

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, no_update
from dash.exceptions import PreventUpdate

# Dash Canvas for annotation
from dash_canvas import DashCanvas
from dash_canvas.utils import parse_jsonstring


#
# ---------------------------------------------------------------------
# 1) HELPER FUNCTIONS
# ---------------------------------------------------------------------
#
def background_subtraction(video_path, output_path=None):
    """
    Example background subtraction using OpenCV MOG2.
    Replace with your real script if needed.
    """
    if output_path is None:
        output_path = os.path.splitext(video_path)[0] + "_bg_sub.avi"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, detectShadows=False)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h), False)  # grayscale

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fgmask = fgbg.apply(frame)
        out.write(fgmask)
    cap.release()
    out.release()
    return output_path

def load_hf_model(repo_id, filename, token):
    """
    Download a model file (.pt) from Hugging Face.
    """
    downloaded_path = hf_hub_download(repo_id=repo_id, filename=filename, token=token)
    return downloaded_path

def autodetect_particles_in_video(video_path, model_path, output_csv_path=None):
    """
    Mock function for detection. Replace with your real model inference code.
    """
    if output_csv_path is None:
        output_csv_path = os.path.splitext(video_path)[0] + "_detections.csv"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video for autodetection: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    detections = []
    for f_idx in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        # Example: Single bounding box in center
        h, w = frame.shape[:2]
        box_w, box_h = 50, 50
        x = (w - box_w) // 2
        y = (h - box_h) // 2
        detections.append([f_idx, x, y, box_w, box_h])

    cap.release()
    df = pd.DataFrame(detections, columns=['frame_idx', 'x', 'y', 'w', 'h'])
    df.to_csv(output_csv_path, index=False)
    return output_csv_path

def train_model(api_key, hf_repo_id, hf_token, selected_weight, epochs, batch_size, patience):
    """
    Mock training routine. Replace with your real training code.
    """
    time.sleep(1)  # pretend training
    msg = (f"Training (mock) with epochs={epochs}, batch_size={batch_size}, "
           f"patience={patience}. Base model: {selected_weight}.")
    return msg

def upload_annotation_to_roboflow(api_key, workspace, project_name, image_path):
    """
    Upload an annotated image to your Roboflow project.
    """
    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project_name)
    upload_info = proj.upload(image_path)
    return upload_info

def base64_encode_image(raw_bytes):
    return base64.b64encode(raw_bytes).decode('utf-8')

def measure_particle_properties(cnt):
    """
    Return dict with area, perimeter, bounding-box height, circularity, etc.
    """
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    x, y, w, h = cv2.boundingRect(cnt)
    circularity = 4 * math.pi * (area / ((perimeter + 1e-8) ** 2))

    return {
        'area': area,
        'height': h,
        'perimeter': perimeter,
        'circularity': circularity
    }

def track_video(video_path, model_path, fps, export_csv=False, export_avi=False):
    """
    Example "tracking" function:
      1) Background subtract
      2) Find largest contour
      3) Save CSV + annotated AVI
    """
    # BG-sub
    bg_sub_video_path = background_subtraction(video_path)

    cap = cv2.VideoCapture(bg_sub_video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open BG-subtracted video: {bg_sub_video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if export_avi:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_avi_path = os.path.splitext(video_path)[0] + "_tracked.avi"
        out_avi = cv2.VideoWriter(out_avi_path, fourcc, fps, (width, height))

    rows = []
    prev_center = None
    prev_velocity = 0

    for f_idx in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        # threshold
        _, thresh = cv2.threshold(frame, 30, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            props = measure_particle_properties(cnt)
            M = cv2.moments(cnt)
            cx = int(M["m10"] / (M["m00"] + 1e-8))
            cy = int(M["m01"] / (M["m00"] + 1e-8))

            # velocity, acceleration
            if prev_center is not None:
                dist = math.hypot(cx - prev_center[0], cy - prev_center[1])
                velocity = dist * fps
                acceleration = (velocity - prev_velocity) * fps
            else:
                velocity = 0
                acceleration = 0

            row = {
                'frame': f_idx,
                'time': f_idx / fps,
                'x': cx,
                'y': cy,
                'area': props['area'],
                'height': props['height'],
                'perimeter': props['perimeter'],
                'circularity': props['circularity'],
                'velocity': velocity,
                'acceleration': acceleration,
            }
            rows.append(row)

            prev_center = (cx, cy)
            prev_velocity = velocity

            if export_avi:
                color_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                cv2.drawContours(color_frame, [cnt], -1, (0, 0, 255), 2)
                cv2.circle(color_frame, (cx, cy), 5, (255, 0, 0), -1)
                out_avi.write(color_frame)
        else:
            if export_avi:
                out_avi.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))

    cap.release()
    if export_avi:
        out_avi.release()

    if export_csv:
        df = pd.DataFrame(rows)
        csv_out_path = os.path.splitext(video_path)[0] + "_tracked.csv"
        df.to_csv(csv_out_path, index=False)

    return True


#
# ---------------------------------------------------------------------
# 2) APP LAYOUT
# ---------------------------------------------------------------------
#
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dcc.Store(id='stored-roboflow-api-key'),
    dcc.Store(id='stored-hf-repo-id'),
    dcc.Store(id='stored-hf-token'),
    dcc.Store(id='stored-selected-weight'),
    dcc.Store(id='stored-training-params'),
    dcc.Store(id='uploaded-training-video-path'),

    html.H1("Cell Tracking Interface"),

    dbc.Row([
        #
        # SETTINGS
        #
        dbc.Col([
            html.H3("Settings", className="text-primary mt-3"),

            html.H6("Roboflow Credentials"),
            dbc.InputGroup([
                dbc.Input(id="roboflow_api_key", type="text", placeholder="Roboflow API Key"),
                dbc.Button("Load", id="load_roboflow_key", color="primary")
            ], className="mb-2"),
            html.Div(id="roboflow_status", className="text-secondary mb-3"),

            html.H6("Hugging Face Credentials"),
            dbc.InputGroup([
                dbc.Input(id="hf_repo_id", type="text", placeholder="Repo ID"),
                dbc.Input(id="hf_token", type="text", placeholder="HF Token"),
                dbc.Button("Load HF", id="load_hf_token", color="primary")
            ], className="mb-2"),
            html.Div(id="huggingface_status", className="text-secondary mb-3"),

            html.H6("Select Model Weight"),
            dcc.Dropdown(
                id="hf_model_weights_dropdown",
                options=[],
                placeholder="No weights loaded yet",
                className="mb-2"
            ),
            html.Div(id="model_weight_status", className="text-secondary mb-3"),

            html.H6("Training Params"),
            dbc.InputGroup([
                dbc.Input(id="training_epochs", type="number", placeholder="epochs"),
                dbc.Input(id="training_batch", type="number", placeholder="batch size"),
                dbc.Input(id="training_patience", type="number", placeholder="patience"),
                dbc.Button("Set Params", id="load_training_params", color="primary")
            ], className="mb-2"),
            html.Div(id="training_params_status", className="text-secondary mb-3"),
        ], width=3),

        #
        # TRAINING
        #
        dbc.Col([
            html.H3("Training", className="text-primary mt-3"),

            html.H6("Upload Training Video"),
            dcc.Upload(
                id="upload_training_video",
                children=html.Div(["Drag & Drop or Click to Select Video"]),
                style={
                    "width": "100%",
                    "height": "60px",
                    "lineHeight": "60px",
                    "borderWidth": "1px",
                    "borderStyle": "dashed",
                    "borderRadius": "5px",
                    "textAlign": "center",
                    "marginBottom": "10px",
                },
                multiple=False
            ),
            html.Div(id="video_upload_status", className="text-warning mb-3"),  # <--- New for upload messages

            dbc.Button("Apply Autodetection", id="apply_autodetection_btn", color="info", className="mb-2"),
            html.Div(id="autodetection_status", className="text-warning mb-3"),

            # Annotation
            html.H5("Annotation"),
            dbc.InputGroup([
                dbc.Input(id="annotation_frame_number", type="number", placeholder="Frame #"),
                dbc.Button("Load Frame", id="annotation_load_frame_btn", color="secondary")
            ], className="mb-2"),
            dcc.Slider(
                id="annotation_frame_slider",
                min=0, max=0, step=1, value=0,
                marks=None,
                tooltip={"always_visible": True},
                className="mb-2"
            ),

            DashCanvas(
                id='annotation_canvas',
                width=480,
                height=360,
                lineWidth=2,
                goButtonTitle='Annotate',
                tool='rectangle',
                filename='',
                hide_buttons=['zoom', 'pan', 'line', 'pencil', 'select']
            ),
            dbc.Button("Save Annotation", id="save_annotation_btn", color="success", className="my-2"),
            html.Div(id="annotation_save_status", className="text-success mb-3"),

            dbc.Button("Train Model", id="train_model_btn", color="danger", className="mb-2"),
            html.Div(id="training_status", className="text-info mb-2"),

        ], width=5),

        #
        # TRACKING
        #
        dbc.Col([
            html.H3("Tracking", className="text-primary mt-3"),

            dbc.InputGroup([
                dbc.Input(id="tracking_folder_path", placeholder="Folder containing videos"),
                dbc.Button("Set Folder", id="set_folder_btn", color="primary")
            ], className="mb-2"),

            dbc.InputGroup([
                dbc.Input(id="tracking_frame_rate", type="number", placeholder="Frame Rate"),
                dbc.Button("Set Rate", id="set_frame_rate_btn", color="primary")
            ], className="mb-2"),

            dbc.Label("Export Options:", className="mt-2"),
            dbc.Checklist(
                options=[
                    {"label": "CSV of Raw Data", "value": "csv"},
                    {"label": "AVI Video w/ Tracking", "value": "avi"},
                ],
                value=["csv"],
                id="export_file_types",
                inline=True,
                className="mb-3"
            ),

            dbc.Button("Run Tracking", id="run_tracking_btn", color="warning", className="mb-2"),

            dbc.Progress(
                id="tracking_progress_bar",
                value=0,
                striped=True,
                animated=True,
                style={"height": "20px"},
                className="mb-2"
            ),

            html.Div(id="tracking_status", className="text-primary"),
        ], width=4)
    ])
], fluid=True)

#
# ---------------------------------------------------------------------
# 3) CALLBACKS
# ---------------------------------------------------------------------
#

#
# ====== SETTINGS ======
#
@app.callback(
    Output("stored-roboflow-api-key", "data"),
    Output("roboflow_status", "children"),
    Input("load_roboflow_key", "n_clicks"),
    State("roboflow_api_key", "value"),
    prevent_initial_call=True
)
def store_roboflow_key(n_clicks, rf_key):
    if not rf_key:
        return no_update, "No Roboflow API key entered."
    return rf_key, f"Roboflow key loaded: {rf_key[:10]}..."

@app.callback(
    Output("stored-hf-repo-id", "data"),
    Output("stored-hf-token", "data"),
    Output("hf_model_weights_dropdown", "options"),
    Output("huggingface_status", "children"),
    Input("load_hf_token", "n_clicks"),
    State("hf_repo_id", "value"),
    State("hf_token", "value"),
    prevent_initial_call=True
)
def store_hf_creds(n_clicks, repo_id, token):
    if not repo_id or not token:
        return no_update, no_update, no_update, "Enter valid Hugging Face info."
    # Known model "sep13.pt", for example
    model_list = [{"label": "sep13.pt", "value": "sep13.pt"}]
    msg = f"Hugging Face creds loaded. Found {len(model_list)} weight(s)."
    return repo_id, token, model_list, msg

@app.callback(
    Output("stored-selected-weight", "data"),
    Output("model_weight_status", "children"),
    Input("hf_model_weights_dropdown", "value"),
    State("stored-hf-repo-id", "data"),
    State("stored-hf-token", "data"),
    prevent_initial_call=True
)
def select_model_weight(weight_filename, repo_id, token):
    if not weight_filename:
        raise PreventUpdate
    try:
        downloaded_path = load_hf_model(repo_id, weight_filename, token)
        msg = f"Selected weight: {weight_filename}, downloaded to {downloaded_path}."
        return weight_filename, msg
    except Exception as e:
        return no_update, f"Error downloading model: {str(e)}"

@app.callback(
    Output("stored-training-params", "data"),
    Output("training_params_status", "children"),
    Input("load_training_params", "n_clicks"),
    State("training_epochs", "value"),
    State("training_batch", "value"),
    State("training_patience", "value"),
    prevent_initial_call=True
)
def store_training_params(n_clicks, epochs, batch_size, patience):
    if not epochs or not batch_size or not patience:
        return no_update, "Incomplete training parameters."
    params = {"epochs": epochs, "batch_size": batch_size, "patience": patience}
    return params, f"Training params set: {params}"

#
# ====== TRAINING ======
#

# 1) Upload training video
@app.callback(
    Output("uploaded-training-video-path", "data"),
    Output("video_upload_status", "children"),  # <-- replaced autodetection_status to avoid duplication
    Input("upload_training_video", "contents"),
    State("upload_training_video", "filename"),
    prevent_initial_call=True
)
def on_training_video_upload(contents, filename):
    if not contents:
        raise PreventUpdate
    # decode base64
    content_type, content_string = contents.split(",", 1)
    decoded = base64.b64decode(content_string)

    unique_name = f"training_{uuid.uuid4().hex}_{filename}"
    with open(unique_name, "wb") as f:
        f.write(decoded)

    return unique_name, f"Uploaded training video: {filename}"

# 2) Apply autodetection
@app.callback(
    Output("autodetection_status", "children"),
    Input("apply_autodetection_btn", "n_clicks"),
    State("uploaded-training-video-path", "data"),
    State("stored-selected-weight", "data"),
    prevent_initial_call=True
)
def autodetect_btn_clicked(n_clicks, video_path, model_weight):
    if not video_path:
        return "No video to process for autodetection."
    if not model_weight:
        return "No model weight selected."

    try:
        bg_sub_path = background_subtraction(video_path)
        out_csv = autodetect_particles_in_video(bg_sub_path, model_weight)
        return f"Autodetection complete. Results in {out_csv}"
    except Exception as e:
        return f"Autodetection error: {str(e)}"

# 3) Load frame for annotation
@app.callback(
    Output("annotation_canvas", "image_content"),
    Output("annotation_frame_slider", "max"),
    Output("annotation_frame_slider", "value"),
    Input("annotation_load_frame_btn", "n_clicks"),
    State("uploaded-training-video-path", "data"),
    State("annotation_frame_number", "value"),
    prevent_initial_call=True
)
def load_frame_for_annotation(n_clicks, video_path, frame_num):
    if not video_path or frame_num is None:
        raise PreventUpdate
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_num < 0:
        frame_num = 0
    if frame_num >= total_frames:
        frame_num = total_frames - 1

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise PreventUpdate

    success, buf = cv2.imencode(".png", frame)
    if not success:
        raise PreventUpdate
    b64_image = "data:image/png;base64," + base64_encode_image(buf.tobytes())

    return b64_image, total_frames - 1, frame_num

# 4) Save annotation
@app.callback(
    Output("annotation_save_status", "children"),
    Input("save_annotation_btn", "n_clicks"),
    State("annotation_canvas", "json_data"),
    State("annotation_canvas", "image_content"),
    State("stored-roboflow-api-key", "data"),
    prevent_initial_call=True
)
def save_annotation_btn(n_clicks, canvas_data, b64_image, rf_api_key):
    if not b64_image:
        return "No image loaded."
    header, encoded = b64_image.split(",", 1)
    decoded = base64.b64decode(encoded)
    np_img = np.frombuffer(decoded, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    shapes = parse_jsonstring(canvas_data, shape_type='rectangle')
    for shape in shapes:
        x0, y0 = shape['x0'], shape['y0']
        x1, y1 = shape['x1'], shape['y1']
        cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (0,255,0), 2)

    out_name = f"annotated_{uuid.uuid4().hex}.png"
    cv2.imwrite(out_name, img)

    if rf_api_key:
        try:
            # Adjust "workspace" & "project_name" to your environment
            # e.g. workspace="gt-sulchek-lab", project_name="sep13"
            _info = upload_annotation_to_roboflow(
                api_key=rf_api_key,
                workspace="gt-sulchek-lab",
                project_name="sep13",
                image_path=out_name
            )
            return f"Annotation saved: {out_name}, uploaded to Roboflow."
        except Exception as e:
            return f"Saved: {out_name}, Roboflow upload failed: {str(e)}"
    else:
        return f"Annotation saved locally: {out_name}"

# 5) Train model
@app.callback(
    Output("training_status", "children"),
    Input("train_model_btn", "n_clicks"),
    State("stored-roboflow-api-key", "data"),
    State("stored-hf-repo-id", "data"),
    State("stored-hf-token", "data"),
    State("stored-selected-weight", "data"),
    State("stored-training-params", "data"),
    prevent_initial_call=True
)
def on_train_model(n_clicks, rf_key, hf_repo, hf_token, sel_weight, params):
    if not all([rf_key, hf_repo, hf_token, sel_weight, params]):
        return "Missing settings or parameters."
    msg = train_model(
        api_key=rf_key,
        hf_repo_id=hf_repo,
        hf_token=hf_token,
        selected_weight=sel_weight,
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        patience=params["patience"]
    )
    return msg

#
# ====== TRACKING ======
#
@app.callback(
    Output("tracking_status", "children"),
    Output("tracking_progress_bar", "value"),
    Input("run_tracking_btn", "n_clicks"),
    State("tracking_folder_path", "value"),
    State("tracking_frame_rate", "value"),
    State("export_file_types", "value"),
    prevent_initial_call=True
)
def run_tracking(n_clicks, folder_path, fps, export_types):
    if not folder_path or not os.path.isdir(folder_path):
        return "Invalid folder path.", 0
    if not fps:
        return "Frame rate not set.", 0

    # gather videos
    all_videos = glob.glob(os.path.join(folder_path, "*.mp4"))
    all_videos += glob.glob(os.path.join(folder_path, "*.avi"))
    if not all_videos:
        return "No videos found in folder.", 0

    step = max(1, 100 // len(all_videos))
    progress_val = 0

    for i, vid in enumerate(all_videos):
        track_video(
            video_path=vid,
            model_path="sep13.pt",  # or your local model path
            fps=fps,
            export_csv=("csv" in export_types),
            export_avi=("avi" in export_types)
        )
        progress_val = min(progress_val + step, 100)

    return f"Tracking complete for {len(all_videos)} video(s).", 100

#
# ---------------------------------------------------------------------
# 4) RUN SERVER
# ---------------------------------------------------------------------
#
if __name__ == "__main__":
    app.run_server(debug=True)
