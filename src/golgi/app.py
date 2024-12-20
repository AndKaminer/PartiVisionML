
import dash
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import os
import cv2
import base64
from golgi.inference.main import get_available_weights, download_model, main
from golgi.annotation.main import annotate_image  # Refactored annotation function
from golgi.training.main import train_model

UPLOAD_FOLDER = "uploads"
FRAME_FOLDER = "frames"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FRAME_FOLDER, exist_ok=True)

# Global variables for frame navigation
frame_paths = []
current_frame_index = 0


def create_app():
    """
    Create and configure the Dash app.

    Returns:
        app (Dash): The Dash app instance.
    """
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    app.scripts.config.serve_locally = True
    app.css.config.serve_locally = True



    # Define app layout
    app.layout = dbc.Container([
        html.H1("Golgi Management Dashboard", className="my-4"),

        # Video Upload Section


        html.H4(
            "Annotation",
            style={"fontSize": "32px", "textAlign": "center", "fontWeight": "bold"}  # Change font size here
        ),
        dbc.Label("Upload Video (.avi):"),
        dcc.Upload(
            id="upload-video",
            children=html.Button("Upload Video"),
            accept=".avi",
            style={"width": "100%", "padding": "10px", "textAlign": "center", "border": "1px dashed"}
        ),
        html.Div(id="upload-status", style={"marginTop": "30px"}),

        # Frame Navigation Section
        html.H4("Navigate Frames"),
        html.Div([
            html.Div(id="frame-number", style={"marginBottom": "10px"}),

            # Manual Input for Frame Number
            dbc.Label("Go to Frame:"),
            dcc.Input(
                id="frame-input",
                type="number",
                placeholder="Enter frame number",
                style={"width": "100%", "marginBottom": "10px"}
            ),

            # Arrow Buttons
            dbc.Button("< Previous", id="back-button", color="primary", className="me-2"),
            dbc.Button("Next >", id="forward-button", color="primary"),
        ], style={"textAlign": "center", "marginBottom": "20px"}),

        # Frame Display Section
        html.Div(id="frame-display", style={"textAlign": "center", "marginBottom": "20px"}),

        # Annotation Section
        html.H4("Annotate Frame"),
        dbc.Label("Annotation Type:"),
        dcc.Dropdown(
            id="annotation-type",
            options=[
                {"label": "Auto", "value": "auto"},
                {"label": "Manual", "value": "manual"}
            ],
            placeholder="Select annotation type"
        ),
        dbc.Label("Model Name:"),
        dcc.Input(id="model-name", type="text", placeholder="Enter model name"),
        dbc.Label("API Key: "),
        dcc.Input(id="api-key", type="text", placeholder="Enter API key"),
        dbc.Button("Annotate Frame", id="annotate-button", color="primary", className="my-2"),
        html.Div(id="annotation-status", style={"marginTop": "20px"}),

        # Tracking Section

        html.H4(
            "Tracking",
            style={"fontSize": "32px", "textAlign": "center", "fontWeight": "bold"}  # Change font size here
        ),
        dbc.Label("Video File Path (.avi):"),
        dcc.Input(
            id="video-path",
            type="text",
            placeholder="Enter the full path to the .avi file",
            style={"width": "100%", "marginBottom": "10px"}
        ),
        dbc.Label("Output Folder:"),
        dcc.Input(
            id="output-folder",
            type="text",
            placeholder="Enter the output folder path",
            style={"width": "100%", "marginBottom": "10px"}
        ),
        dbc.Label("Weight Name:"),
        dcc.Dropdown(
            id="weight-name-tracking",
            options=[{"label": weight, "value": weight} for weight in get_available_weights()],
            placeholder="Select a weight name",
            style={"width": "100%", "marginBottom": "10px"}
        ),
        dbc.Button("Run Tracking", id="run-tracking-button", color="primary", className="my-2"),
        html.Div(id="tracking-status", style={"marginTop": "20px"}),

        # Weight Management Section
        html.Hr(),
        html.H4("Weight Management"),
        dbc.Button("List Available Weights", id="list-weights-button", color="primary", className="my-2"),
        html.Div(id="weights-list", style={"marginTop": "30px"}),

        dbc.Label("Repository ID:"),
        dcc.Input(
            id="repo-id",
            type="text",
            placeholder="Enter the Huggingface repository ID",
            style={"width": "100%", "marginBottom": "10px"}
        ),
        dbc.Label("Model Name:"),
        dcc.Input(
            id="model-name-download",
            type="text",
            placeholder="Enter the model name",
            style={"width": "100%", "marginBottom": "10px"}
        ),
        dbc.Label("Huggingface Token:"),
        dcc.Input(
            id="huggingface-token",
            type="password",
            placeholder="Enter your Huggingface token",
            style={"width": "100%", "marginBottom": "10px"}
        ),
        dbc.Button("Download Model", id="download-weights-button", color="primary", className="my-2"),
        html.Div(id="download-status", style={"marginTop": "20px"}),

        # Training
        html.Hr(),
        html.H4("Training Feature"),
        dbc.Label("Epochs:"),
        dcc.Input(
            id="epochs",
            type="number",
            placeholder="Number of epochs",
            style={"width": "100%", "marginBottom": "10px"}
        ),
        dbc.Label("Batch Size:"),
        dcc.Input(
            id="batch",
            type="number",
            placeholder="Batch size",
            style={"width": "100%", "marginBottom": "10px"}
        ),
        dbc.Label("Patience:"),
        dcc.Input(
            id="patience",
            type="number",
            placeholder="Patience for early stopping",
            style={"width": "100%", "marginBottom": "10px"}
        ),
        dbc.Label("Weight Destination:"),
        dcc.Input(
            id="weight-destination",
            type="text",
            placeholder="Destination folder for weights",
            style={"width": "100%", "marginBottom": "10px"}
        ),
        dbc.Label("API Key:"),
        dcc.Input(
            id="api-key-train",
            type="password",
            placeholder="Enter API key",
            style={"width": "100%", "marginBottom": "10px"}
        ),
        dbc.Button("Train Model", id="train-button", color="primary", className="my-2"),
        html.Div(id="training-status", style={"marginTop": "20px"}),

    ])

    # Callbacks
    @app.callback(
        Output("weights-list", "children"),
        Input("list-weights-button", "n_clicks")
    )
    def list_weights(n_clicks):
        if not n_clicks:
            return ""
        try:
            weights = get_available_weights()
            return html.Ul([html.Li(weight) for weight in weights])
        except Exception as e:
            return f"Error listing weights: {str(e)}"

    # Callback for downloading weights
    @app.callback(
        Output("download-status", "children"),
        Input("download-weights-button", "n_clicks"),
        State("repo-id", "value"),
        State("model-name", "value"),
        State("huggingface-token", "value")
    )
    def download_weights(n_clicks, repo_id, model_name, huggingface_token):
        if not n_clicks:
            return ""
        try:
            success = download_model(repo_id, model_name, huggingface_token)
            return "Downloaded successfully!" if success else "Download failed."
        except Exception as e:
            return f"Error downloading weights: {str(e)}"

    @app.callback(
        Output("tracking-status", "children"),
        Input("run-tracking-button", "n_clicks"),
        State("video-path", "value"),
        State("output-folder", "value"),
        State("weight-name-tracking", "value")
    )
    def run_tracking_callback(n_clicks, video_path, output_folder, weight_name):
        """
        Callback to handle tracking functionality when the 'Run Tracking' button is clicked.

        Args:
            n_clicks (int): Number of times the button is clicked.
            video_path (str): Path to the input video.
            output_folder (str): Path to the folder where outputs will be saved.
            weight_name (str): Name of the model weight to use.

        Returns:
            str: Status message indicating success or error.
        """
        if not n_clicks:
            return ""

        # Validate inputs
        if not video_path or not os.path.exists(video_path):
            return "Error: Invalid or missing video file path."
        if not output_folder:
            return "Error: Output folder path is required."
        if not weight_name:
            return "Error: Weight name is required."

        try:
            # Run the tracking process
            message = main(output_folder, video_path, weight_name)
            return f"Tracking completed successfully! Results saved to {output_folder}"
        except FileNotFoundError:
            return "Error: Specified video file not found."
        except Exception as e:
            return f"Error during tracking: {str(e)}"

    @app.callback(
        Output("upload-status", "children"),
        Input("upload-video", "contents"),
        State("upload-video", "filename")
    )
    def upload_and_process_video(contents, filename):
        global frame_paths, current_frame_index
        if contents:
            frame_paths = []
            current_frame_index = 0

            # Decode and save the uploaded video
            content_type, content_string = contents.split(",")
            video_data = base64.b64decode(content_string)
            video_path = os.path.join(UPLOAD_FOLDER, filename)

            with open(video_path, "wb") as f:
                f.write(video_data)

            # Extract frames
            video_capture = cv2.VideoCapture(video_path)
            frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

            for i in range(frame_count):
                ret, frame = video_capture.read()
                if not ret:
                    break
                frame_path = os.path.join(FRAME_FOLDER, f"frame_{i}.jpg")
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)

            video_capture.release()
            return f"Video uploaded successfully! {len(frame_paths)} frames extracted."

        return "No video uploaded."

    @app.callback(
        [Output("frame-display", "children"),
         Output("frame-number", "children")],
        [Input("back-button", "n_clicks"),
         Input("forward-button", "n_clicks"),
         Input("frame-input", "value")],
        prevent_initial_call=True
    )
    def navigate_frames(back_clicks, forward_clicks, frame_input):
        """
        Navigate through frames using arrow buttons or manual input.

        Args:
            back_clicks (int): Number of times the back button was clicked.
            forward_clicks (int): Number of times the forward button was clicked.
            frame_input (int): The frame number entered manually.

        Returns:
            tuple: Frame display (image) and frame number text.
        """
        global current_frame_index
        ctx = dash.callback_context

        # Determine which input triggered the callback
        if not ctx.triggered or not frame_paths:
            return "No frames available.", ""
        trigger = ctx.triggered[0]["prop_id"].split(".")[0]

        # Update current_frame_index based on the triggering input
        if trigger == "back-button":
            current_frame_index = max(0, current_frame_index - 1)
        elif trigger == "forward-button":
            current_frame_index = min(len(frame_paths) - 1, current_frame_index + 1)
        elif trigger == "frame-input" and frame_input is not None:
            if 1 <= frame_input <= len(frame_paths):
                current_frame_index = frame_input - 1
            else:
                return "Invalid frame number.", f"Invalid frame number: {frame_input}"

        # Get the current frame path
        frame_path = frame_paths[current_frame_index]
        encoded_image = base64.b64encode(open(frame_path, "rb").read()).decode()
        frame_img = html.Img(src=f"data:image/jpeg;base64,{encoded_image}", style={"width": "80%"})
        return frame_img, f"Frame {current_frame_index + 1} / {len(frame_paths)}"

    @app.callback(
        Output("annotation-status", "children"),
        Input("annotate-button", "n_clicks"),
        State("annotation-type", "value"),
        State("model-name", "value"),
        State("api-key", "value")
    )

    def annotate_frame_callback(n_clicks, annotation_type, model_name, api_key):
        global current_frame_index


        # Ensure the callback is triggered by a button click
        if not n_clicks:
            return ""

        # Validate API key only after the button is clicked
        if not api_key:
            return "Error: An API key is required to perform annotation."

        # Validate frame paths
        if not frame_paths:
            return "No frames available for annotation."

        # Validate annotation type
        if not annotation_type or annotation_type not in ["auto", "manual"]:
            return "Invalid annotation type. Please select 'auto' or 'manual'."

        # Validate model name for auto annotation
        if annotation_type == "auto" and not model_name:
            return "Model name is required for auto annotation."

        try:
            # Get the current frame path
            frame_path = frame_paths[current_frame_index]

            # Call the annotate_image function

            result = annotate_image(
                annotation_type=annotation_type,
                image_path=frame_path,
                model=model_name,
                api_key=api_key
            )
            return result
        except FileNotFoundError:
            return f"Error: Frame file not found at {frame_paths[current_frame_index]}."
        except ValueError as ve:
            return f"Input Error: {ve}"
        except RuntimeError as re:
            return f"Runtime Error: {re}"
        except Exception as e:
            return f"Unexpected Error: {e}"


    @app.callback(
        Output("training-status", "children"),
        Input("train-button", "n_clicks"),
        State("epochs", "value"),
        State("batch", "value"),
        State("patience", "value"),
        State("weight-destination", "value"),
        State("api-key-train", "value")
    )
    def train_callback(n_clicks, epochs, batch, patience, weight_destination, api_key):
        """
        Callback to handle the training process when the 'Train Model' button is clicked.

        Args:
            n_clicks (int): Number of times the button is clicked.
            epochs (int): Number of epochs to train.
            batch (int): Batch size.
            patience (int): Patience for early stopping.
            weight_destination (str): Destination folder for weights.
            api_key (str): API key for Roboflow authentication.

        Returns:
            str: Status message indicating success or error.
        """
        if not n_clicks:
            return ""

        # Validate inputs
        if not epochs or not batch or not patience or not weight_destination or not api_key:
            return "Error: All fields (epochs, batch, patience, weight destination, and API key) are required."

        try:
            # Call the train_model function with the API key
            result = train_model(epochs, batch, patience, weight_destination, api_key=api_key)
            return result
        except Exception as e:
            return f"Error during training: {str(e)}"
    return app

def run():
    app = create_app()
    app.run_server(debug=True)


if __name__ == "__main__":
    run()
