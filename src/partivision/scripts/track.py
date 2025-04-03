from partivision.inference import InferencePipeline, WeightManager, Configs

VIDEO_PATH = ""

model_name = "sep23.pt"
framerate = 1
window_width = 100
scaling_factor = 3
um_per_pixel = .74
output_folder = Configs.temp_path


WeightManager.select_current_model(model_name)
model = WeightManager.get_model()

inference_pipeline = InferencePipeline(model, framerate, window_width, scaling_factor, um_per_pixel, output_folder)
inference_pipeline.process_video(VIDEO_PATH)
