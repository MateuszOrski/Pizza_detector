import gradio as gr
#importing model
from backend_function import is_black_or_white

# input model, interface for putting image, iterface for model output, example images for fast uploading and running model
demo = gr.Interface(is_black_or_white, gr.Image(), gr.Text(), examples=[["dark_themed_image.jpg"], ["white_themed_image.jpg"], ["romania_flag.jpg"]])

if __name__ == "__main__":
    demo.launch()