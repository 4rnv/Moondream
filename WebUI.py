import argparse
import torch
import gradio as gr
from moondream import detect_device, LATEST_REVISION, Moondream
from threading import Thread
from transformers import TextIteratorStreamer, AutoTokenizer
from PIL import ImageDraw
import re
from torchvision.transforms.v2 import Resize

theme = gr.themes.Default(primary_hue="pink").set(
    loader_color="#ff48a5",
    slider_color="#ff48a5",
    block_radius="None",
    input_radius="None",
    container_radius="None",
    body_background_fill="#222",
    body_background_fill_dark="#222",
    block_border_width=2,
    input_border_width=2,
    block_background_fill="#444",
    input_background_fill="#444",
    block_border_color="#ff48a5",
    input_border_color="#ff48a5",
    button_border_width=4,
    button_primary_background_fill="#ff48a5",
    button_secondary_background_fill="#ff48a5",
    button_primary_background_fill_dark=None,
    button_primary_background_fill_hover="#e03080",
    button_primary_background_fill_hover_dark=None,
    button_primary_border_color="#ff48a5",
    button_primary_border_color_dark="#ff48a5",
    button_primary_border_color_hover="#ff48a5",
    button_primary_border_color_hover_dark="#ff48a5",
    button_secondary_background_fill_dark=None,
    button_secondary_background_fill_hover="#e03080",
    button_secondary_background_fill_hover_dark=None,
    button_secondary_border_color="#ff48a5",
    button_secondary_border_color_dark="#ff48a5",
    button_secondary_border_color_hover="#ff48a5",
    button_secondary_border_color_hover_dark="#ff48a5",
    button_large_radius="None",
    button_small_radius="None"
)

parser = argparse.ArgumentParser()
parser.add_argument("--cpu", action="store_true")
args = parser.parse_args()

if args.cpu:
    device = torch.device("cpu")
    dtype = torch.float32
else:
    device, dtype = detect_device()
    if device != torch.device("cpu"):
        print("Using device:", device)
        print("If you run into issues, pass the `--cpu` flag to this script.")
        print()

model_id = "vikhyatk/moondream2"
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=LATEST_REVISION)
moondream = Moondream.from_pretrained(
    model_id, revision=LATEST_REVISION, torch_dtype=dtype
).to(device=device)
moondream.eval()


def answer_question(img, prompt):
    image_embeds = moondream.encode_image(img)
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
    thread = Thread(
        target=moondream.answer_question,
        kwargs={
            "image_embeds": image_embeds,
            "question": prompt,
            "tokenizer": tokenizer,
            "streamer": streamer,
        },
    )
    thread.start()

    buffer = ""
    for new_text in streamer:
        buffer += new_text
        yield buffer


def extract_floats(text):
    # Regular expression to match an array of four floating point numbers
    pattern = r"\[\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*\]"
    match = re.search(pattern, text)
    if match:
        # Extract the numbers and convert them to floats
        return [float(num) for num in match.groups()]
    return None  # Return None if no match is found


def extract_bbox(text):
    bbox = None
    if extract_floats(text) is not None:
        x1, y1, x2, y2 = extract_floats(text)
        bbox = (x1, y1, x2, y2)
    return bbox


def process_answer(img, answer):
    if extract_bbox(answer) is not None:
        x1, y1, x2, y2 = extract_bbox(answer)
        draw_image = Resize(768)(img)
        width, height = draw_image.size
        x1, x2 = int(x1 * width), int(x2 * width)
        y1, y2 = int(y1 * height), int(y2 * height)
        bbox = (x1, y1, x2, y2)
        ImageDraw.Draw(draw_image).rectangle(bbox, outline="red", width=3)
        return gr.update(visible=True, value=draw_image)

    return gr.update(visible=False, value=None)


with gr.Blocks(theme=theme) as demo:
    gr.Markdown(
        """
        # Area 51 Labs 👽
        """
    )
    with gr.Row():
        prompt = gr.Textbox(label="Input Prompt", value="Describe this image.", scale=4)
        submit = gr.Button("Submit")
    with gr.Row():
        img = gr.Image(type="pil", label="Upload an Image")
        with gr.Column():
            output = gr.Markdown(label="Response")
            ann = gr.Image(visible=False, label="Annotated Image")

    submit.click(answer_question, [img, prompt], output)
    prompt.submit(answer_question, [img, prompt], output)
    output.change(process_answer, [img, output], ann, show_progress=False)

demo.queue().launch(share=True, debug=True)
