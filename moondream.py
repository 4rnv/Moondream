#Make sure to run 'pip install transformers einops'

from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

model_id = "vikhyatk/moondream2"
revision = "2024-08-26"
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, revision=revision
)
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

image = Image.open('<PATH/TO/IMAGE>')
enc_image = model.encode_image(image)
print(model.answer_question(enc_image, "Describe the image in as much detail at possible.", tokenizer))