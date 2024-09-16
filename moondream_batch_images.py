from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

model_id = "vikhyatk/moondream2"
revision = "2024-08-26"
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, revision=revision
)
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

answers = model.batch_answer(
    images=[Image.open('/content/drive/MyDrive/1720934164227801.jpg'), Image.open('/content/drive/MyDrive/HBR.jpg')],
    prompts=["Describe this image.", "Are there people in this image?"],
    tokenizer=tokenizer,
)
print(answers)