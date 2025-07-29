import streamlit as st
import openai
from PIL import Image
import io
import base64
import os

from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    openai.api_key = st.text_input("Enter your OpenAI API Key", type="password")

st.set_page_config(page_title="Vision Poet with Object Detection", layout="centered")
st.title("🎯 Object Aware Vision Poet")
st.caption("Capture a photo — we'll detect objects and write a poem about them.")

# --- Capture from Webcam ---
img_file = st.camera_input("📸 Take a picture")

if img_file and openai.api_key:
    st.image(img_file, caption="Your Photo", use_column_width=True)

    img = Image.open(img_file)
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode()

    with st.spinner("🔍 Detecting objects and writing poem..."):
        try:
            # Request to GPT-4o with vision
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You're an AI that can visually analyze images and write poetic descriptions.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "First, list up to 4 distinct recognizable objects in this image.\n"
                                    "Then, write a short poetic description inspired by those objects."
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                            },
                        ],
                    },
                ],
                max_tokens=400,
            )

            result = response.choices[0].message.content
            st.markdown("### 🎯 Detected Objects & Poem")
            st.write(result)

        except Exception as e:
            st.error(f"❌ OpenAI Vision API Error: {e}")
