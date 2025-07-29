import streamlit as st
import openai
import base64
import os

# -------------------
# SETUP OPENAI KEY
# -------------------
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    openai.api_key = st.text_input("Enter your OpenAI API Key", type="password")


st.set_page_config(page_title="Mandala Art Generator 🎨", layout="centered")
st.title("🌀 Mandala Art Generator")
st.caption("Describe your dream mandala and let DALL·E 3 generate it for you.")

# -------------
# INPUT SECTION
# -------------
prompt = st.text_input("🎨 Mandala Description", placeholder="e.g. Intricate floral mandala with cosmic background, vibrant colors")

size = st.selectbox("🖼️ Image Size", ["1024x1024", "1024x1792", "1792x1024"], index=0)

if st.button("✨ Generate Mandala"):
    if not openai.api_key or openai.api_key.strip() == "":
        st.error("OpenAI API key is missing.")
    elif not prompt.strip():
        st.warning("Please enter a mandala description.")
    else:
        with st.spinner("Generating your mandala..."):
            try:
                response = openai.images.generate(
                    model="dall-e-3",
                    prompt=prompt,
                    size=size,
                    quality="standard",
                    n=1
                )
                image_url = response.data[0].url
                st.image(image_url, caption="Your Mandala Art", use_column_width=True)
                st.success("Mandala generated successfully!")

                # Download option
                img_data = f'<a href="{image_url}" download="mandala.png">📥 Download Mandala</a>'
                st.markdown(img_data, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error: {str(e)}")
