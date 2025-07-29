import streamlit as st
from openai import OpenAI
from pytesseract import pytesseract
from pdf2image import convert_from_path
import cv2
import numpy as np
import json

def ocr_from_file(uploaded):
    name = uploaded.name.lower()
    if name.endswith(".pdf"):
        pages = convert_from_path(uploaded, dpi=300)
        texts = [pytesseract.image_to_string(np.array(p)) for p in pages]
        text = "\n".join(texts)
    else:
        arr = np.frombuffer(uploaded.read(), np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        text = pytesseract.image_to_string(img)
    return text

def extract_tests_via_function_call(text, api_key):
    client = OpenAI(api_key=api_key)
    functions = [
        {
            "name": "extract_lab_tests",
            "description": "Extract lab tests from prescription text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "lab_tests": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "test_name": {"type": "string"},
                                "details": {"type": "string"}
                            },
                            "required": ["test_name"]
                        }
                    }
                },
                "required": ["lab_tests"],
                "additionalProperties": False
            }
        }
    ]
    prompt = (
        "You are given prescription text. "
        "Return a JSON with one key 'lab_tests', containing an array of objects. "
        "Each object should have 'test_name' (mandatory) and 'details' (optional)."
    )
    resp = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text}
        ],
        functions=functions,
        function_call={"name": "extract_lab_tests"}
    )
    args = resp.choices[0].message.function_call.arguments
    return json.loads(args)["lab_tests"]

def main():
    st.set_page_config(page_title="Lab Test Extractor", layout="centered")
    st.title("⚕️ RX Prescription Lab Test Extractor")

    api_key = st.text_input("OpenAI API Key", type="password")
    uploaded = st.file_uploader("Upload prescription (PDF, JPG, PNG)", type=["pdf", "jpg", "jpeg", "png"])

    if st.button("Extract"):
        if not api_key or not api_key.startswith("sk-"):
            st.error("Enter a valid API key (must start with sk-)")
            return
        if not uploaded:
            st.error("Please upload a prescription file.")
            return

        with st.spinner("Performing OCR..."):
            text = ocr_from_file(uploaded)

        st.subheader("📝 OCR Text")
        st.text_area("Extracted Text:", text, height=200)

        with st.spinner("Extracting lab tests..."):
            try:
                tests = extract_tests_via_function_call(text, api_key)
            except Exception as e:
                st.error(f"OpenAI API error: {e}")
                return

        if tests:
            st.subheader("✅ Extracted Lab Tests")
            st.table(tests)
        else:
            st.warning("No lab tests were extracted.")

if __name__ == "__main__":
    main()
