import streamlit as st
import numpy as np
import cv2
from PIL import Image
import onnxruntime
import io

# Load ONNX model
def load_model(model_path):
    return onnxruntime.InferenceSession(model_path)

# Preprocess the image (resize to 128x128 and normalize)
def preprocess_image(image: Image.Image):
    image = image.resize((128, 128))  # Resize to 128x128
    image_array = np.array(image).astype(np.float32) / 255.0  # Convert to numpy array and normalize
    image_array = image_array.transpose(2, 0, 1)  # Change shape to (C, H, W)
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Run inference on the image and return the mask
def run_inference(model, image_array):
    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name
    mask = model.run([output_name], {input_name: image_array})[0]
    return mask

# Post-process the mask to resize it back to original size and convert to image
def postprocess_mask(mask, original_size):
    mask = np.squeeze(mask)  # Remove single-dimensional entries from the shape
    
    # Example for 2-channel output (background vs. foreground):
    if mask.ndim == 3 and mask.shape[0] == 2:
        # Compare the two channels and create a binary mask
        output_mask = np.where(mask[1] > mask[0], 255, 0).astype(np.uint8)
    else:
        raise ValueError(f"Unsupported mask shape: {mask.shape}")
    
    # Resize mask to original image size
    mask_resized = Image.fromarray(output_mask)
    mask_resized = mask_resized.resize(original_size, Image.NEAREST)
    return mask_resized

def main():
    # Configure page to be centered
    st.set_page_config(page_title="Image Segmentation App", layout="centered")

    # Inject custom CSS for card layout & button colors
    st.markdown(
        """
        <style>
        /* Center main content */
        .main, .block-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        /* Card container */
        .card {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 0.5rem;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            width: 400px;
            text-align: center;
            margin-top: 5rem; /* push card down from the top */
        }
        .card h2 {
            margin-bottom: 1.5rem;
        }

        /* Make the form's Submit button blue */
        div[data-testid="stForm"] button {
            background-color: #3B82F6 !important; /* Tailwind 'blue-600' */
            color: white !important;
            border: none !important;
            border-radius: 0.25rem !important;
            font-size: 1rem !important;
            padding: 0.5rem 1rem !important;
            cursor: pointer !important;
            width: 100% !important; /* full width in the form */
            margin-top: 1rem;
        }

        /* Make the download button green */
        div[data-testid="stDownloadButton"] button {
            background-color: #10B981 !important; /* Tailwind 'green-500' */
            color: white !important;
            border: none !important;
            border-radius: 0.25rem !important;
            font-size: 1rem !important;
            padding: 0.5rem 1rem !important;
            cursor: pointer !important;
            margin-top: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Begin the "card" container
    st.markdown("<h2>Image Segmentation App</h2>", unsafe_allow_html=True)

    # Create a form with a file uploader and a submit button
    with st.form("segmentation_form"):
        uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
        submit_button = st.form_submit_button("Submit")

    # Close the card container
    st.markdown('</div>', unsafe_allow_html=True)

    # If user submits and an image is uploaded, run inference
    if submit_button and uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        original_size = image.size
        
        # Preprocess and run model inference
        input_tensor = preprocess_image(image)
        session = load_model("UNet_Based_Model.onnx")  # Update with your model path
        mask = run_inference(session, input_tensor)
        
        # Post-process the mask
        mask = postprocess_mask(mask, original_size)
        
        # Provide a download button for the mask
        buf = io.BytesIO()
        mask.save(buf, format="PNG")
        byte_im = buf.getvalue()
        
        st.download_button(
            label="Download Mask",
            data=byte_im,
            file_name="mask.png",
            mime="image/png"
        )

if __name__ == "__main__":
    main()
