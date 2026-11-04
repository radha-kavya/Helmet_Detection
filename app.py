
import streamlit as st
import torch
import cv2
import numpy as np

from PIL import Image
from torchvision.transforms.functional import to_tensor

from src.model_transfer import create_transfer_model


# Page configuration
st.set_page_config(
    page_title="Helmet Detection AI",
    page_icon="⛑️",
    layout="wide"
)

st.title("⛑️ Helmet Detection System")

st.write(
    "Detect helmets and missing helmets in construction-site images "
    "using a CNN-based Faster R-CNN model."
)


# Load trained model
@st.cache_resource
def load_model():

    model = create_transfer_model(num_classes=3)

    checkpoint = torch.load(
        "models/transfer_epoch_2.pth",
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(checkpoint)
    model.eval()

    return model


# Detection function
def detect_helmet(image):

    model = load_model()

    image_tensor = to_tensor(image)

    with torch.no_grad():
        predictions = model([image_tensor])[0]

    image_array = np.array(image).copy()

    image_array = cv2.cvtColor(
        image_array,
        cv2.COLOR_RGB2BGR
    )

    class_names = {
        1: "Helmet",
        2: "No Helmet"
    }

    count = 0

    for box, label, score in zip(
        predictions["boxes"],
        predictions["labels"],
        predictions["scores"]
    ):

        if score >= 0.50:

            x1, y1, x2, y2 = map(
                int,
                box.tolist()
            )

            class_id = label.item()

            class_name = class_names.get(
                class_id,
                "Unknown"
            )

            color = (
                (0, 255, 0)
                if class_id == 1
                else (0, 0, 255)
            )

            cv2.rectangle(
                image_array,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            cv2.putText(
                image_array,
                f"{class_name}: {score:.2f}",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

            count += 1

    result = cv2.cvtColor(
        image_array,
        cv2.COLOR_BGR2RGB
    )

    return result, count


# Choose detection method
st.subheader("Choose Detection Method")

tab1, tab2 = st.tabs([
    "Image Upload",
    "Webcam Detection"
])


# IMAGE UPLOAD
with tab1:

    st.subheader("Upload Construction Image")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        key="image_upload"
    )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        if st.button(
            "Detect Helmet",
            key="image_detect"
        ):

            with st.spinner("Detecting helmets..."):

                result, count = detect_helmet(image)

            st.subheader("Detection Results")

            st.image(
                result,
                caption="Helmet Detection Output",
                use_container_width=True
            )

            st.success(
                f"Total detections: {count}"
            )


# WEBCAM DETECTION
with tab2:

    st.subheader("Webcam Helmet Detection")

    st.write(
        "Allow camera permission and capture an image "
        "to detect helmets."
    )

    camera_image = st.camera_input(
        "Capture an image using your webcam"
    )

    if camera_image is not None:

        image = Image.open(
            camera_image
        ).convert("RGB")

        if st.button(
            "Detect from Webcam",
            key="webcam_detect"
        ):

            with st.spinner("Analyzing webcam image..."):

                result, count = detect_helmet(image)

            st.subheader("Webcam Detection Results")

            st.image(
                result,
                caption="Webcam Helmet Detection Output",
                use_container_width=True
            )

            st.success(
                f"Total detections: {count}"
            )


# Footer
st.divider()

st.caption(
    "Helmet Detection AI | PyTorch | Faster R-CNN | Streamlit"
)

