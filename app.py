import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

# --- PAGE CONFIGURATION ---
# Sets the page title, icon, and layout for a more professional look.
st.set_page_config(
    page_title="Duality AI - Safety Auditor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- THEME AND STYLING ---
# Injects custom CSS for a dark, space-themed UI.
st.markdown("""
<style>
    /* Dark theme for the main content and sidebar */
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .stApp {
         background-color: #0E1117;
    }
    /* Style for titles and headers */
    h1, h2, h3 {
        color: #00A6FF; /* A bright blue for a futuristic feel */
    }
    /* Style for buttons and widgets */
    .stButton>button {
        color: #FAFAFA;
        background-color: #00A6FF;
        border: none;
    }
</style>
""", unsafe_allow_html=True)


# --- MODEL LOADING ---
# Caches the model to prevent reloading on every interaction, making the app much faster.
@st.cache_resource
def load_yolo_model():
    """Loads the YOLOv8 model from the specified path."""
    try:
        model = YOLO('best.pt')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_yolo_model()

# --- SIDEBAR ---
# All user controls are placed in the sidebar for a clean layout.
with st.sidebar:
    st.header("🚀 Mission Control")
    st.write("Configure the safety scan parameters.")
    
    # Confidence Threshold Slider
    confidence = st.slider(
        "Detection Confidence Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.5, 
        step=0.05
    )
    
    # Input source selection
    source_option = st.radio(
        "Select Input Source:",
        ["Upload an Image", "Use Webcam"]
    )
    
    source_img = None
    if source_option == "Upload an Image":
        uploaded_file = st.file_uploader(
            "Upload an image of the sector...", 
            type=["jpg", "jpeg", "png"]
        )
        if uploaded_file:
            source_img = Image.open(uploaded_file)
    else:
        camera_input = st.camera_input("Position camera and take snapshot...")
        if camera_input:
            source_img = Image.open(camera_input)

# --- MAIN PAGE ---
st.title("🛰 Duality AI Safety Equipment Auditor")
st.write("This application uses a YOLOv8 model to perform a real-time safety audit of a designated sector.")

# Two-column layout for side-by-side comparison
col1, col2 = st.columns(2)

if source_img:
    with col1:
        st.write("### Original Sector Image")
        st.image(source_img, caption="Image to be analyzed.", use_column_width=True)

    if model:
        # Perform prediction only if an image has been uploaded/captured
        with st.spinner("Analyzing telemetry... scanning for safety equipment..."):
            # Convert PIL image to OpenCV format (numpy array)
            img_cv = np.array(source_img)
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

            # Run the model
            results = model.predict(img_cv, conf=confidence)
            
            # Draw the bounding boxes on the image
            result_plotted = results[0].plot()
            
            # Convert the result back to RGB for displaying in Streamlit
            result_rgb = cv2.cvtColor(result_plotted, cv2.COLOR_BGR2RGB)

            with col2:
                st.write("### Audit Results")
                st.image(result_rgb, caption='Detected Safety Equipment.', use_column_width=True)
                
                # List the detected objects
                st.write("#### Inventory of Detected Items:")
                names = model.names
                detected_objects = {}
                if results[0].boxes:
                    for box in results[0].boxes:
                        class_name = names[int(box.cls)]
                        detected_objects[class_name] = detected_objects.get(class_name, 0) + 1
                    
                    if detected_objects:
                        for obj, count in detected_objects.items():
                            st.success(f"✔ Found {count}x {obj}")
                    else:
                        st.warning("No safety equipment detected above the confidence threshold.")
                else:
                    st.info("Scan complete. No objects detected.")