🩻 Bone Fracture Detection Using Deep Learning

An AI-powered image classification project that uses Deep Learning and Convolutional Neural Networks (CNN) to classify X-ray images as Fractured or Not Fractured.

The project also includes an interactive Streamlit web application where users can upload an X-ray image and receive a model prediction with confidence and probability scores.

🚀 Key Features
🩻 X-ray image upload
🧠 CNN-based image classification
🔍 Fracture / Not Fracture prediction
📊 Prediction confidence score
📈 Fracture and non-fracture probabilities
🌐 Interactive Streamlit web interface
⚡ TensorFlow/Keras model inference
🛠️ Technologies Used
Python
TensorFlow
Keras
NumPy
Pillow (PIL)
Streamlit
CNN / Deep Learning
Image Classification
🧠 Model

The trained CNN model processes X-ray images by:

Loading the uploaded X-ray image
Converting it to RGB format
Resizing the image to 224 × 224
Normalizing pixel values
Passing the image through the trained neural network
Generating a prediction probability
Displaying the final classification and confidence
📂 Dataset

The model was trained using a bone X-ray fracture dataset containing images categorized into:

Fractured
Not Fractured

The dataset was divided into training, validation, and testing sets for model development and evaluation.

🔄 Application Workflow
Upload X-Ray
     ↓
Image Preprocessing
     ↓
Resize to 224 × 224
     ↓
Normalize Pixel Values
     ↓
CNN Model
     ↓
Prediction Probability
     ↓
Fractured / Not Fractured
     ↓
Confidence & Probability Scores
📁 Project Structure
Radiology-XRay-Fracture-Detection/
│
├── app.py
├── fracture_model.keras
├── requirements.txt
├── README.md
│
├── train/
├── val/
└── test/
💻 Installation

Clone the repository:

git clone <your-github-repository-url>

Navigate to the project folder:

cd Radiology-XRay-Fracture-Detection

Create and activate a virtual environment:

python -m venv .venv

Activate it on Windows:

.venv\Scripts\activate

Install the required packages:

pip install -r requirements.txt
▶️ Run the Application

Start the Streamlit application:

streamlit run app.py

The application will open in your browser.

Upload an X-ray image and click Analyze X-Ray to generate the prediction.

📊 Prediction Output

The application provides:

Predicted class
Model confidence
Fracture probability
Not-fracture probability
📸 Application Preview

Add screenshots of your Streamlit application here.

Example:

![Application Screenshot]

⚠️ Disclaimer

This project is intended for educational and demonstration purposes only. The model output should not be considered a medical diagnosis or used as a substitute for evaluation by a qualified healthcare professional.

🔮 Future Improvements
Improve model accuracy with additional training data
Experiment with transfer learning models such as ResNet or EfficientNet
Add Grad-CAM visualization for model interpretability
Improve image preprocessing and augmentation
Add model performance metrics and evaluation charts
Deploy the application as a cloud-based Streamlit application
👩‍💻 Project

This project demonstrates the practical application of Python, Deep Learning, Computer Vision, TensorFlow, CNN, and Streamlit to an image classification problem.
