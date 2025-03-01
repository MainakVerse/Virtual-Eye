# DeepLensX
DeepLensX is a Streamlit app that integrates MobileNetV2 and a CIFAR-10 model for image classification. Users can upload images and receive predictions with confidence scores from either model. It features a sleek navigation bar for easy switching and real-time results, which is ideal for learning and practical use.

## Key Features

- **Dual Model Support**:
  - **MobileNetV2 (ImageNet)**: Recognizes 1,000 different classes from the ImageNet dataset, including everyday objects, animals, and vehicles.
  - **Custom CIFAR-10 Model**: Specializes in classifying images into one of ten specific categories such as airplanes, automobiles, and birds.

- **Intuitive Interface**:
  - **Navigation Bar**: Seamlessly switch between MobileNetV2 and CIFAR-10 models using a sleek sidebar menu.
  - **Real-Time Classification**: Upload an image to receive immediate predictions with confidence scores.

- **Educational and Practical Use**:
  - Ideal for learning about deep learning models and their performance.
  - Useful for practical applications where image classification is needed.

## Getting Started

### Prerequisites

- Python 3.7 or later
- A web browser

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/JayRathod341997/DeepLensX.git
   cd DeepLensX
2. **Create and activate a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows use `venv\Scripts\activate`
3. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
4. **Start the Streamlit app**:
    ```bash
    streamlit run app.py
5. **Open the app**: 
    The app will open in your default web browser. If not, navigate to http://localhost:8501


### Usage
  1. Use the navigation bar to select either the MobileNetV2 or CIFAR-10 model.
  2. Upload an image file (JPG or PNG).
  3. View the classification results and confidence scores.

### Contributing
  Feel free to fork the repository, open issues, or submit pull requests to contribute to the project.

### Acknowledgements
  - Streamlit
  - TensorFlow
