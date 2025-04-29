# 🚀 Project Title: Motion Detection

## 👥 Team Members:

- Manish :2401730059
- Abhay :2401730060
- Chirag : 2401730107
- Pratyush : 2401730114

## 📝 Short Project Description:

This project focuses on detecting violent actions in video footage using deep learning techniques. The goal is to automatically analyze video content and identify instances of violence, which can be useful for surveillance systems, public safety monitoring, and smart security applications. The model processes video frames and classifies them as either violent or non-violent, enabling real-time or batch-based analysis of video streams.



🔍 Key Features:

Frame extraction and preprocessing from video input



Deep learning-based classification model (e.g., CNN, LSTM, or 3D CNN)



Visualization of predictions and detected violent segments



Support for real-time and offline video analysis



🚀 Technologies Used:

Python, OpenCV



TensorFlow / PyTorch



NumPy, Matplotlib



Streamlit / Flask (for optional web interface)📽️

&#x20;Link to Video Explanation:



## 🛠️ Steps to Run/Execute the Project:

🚀 Steps to Execute the Violence Detection Project

1\. Clone the Repository

Clone this repository to your local machine.



bash

Copy

Edit

git clone https\://github.com/Manish-Dagar/mini-project.git

cd violence-detection

2\. Set Up the Environment

Create a virtual environment and activate it:



Using venv:



bash

Copy

Edit

python3 -m venv venv

source venv/bin/activate  # For Linux/MacOS

venv\Scripts\activate     # For Windows

Using conda:



bash

Copy

Edit

conda create --name violence-detection python=3.8

conda activate violence-detection

3\. Install Dependencies

Install the required Python libraries and dependencies by running:



bash

Copy

Edit

pip install -r requirements.txt

4\. Download the Pre-trained Model Weights

If the model weights (ModelWeights.weights.h5) are not provided in the repo, you can manually download them and place them in the project directory. Alternatively, you can train the model by running the Jupyter notebook pp.ipynb.



5\. Run the Jupyter Notebook (pp.ipynb) to Train the Model

Open the pp.ipynb Jupyter notebook.



Follow the instructions inside the notebook to load and process the dataset, build the model, and start the training process.



Once training is complete, the model weights will be saved as ModelWeights.weights.h5.



6\. Run the Live Violence Detection Script (live.py)

This file allows you to use a webcam for real-time violence detection. After loading the model weights, it captures live video, processes each frame, and displays whether the scene is classified as "Violence" or "Non-Violence."



Make sure your webcam is connected and functioning.



Run the live.py script:



bash

Copy

Edit

python live.py

The live detection will display on your webcam feed with the respective classification (Violence or Non-Violence) and the confidence level.



Press 'q' to stop the live video feed.



7\. (Optional) Adjust Parameters

If you want to change any parameters, such as the model architecture or training settings, modify the respective code in pp.ipynb.



You can adjust the frame buffer size in live.py or tweak the video capture settings for different resolutions or sources.

