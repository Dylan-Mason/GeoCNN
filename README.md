# GeoCNN
This repository contains the **GeoGuessr Prediction Model**, a machine learning-based project designed to predict country and city names from images. The project includes a Flask web application for user interaction and a trained deep learning model for predictions.

## Description
The GeoGuessr Prediction Model uses a convolutional neural network (CNN) to predict the **country** and optionally the **city** based on an input image. Users can:
- Load images from a preexisting dataset.
- Upload their own images for prediction.

## General File Structure
Below is a summary of the key files and directories in this repository:
GeoCNN/
├── demoapp.py             # Flask backend
├── static/                # Static assets (CSS, JavaScript)
│   ├── styles.css         # Main CSS file for styling
├── templates/             # HTML templates
│   ├── index.html         # Main web page template
│   ├── result.html        # Main web page template
│   90country70city.pth    # Trained model file (stored on Google Drive)
│   Newdataset/            # Dataset images (stored on Google Drive)
│   ├── 0.png              #images
│   ├── 1.png              #images
│   ├── etc                #images
│   test.csv               # Ground truth CSV for testing
├── README.md              # Project description (this file)

## Large Files
Due to GitHub's file size limits, large files are stored externally and can be downloaded using the following link:
[Google Drive Folder](https://drive.google.com/drive/folders/1oKhepUJLK072eyGd_JhMuo5s-UIAcM-5?usp=sharing)

### Files on Google Drive:
1. `90country70city.pth`: The trained model file.
2. `Newdataset/10000.zip`: Dataset containing images used for predictions.
3. `text.csv`: Csv used for verifying predictions.

## How to Run the Project
1. Clone this repository:
   git clone https://github.com/your-username/GeoCNN.git

