# License Plate Detection and Recognition

## Overview
License Plate Detection and Recognition is a project aimed at detecting and extracting license plate numbers from images using TensorFlow Object Detection API and EasyOCR. The project utilizes a pre-trained object detection model, specifically "ssd_mobilenet_v2_fpn_keras", to identify license plates in images. Subsequently, an optical character recognition (OCR) model, "easyocr" in this example, is used to extract the text from the detected license plates. Finally, only the license plate numbers are extracted from the OCR results, disregarding any other text such as state names.

## Problem Statement
The accurate detection and recognition of license plates from images are essential for various applications, including traffic management, parking enforcement, and law enforcement. This project addresses this challenge by developing an end-to-end pipeline to detect license plates and extract their numbers reliably.

## Dataset
The dataset comprises images containing vehicles with visible license plates. These images are annotated with bounding boxes around the license plates to facilitate training the object detection model.

## Models Used
- **TensorFlow Object Detection API**: The project utilizes the TensorFlow Object Detection API to train an object detection model, "ssd_mobilenet_v2_fpn_keras", for detecting license plates in images.
- **EasyOCR**: EasyOCR is employed for optical character recognition, specifically to extract text from the detected license plates.

## Technologies Used
- TensorFlow
- EasyOCR
- Python
- Jupyter Notebook

## Learning Objectives
The primary objectives of this project include:
- Understanding and implementing object detection using TensorFlow Object Detection API.
- Utilizing optical character recognition techniques for text extraction.
- Integrating multiple deep learning models into an end-to-end pipeline.

## Approach and Methodology
The project follows these steps:
1. **Object Detection**: Training an object detection model to detect license plates in images using TensorFlow Object Detection API.
2. **Text Extraction**: Utilizing EasyOCR to extract text from the detected license plates.
3. **Post-processing**: Filtering out and extracting only the license plate numbers from the OCR results.

## Model Performance
The performance of the object detection model is evaluated using standard metrics such as mean Average Precision (mAP). Additionally, the accuracy of license plate number extraction is assessed to ensure reliable recognition.

## Impact
Accurate license plate detection and recognition can have significant implications for various industries, including:
- Law enforcement: Automated identification of vehicles for surveillance and security purposes.
- Transportation management: Monitoring traffic flow and parking enforcement.
- Customer service: Enhancing user experience in applications such as toll collection and parking systems.

## Future Enhancements
Future enhancements and experiments to improve model performance and application include:
- Fine-tuning object detection models for better accuracy in license plate detection.
- Training custom OCR models to handle specific fonts or challenging conditions.
- Integration with real-time video streams for dynamic license plate recognition applications.


## License
This project is licensed under the [MIT License](LICENSE).
