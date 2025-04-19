# 🎙️ Cognitive Speech Analysis API

This FastAPI-based application analyzes speech audio files to assess potential cognitive anomalies based on clinical speech features such as **pause ratio**, **lexical diversity**, and **pitch variability**.

---

## 🔧 Features

- Upload `.wav` audio files for real-time analysis
- Preprocesses audio using ASR (Automatic Speech Recognition)
- Extracts linguistic and paralinguistic speech features
- Analyzes anomalies using machine learning
- Returns clinically relevant metrics and risk assessment in JSON format



Hosted URL : http://ec2-13-232-222-184.ap-south-1.compute.amazonaws.com/

Due to the size of the model and the server, the response time of the api is quite high ranging around 2-3 minutes for an average response.

WE shall use the route : http://ec2-13-232-222-184.ap-south-1.compute.amazonaws.com/analyze/ with a post request and an audio file in the body, screenshot attached:
![image](https://github.com/user-attachments/assets/5373b6a2-d13e-49b1-8848-481022845441)


I also tested it using the Inbuilt swagger functionality of the Fastapi:
![ss](https://github.com/user-attachments/assets/d40ad304-d5c0-494a-b5dc-1f4ceb81c3b7)





