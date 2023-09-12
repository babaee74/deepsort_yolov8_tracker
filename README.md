# This repository is an implementation of a human Tracker with DeepSort and yolov8-pose

Few days ago, a colleague of mine [talked about this ] (
https://www.linkedin.com/posts/activity-7105248139242016769-uNk4?utm_source=share&utm_medium=member_android) and I decided to replicate their software. I will improve upon this repository and will add diffeerent models using **Pytorch**.


## How To Run:
1- First clone the DeepSORT repository because we needed to make few changes in order to be compatible with tensorflow2 and also change linear_assignment of **sklearn** to scipy.
```
git clone https://github.com/babaee74/deep_sort.git
```
2- install requirements
```
pip install -r requirements.txt
```
3- download pretrain DeepSORT feature model from [this link] (https://drive.google.com/drive/folders/18fKzfqnqhqW3s9zwsCbnVJ5XF2JFeqMp)
4- modify detect.py and change **ENC_MODEL_PATH** and **video_path** parameters based on your needs
