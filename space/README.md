---
title: CNN Image Classifier on Hadoop
emoji: 🎨
colorFrom: yellow
colorTo: red
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Draw a doodle and a CNN guesses what it is
---

Draw something and a small CNN guesses what it is after every stroke. The model runs in your browser with ONNX Runtime Web, so nothing you draw is uploaded.

The model is a ResNet written from scratch in PyTorch. It was trained on Quick, Draw! doodles that a Hadoop pipeline (HDFS, YARN, MapReduce) cleaned and turned into images. The code and the results are at https://github.com/adwitiyashukla/cnn-image-classifier-on-hadoop.

The drawings come from Google's Quick, Draw! dataset, which is shared under CC BY 4.0.
