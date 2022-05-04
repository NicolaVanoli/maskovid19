# maskovid19

### MOTIVATION & DESCRIPTION
In such a particular and unique historical moment as the one we are experiencing, characterized by the spread of the Covid-19 virus, instead of sitting idle and let negative thoughts grow day by day, we decided to develop a machine learning tool able to identify people who wear a mask from those who do not wear it.
This project is inspired by the works of:
1.	The perceptron (https://github.com/aieml/face-mask-detection-keras)
2.	Prajna Bhandary (https://github.com/prajnasb/observations)
3.	Adrian Rosebrock (https://www.pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning/)
Maskovid-19 uses two deep neural networks for different tasks: the first one is a pre-trained CNN able to detect faces when a picture is given it as input and is (hopefully) not going to be trained by us for better results. Once the faces are recognized by this neural network, the input image is cropped and only the part of the picture containing the face/faces is returned as output.
![Immagine1](https://user-images.githubusercontent.com/59766551/166697723-f9d8f19b-b428-4425-a6b2-d2324c307e2f.jpg) --->
![Immagine2](https://user-images.githubusercontent.com/59766551/166697781-764d8045-c025-4fba-8327-4efcd75a8770.jpg)


Once the picture has been cropped the second CNN is asked to classify the face as a mask or not-mask user.
In order to achieve these results we train a pre-trained CNN with a specific dataset containing only faces of people wearing (and not wearing) face masks. In particular, we use transfer learning and add few new layers to the original CNN. By doing that, we basically ask the neural network to add 2 new classes (MASK / NOT-MASK).
Once the classification is completed, the original picture is restored, the face is highlighted and a RED/GREEN square tells us if the user is wearing/not wearing a face-mask.
![Immagine3](https://user-images.githubusercontent.com/59766551/166698226-b4bc01a7-0c50-4e51-ab42-e81992c82e71.png)
### DURATION & DELIVERABLES
The goal of this project is trying to have a robust and useful tool that could be used in several different situations. The dream would be to succeed in developing a real-time mask detector for videos (security cams in parks, stores, etc.). This last task, however, will be only faced if we will have enough time.
The hardest parts for us, at this stage, is being able to produce a solid training set (since online there are only artificial ones) and train the pre-existing CNN (we will most likely use “MobileNetV2”).
Training process will be done on: CPU – Ryzen 7 3700x; GPU – Nvidia 2070 Super; 32GB RAM. We expect to have it ready for the 15/06 exam session.
