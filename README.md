# InferFace

This project contains an end-to-end Deep Learning pipeline to classify age*, gender** and race*** from a facial photograph.
It is based on two networks:
- InceptionResNetV1 trained on the VGGFace2 data set that takes a facial photograph as input. It runs MTCNN on it
  to find the face's bounding box and creates a 512-dimensional embedding vector as output.
- A custom classifier head that takes the embedding vector as input and classifies its age, gender and race. 


## Description

A longer description of your project goes here...


## Note

This project has been set up using PyScaffold 3.2.3. For details and usage
information on PyScaffold see https://pyscaffold.org/.


## To dos
### Which expressions do I need?
Actually, all of them since it doesn't make sense to just augment the RAF-DB data set partially => that could
introduce variability in the augmented sub-groups leading to misleading results.

### ExpW EDA plan
- how is the distribution between different expression classes?

![ExpW distribution](./images/expw-expression-dist.png)

#### Annotation
It's really fucking slow b.c. finding the faces in the first place sucks ass.
Idea:
- do a pre-selection of the image names via words such as "black", "asian", "african", "chinese"