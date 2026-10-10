# Deep Learning Methodology Notes

**Viraj – CSI6209**

## Overview
I used a pretrained FaceNet model to turn each face photo into a 512-dimensional embedding, then trained a Logistic Regression classifier on those embeddings.

## Steps

1. **Preprocessing**  
   Images are resized to 160×160 and scaled using the official FaceNet standardisation.

2. **Feature extraction**  
   The pretrained `InceptionResnetV1` (trained on VGGFace2) is used in evaluation mode.  
   No gradients are calculated (`torch.no_grad()`), because we are not fine-tuning the network.

3. **Classifier**  
   A simple Logistic Regression is trained on the embeddings.  
   The main hyperparameter is `C` (regularisation strength).  
   I tuned it on the validation set only.

4. **Final settings**  
   - Pretrained model: VGGFace2  
   - Classifier: LogisticRegression  
   - C: (fill in after experiments)  
   - Batch size: 32

## Why this approach?
Training a full deep network from scratch would need far more data and time than we have.  
Using a pretrained face network (transfer learning) gives strong features quickly, and a linear classifier on top is enough for good results.