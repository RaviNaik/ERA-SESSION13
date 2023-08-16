# ERA-SESSION13 YoloV3 with Pytorch Lightning & Gradio

HF Link:

### Tasks:
1. Move the code to PytorchLightning
2. Train the model to reach such that all of these are true:
    - Class accuracy is more than 75%
    - No Obj accuracy of more than 95%
    - Object Accuracy of more than 70% (assuming you had to reduce the kernel numbers, else 80/98/78)
    - Ideally trailed till 40 epochs
3. Add these training features:
    - Add multi-resolution training - the code shared trains only on one resolution 416
    - Add Implement Mosaic Augmentation only 75% of the times
    - Train on float16
    - GradCam must be implemented.
4. Things that are allowed due to HW constraints:
    - Change of batch size
    - Change of resolution
    - Change of OCP parameters
5. Once done:
    - Move the app to HuggingFace Spaces
    - Allow custom upload of images
    - Share some samples from the existing dataset
    - Show the GradCAM output for the image that the user uploads as well as for the samples.
6. Mention things like:
    - classes that your model support
    - link to the actual model
7. Assignment:
    - Share HuggingFace App Link
    - Share LightningCode Link on Github
    - Share notebook link (with logs) on GitHub
