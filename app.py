import gradio as gr
import numpy as np
from PIL import Image
import torch

import os
import shutil
import config
from models.yolo import YOLOv3
from utils.data import PascalDataModule
from utils.loss import YoloLoss
from utils.gradcam import generate_gradcam
from utils.utils import generate_result
from markdown import model_stats, data_stats

datamodule = PascalDataModule(
    train_csv_path=f"{config.DATASET}/train.csv",
    test_csv_path=f"{config.DATASET}/test.csv",
    batch_size=config.BATCH_SIZE,
    shuffle=config.SHUFFLE,
    num_workers=os.cpu_count() - 1,
)
datamodule.setup()


class FilterModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.yolo = YOLOv3.load_from_checkpoint(
            "model.ckpt",
            in_channels=3,
            num_classes=config.NUM_CLASSES,
            epochs=config.NUM_EPOCHS,
            loss_fn=YoloLoss,
            datamodule=datamodule,
            learning_rate=config.LEARNING_RATE,
            maxlr=config.LEARNING_RATE,
            scheduler_steps=len(datamodule.train_dataloader()),
            device_count=config.NUM_WORKERS,
        )
        self.yolo = self.yolo.to("cpu")

    def forward(self, x):
        x = self.yolo(x)
        return x[-1]


model = FilterModel()

prediction_image = None


def upload_file(files):
    file_paths = [file.name for file in files]
    return file_paths


def read_image(path):
    img = Image.open(path)
    img.load()
    data = np.asarray(img, dtype="uint8")
    return data


def sample_images():
    all_imgs = os.listdir(config.IMG_DIR)
    rand_inds = np.random.random_integers(0, len(all_imgs), 10).tolist()
    images = [f"{config.IMG_DIR}/{all_imgs[ind]}" for ind in rand_inds]
    return images


def get_gradcam_images(gradcam_layer, images, gradcam_opacity):
    gradcam_images = []
    target_layers = [model.yolo.layers[int(gradcam_layer)]]
    gradcam_images = generate_gradcam(
        model=model,
        target_layers=target_layers,
        images=images,
        use_cuda=False,
        transparency=gradcam_opacity,
    )
    return gradcam_images


def show_hide_gradcam(status):
    if not status:
        return [gr.update(visible=False) for i in range(3)]
    return [gr.update(visible=True) for i in range(3)]


def set_prediction_image(evt: gr.SelectData, gallery):
    global prediction_image
    if isinstance(gallery[evt.index], dict):
        prediction_image = gallery[evt.index]["name"]
    else:
        prediction_image = gallery[evt.index][0]["name"]


def predict(is_gradcam, gradcam_layer, gradcam_opacity):
    gradcam_images = None
    img = read_image(prediction_image)
    image_transformed = config.test_transforms(image=img, bboxes=[])["image"]
    if is_gradcam:
        images = [image_transformed]
        gradcam_images = get_gradcam_images(gradcam_layer, images, gradcam_opacity)

    data = image_transformed.unsqueeze(0)

    if not os.path.exists("output"):
        os.mkdir("output")
    else:
        shutil.rmtree("output")
        os.mkdir("output")
    generate_result(
        model=model.yolo,
        data=data,
        thresh=0.6,
        iou_thresh=0.5,
        anchors=model.yolo.scaled_anchors,
    )
    result_images = os.listdir("output")
    result_images = [
        f"output/{file}" for file in result_images if file.endswith(".png")
    ]
    return {
        output: gr.update(value=result_images[0]),
        gradcam_output: gr.update(value=gradcam_images[0]),
    }


with gr.Blocks() as app:
    gr.Markdown("## ERA Session13 - PASCAL-VOC Object Detection with YoloV3")
    with gr.Row():
        with gr.Column():
            with gr.Box():
                is_gradcam = gr.Checkbox(
                    label="GradCAM Images",
                    info="Display GradCAM images?",
                )
                gradcam_layer = gr.Dropdown(
                    choices=list(range(len(model.yolo.layers))),
                    label="Select the layer",
                    info="Please select the layer for which the GradCAM is required",
                    interactive=True,
                    visible=False,
                )
                gradcam_opacity = gr.Slider(
                    minimum=0,
                    maximum=1,
                    value=0.6,
                    label="Opacity",
                    info="Opacity of GradCAM output",
                    interactive=True,
                    visible=False,
                )

                is_gradcam.input(
                    show_hide_gradcam,
                    inputs=[is_gradcam],
                    outputs=[gradcam_layer, gradcam_opacity],
                )
            with gr.Box():
                # file_output = gr.File(file_types=["image"])
                with gr.Group():
                    upload_gallery = gr.Gallery(
                        value=None,
                        label="Uploaded images",
                        show_label=False,
                        elem_id="gallery_upload",
                        columns=5,
                        rows=2,
                        height="auto",
                        object_fit="contain",
                    )
                    upload_button = gr.UploadButton(
                        "Click to Upload images",
                        file_types=["image"],
                        file_count="multiple",
                    )
                    upload_button.upload(upload_file, upload_button, upload_gallery)

                with gr.Group():
                    sample_gallery = gr.Gallery(
                        value=sample_images,
                        label="Sample images",
                        show_label=True,
                        elem_id="gallery_sample",
                        columns=5,
                        rows=2,
                        height="auto",
                        object_fit="contain",
                    )

                upload_gallery.select(set_prediction_image, inputs=[upload_gallery])
                sample_gallery.select(set_prediction_image, inputs=[sample_gallery])

            run_btn = gr.Button()
        with gr.Column():
            with gr.Box():
                output = gr.Image(value=None, label="Model Result")
            with gr.Box():
                gradcam_output = gr.Image(value=None, label="GradCAM Image")

        run_btn.click(
            predict,
            inputs=[
                is_gradcam,
                gradcam_layer,
                gradcam_opacity,
            ],
            outputs=[output, gradcam_output],
        )

    with gr.Row():
        with gr.Box():
            with gr.Row():
                with gr.Column():
                    with gr.Box():
                        gr.Markdown(model_stats)
                with gr.Column():
                    with gr.Box():
                        gr.Markdown(data_stats)

app.launch()
