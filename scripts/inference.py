import torch
import torchvision
import fastai
from fastai.vision.all import *
from fastai.vision.core import *
from fastai.callback import *
from fastai.metrics import *
import PIL
import gradio as gr
import numpy as np
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
from timm import create_model


def get_dls(bs: int, size: int):
    """
    Returns a pair of training and validation dataloaders.

            Parameters:
                    bs (int): Batch size
                    size (int): Size of images after resizing

            Returns:
                    Dataloaders of batch size `bs`
    """
    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        get_y=parent_label,
        splitter=GrandparentSplitter(valid_name="val"),
        item_tfms=RandomResizedCrop(size, min_scale=0.75),
        batch_tfms=batch_tfms,
    )
    return dblock.dataloaders(path, bs=bs, num_workers=0).cuda()


def create_timm_body(arch: str, pretrained=True, cut=None):
    """
    Returns the body of a timm model suited to fast.ai.

            Parameters:
                    arch (str): Architecture name
                    pretrained (bool): Load pretrained weights
                    cut: Rule when splitting model

            Returns:
                    A timm model splitted according to given rules and suited for fast.ai `Learner`
    """
    model = create_model(arch, pretrained=pretrained)
    if cut is None:
        ll = list(enumerate(model.children()))
        cut = next(i for i, o in reversed(ll) if has_pool_type(o))
    if isinstance(cut, int):
        return nn.Sequential(*list(model.children())[:cut])
    elif callable(cut):
        return cut(model)
    else:
        raise NamedError("cut must be either integer or function")


def predict(input: Image):
    """
    Returns a string of predicted class name and its probability.

            Parameters:
                    input (Image): PIL Image to be predicted

            Returns:
                    res (str): Result after prediction
    """
    im = TensorImage(normalize_fn(totensor(input)))
    cat, cat_idx, probs = learn.predict(im)
    res = f"Model predicted {cat} with {max(probs):.3f} probability"
    return res


def open_image(fname, size=224):
    img = PIL.Image.open(fname).convert("RGB")
    img = img.resize((size, size))
    return img


path = Path("chest_xray/chest_xray")

batch_tfms = [Normalize.from_stats(*imagenet_stats), *aug_transforms()]

dls = get_dls(64, 224)

body = create_timm_body("efficientnet_b3a", pretrained=True)
nf = num_features_model(nn.Sequential(*body.children())) * (2)
head = create_head(nf, dls.c)

model = nn.Sequential(body, head)
apply_init(model[1], nn.init.kaiming_normal_)

learn = Learner(dls, model, loss_func=LabelSmoothingCrossEntropy(), metrics=accuracy)

learn = learn.load("best models\\efficientnetb3a-1")

totensor = torchvision.transforms.ToTensor()
normalize_fn = torchvision.transforms.Normalize(*imagenet_stats)

img = open_image(path / "train/NORMAL/IM-0115-0001.jpeg")
print(predict(img))

gr_interface = gr.Interface(
    fn=predict,
    inputs=gr.inputs.Image(shape=(224, 224)),
    outputs="text",
    title="Chest X-Ray Pneumonia Classifier",
)

gr_interface.launch(inline=False, share=True)
