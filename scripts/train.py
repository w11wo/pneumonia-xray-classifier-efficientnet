import torch
import fastai
from fastai.vision.all import *
from fastai.vision.core import *
from fastai.callback import *
from fastai.metrics import *
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


def get_test_dls(bs: int, size: int, test_folder: str):
    """
    Returns a pair of training and testing dataloaders.

            Parameters:
                    bs (int): Batch size
                    size (int): Size of images after resizing
                    test_folder (str): Path to test folder

            Returns:
                    Dataloaders of batch size `bs`
    """
    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        get_y=parent_label,
        splitter=GrandparentSplitter(valid_name=test_folder),
        item_tfms=Resize(size),
        batch_tfms=batch_tfms,
    )
    return dblock.dataloaders(path, bs=bs, num_workers=0).cuda()


def plot_confusion_matrix(y_pred, y_true, vocab):
    """
    Plots a confusion matrix given ground truth and predictions.

            Parameters:
                    y_pred: Class predictions
                    y_true: Ground truth classes
                    vocab: Vocabulary of classes

            Returns:
                    None
    """
    y_pred = y_pred.argmax(dim=-1)
    cm = confusion_matrix(y_true, y_pred)

    fig = plt.figure(figsize=(8, 8), dpi=60)
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    tick_marks = np.arange(len(vocab))
    plt.xticks(tick_marks, vocab, rotation=90)
    plt.yticks(tick_marks, vocab, rotation=0)

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        coeff = f"{cm[i, j]}"
        plt.text(
            j,
            i,
            coeff,
            horizontalalignment="center",
            verticalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    ax = fig.gca()
    ax.set_ylim(len(vocab) - 0.5, -0.5)

    plt.tight_layout()
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.grid(False)

    print(f"Precision: {precision_score(y_true, y_pred):.3f}")
    print(f"Recall: {recall_score(y_true, y_pred):.3f}")


path = Path("chest_xray/chest_xray")

batch_tfms = [Normalize.from_stats(*imagenet_stats), *aug_transforms()]

dls = get_dls(64, 224)

body = create_timm_body("efficientnet_b3a", pretrained=True)
nf = num_features_model(nn.Sequential(*body.children())) * (2)
head = create_head(nf, dls.c)

model = nn.Sequential(body, head)
apply_init(model[1], nn.init.kaiming_normal_)

learn = Learner(
    dls, model, loss_func=LabelSmoothingCrossEntropy(), metrics=accuracy, cbs=MixUp()
)

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    learn.model = nn.DataParallel(learn.model)

learn.fit_one_cycle(10, 6e-3, wd=0.1, cbs=SaveModelCallback(fname="best-val-loss"))
learn.save("efficientnetb3a-1")

test_dl = get_test_dls(64, 224, "test")
learn.dls = test_dl

preds, targs = learn.get_preds()
print(accuracy(preds, targs))

plot_confusion_matrix(preds, targs, dls.vocab)
