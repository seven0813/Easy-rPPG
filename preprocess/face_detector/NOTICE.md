# Third-party notice

The YOLO5Face implementation, model configuration, checkpoint, and Haar
Cascade asset in this directory were migrated from
[`seven0813/My-rPPG-Toolbox`](https://github.com/seven0813/My-rPPG-Toolbox)
at commit `6b76ba6ce4d2b5516adf42ea6d689a5d7af5db72` for compatibility
with its face-cropping pipeline.

The Toolbox/Y5F files remain subject to the Responsible AI License included
in `LICENSE-RAIL.txt`; the Easy-rPPG root MIT license does not replace those
terms. The checkpoint is `ckpts/Y5sF_WFRGB.pth`, and the original detector
was trained on WiderFace as documented by rPPG-Toolbox. The Haar Cascade XML
retains its embedded Intel Open Source Computer Vision Library license notice.

Only package imports and checkpoint-loading compatibility were adapted for
use inside Easy-rPPG. Detection thresholds, model structure, preprocessing,
NMS, and face-selection behavior are preserved.
