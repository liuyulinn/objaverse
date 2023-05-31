import torch
def get_focal_from_fovy(x_fovy, y_fovy, width, heigh):
    x_focal = 0.5 * width / torch.tan(x_fovy / 2) 