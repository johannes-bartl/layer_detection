import yaml
import matplotlib.pyplot as plt
import ast

def plot_grid(img_batch,gt_batch):
    fig, ax = plt.subplots(ncols=4, nrows=4, )
    for a, img in zip(ax.flatten(), img_batch):
        a.imshow(img)
    plt.axis('off')
    plt.show()

    fig, ax = plt.subplots(ncols=4, nrows=4, )
    for a, gt in zip(ax.flatten(), gt_batch):
        a.imshow(gt)
    plt.axis('off')
    plt.show()



def parse_dict(d):
    for key, value in d.items():
        if isinstance(value, dict):
            d[key] = parse_dict(value)
        elif isinstance(value, str):
            try:
                d[key] = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                pass
    return d

def parse_cfg(path):
    with open(path, 'r') as stream:
        cfg = yaml.safe_load(stream)
    return parse_dict(cfg)