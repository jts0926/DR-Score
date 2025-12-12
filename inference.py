import pydicom
from PIL import Image
import einops as ei
import matplotlib.pyplot as plt
import torchvision
import torch
import seaborn as sns
import pandas as pd
import glob
import os
import numpy as np
import cv2
from tqdm import tqdm


def inference_Xray(net, df_train, df_val, tsfm, id, side, show_img=True, print_pred=True):
    df = pd.concat([df_train, df_val], axis=0)
    fname = df[(df.ID == id) & (df.SIDE == side)]['PATH'].values[0]
    y_true = df[(df.ID == id) & (df.SIDE == side)]['event'].values[0]

    img = Image.open(fname).convert('RGB')
    img = torchvision.transforms.functional.rgb_to_grayscale(img)
    h, w = np.array(img).shape
    img_ori_shape = [h, w]

    print('2D Image shape:', img_ori_shape)
    img = tsfm(img)  # transform to 2D tensor list [1, 224, 224]
    img = torch.unsqueeze(img, dim=1)
    img = img.to('cuda')
    print(img.shape)

    net.to('cuda:0')
    net.eval()

    y_pred = net(img).detach().cpu().numpy()
    attention = net.Att
    # If attention has a shape like [1, 1, 25], squeeze the second dimension
    if attention.dim() == 3 and attention.size(1) == 1:
        attention = attention.squeeze(1)

    #print(attention.shape)
    patch_num1d = np.sqrt(attention.shape[1]).astype(int)
    patch_size1d = img_ori_shape[0]//patch_num1d
    attention_reshape = ei.rearrange(attention, 'b (k h) -> b k h', k=patch_num1d)
    
    if print_pred:
        print('2D Image shape:', img_ori_shape)
        print('Event: {} Risk: {}'.format(y_true, y_pred))

    img = img[0].cpu()
    att_map = attention_reshape.cpu()

    att_map = torch.repeat_interleave(att_map, patch_size1d, dim=2)
    att_map = torch.repeat_interleave(att_map, patch_size1d, dim=1)

    shape_trsf = torchvision.transforms.Resize(img_ori_shape)

    att_map = shape_trsf(att_map)
    att_map = torch.squeeze(att_map, dim=0).detach().cpu().numpy()

    if show_img:
        img = shape_trsf(img)
        #img = torch.squeeze(img, dim=0)
        #img = torch.permute(img, (1, 2, 0)).detach().cpu().numpy()
        img = torch.squeeze(img, dim=0).detach().cpu().numpy()
        img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        img = img.astype(np.uint8)

        plt.imshow(att_map, cmap="viridis")
        plt.show()
        plt.imshow(img, cmap=plt.cm.binary)
        plt.show()
    return att_map, y_pred

def inference_Xray_to_csv(net, df_train, df_val, tsfm, csv_path, print_pred=True):
    df = pd.concat([df_train, df_val], axis=0)
    net.to('cuda:0')
    net.eval()

    # Create a DataFrame to store attention maps along with ID, SIDE, and y_pred
    data_for_csv = pd.DataFrame()

    for _, row in df.iterrows():
        id = row['ID']
        side = row['SIDE']
        fname = row['PATH']

        img = Image.open(fname).convert('RGB')
        img = torchvision.transforms.functional.rgb_to_grayscale(img)
        img = tsfm(img)  # transform to 2D tensor list [1, 224, 224]
        img = torch.unsqueeze(img, dim=1)
        img = img.to('cuda')

        with torch.no_grad():
            y_pred = net(img).detach().cpu().numpy()
            attention = net.Att
            if attention.dim() == 3 and attention.size(1) == 1:
                attention = attention.squeeze(1)

        # Flatten and convert attention map to numpy array
        attention_flat = attention.cpu().numpy().flatten()

        # Create a row for the CSV file
        row_data = {'ID': id, 'SIDE': side, 'y_pred': y_pred[0]}
        for i, att_value in enumerate(attention_flat):
            row_data[f'att_{i}'] = att_value

        # Append to DataFrame
        data_for_csv = data_for_csv.append(row_data, ignore_index=True)

    # Write the data to a CSV file
    data_for_csv.to_csv(csv_path, index=False)

    print(f'Attention maps along with ID, SIDE, and y_pred written to {csv_path}')


if __name__ == "__main__":
    att_map, img, img_o_shape = Inference(net, df_val, valid_trsfs, ID='Mako 001')
    gen_heatmap(att_map, img, 16, img_o_shape)
