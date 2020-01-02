import torch
import torch.nn.functional as F


def extract_peak(heatmap, max_pool_ks=7, min_score=-5, max_det=100):
    all_peaks = []
    pad = int(max_pool_ks/2)
    heatmap3d = heatmap.unsqueeze(0)
    mxpool = torch.nn.MaxPool2d(max_pool_ks, stride=1, padding=pad)
    pool_out = mxpool(heatmap3d)
    pool_out.squeeze_(0)
    cond = ((torch.eq(heatmap, pool_out)) & (pool_out > min_score))
    vals = pool_out[cond]
    indexes = cond.nonzero()
    for v, idx in zip(vals, indexes):
        all_peaks += [(v.item(), idx[1].item(), idx[0].item())]
    all_peaks = sorted(all_peaks, reverse=True, key=lambda tup: tup[0])
    all_peaks = all_peaks[:max_det]
    return all_peaks



class Detector(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, n_input, n_output, up_conv=False, stride=1):
            super().__init__()
            if(up_conv):
                conv_layer = torch.nn.ConvTranspose2d
            else:
                conv_layer = torch.nn.Conv2d
                
            self.net = torch.nn.Sequential(
              conv_layer(n_input, n_output, kernel_size=3, padding=1, stride=stride, bias=False),
              torch.nn.BatchNorm2d(n_output),
              torch.nn.ReLU(),
            )
            
            # If convolution -> downsamle for residual connection else upsample for up-convolution
            self.sample = None
            if stride != 1 or n_input != n_output:
                self.sample = torch.nn.Sequential(conv_layer(n_input, n_output, 1, stride=stride),
                                                      torch.nn.BatchNorm2d(n_output))
        
        def forward(self, x):
            identity = x
            if self.sample is not None:
                identity = self.sample(x)
            return self.net(x) + identity
        
    def __init__(self, layers=[32,64], n_input_channels=3):
        super().__init__()
        
        self.L = []
        
        # Convolutions
        input_c = 3
        for l in layers:
            output_c = l
            self.L.append(self.Block(input_c, output_c))
            input_c = output_c
        
        # Up convolutions
        rev_l = []
        output_c = 3
        for l in layers:
            input_c = l
            rev_l.append(self.Block(input_c, output_c, up_conv=True))
#            rev_l.append(torch.nn.ConvTranspose2d(input_c, output_c, kernel_size=3, padding=1))
            output_c = input_c 
        rev_l.reverse()
        self.L += rev_l
        
        self.skip_connection_sample = torch.nn.Sequential(torch.nn.Conv2d(self.L[0].net[0].out_channels, input_c, 1), 
                                                          torch.nn.BatchNorm2d(input_c))
         
        self.network = torch.nn.Sequential(*self.L)
        
        
    def forward(self, x):
        l1 = self.L[0](x)
        l2 = self.L[1](l1)
        l3 = self.L[2](l2)
        if(l1.shape == l3.shape):
            l4_inp = l1+l3
        else:
            l4_inp = self.skip_connection_sample(l1)+l3
            
        l4 = self.L[3](l4_inp)
        return l4
        
    def detect(self, image):
        # 3 -> number of classes
        # get max of the 3 values for all spatial location and note the max index - that will be class id
        maxout = image.max(0)
        maxout_vals = maxout.values
        maxout_idx = maxout.indices
        all_detections = []
        all_peaks = extract_peak(maxout_vals, max_det=100)
        for peak in all_peaks:
            class_id = maxout_idx[peak[2],peak[1]]
            detection = (class_id.item(), peak[0], peak[1], peak[2])
            all_detections += [detection]
        return all_detections

    def detect_with_size(self, image):
        """
           Your code here. (extra credit)
           Implement object detection here.
           @image: 3 x H x W image
           @return: List of detections [(class_id, score cx, cy, w/2, h/2), ...],
                    return no more than 100 detections per image
           Hint: Use extract_peak here
        """
        raise NotImplementedError('Detector.detect_with_size')


def save_model(model):
    from torch import save
    from os import path
    return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'det.th'))


def load_model():
    from torch import load
    from os import path
    r = Detector()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'det.th'), map_location='cpu'))
    return r


if __name__ == '__main__':
    """
    Shows detections of your detector
    """
    from .utils import DetectionSuperTuxDataset
    dataset = DetectionSuperTuxDataset('dense_data/valid', min_size=0)
    import torchvision.transforms.functional as TF
    from pylab import show, subplots
    import matplotlib.patches as patches

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    fig, axs = subplots(3, 4)
    model = load_model().eval().to(device)
    for i, ax in enumerate(axs.flat):
        im, kart, bomb, pickup = dataset[i]
        ax.imshow(TF.to_pil_image(im), interpolation=None)
        for k in kart:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='r'))
        for k in bomb:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='g'))
        for k in pickup:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='b'))
        for c, s, cx, cy in model.detect(im.to(device)):
            ax.add_patch(patches.Circle((cx, cy), radius=max(2 + s / 2, 0.1), color='rgb'[c]))
        ax.axis('off')
    show()
