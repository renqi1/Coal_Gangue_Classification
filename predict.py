import torch
import torchvision.transforms as transforms
from PIL import Image

from model import CNN

def main():
    data_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.17], std=[0.05])
    ])
    classes = ('gangue', 'coal',)
    net = CNN().cuda()
    net.load_state_dict(torch.load('./model/CNN.pth'))

    im = Image.open('coal.bmp')
    im = data_transform(im)  # [C, H, W]
    im = torch.unsqueeze(im, dim=0)  # [N, C, H, W]

    with torch.no_grad():
        outputs = net(im.cuda()).cpu()
        predict = torch.max(outputs, dim=1)[1].numpy()
    print('predict result:', classes[int(predict)])


if __name__ == '__main__':
    main()
