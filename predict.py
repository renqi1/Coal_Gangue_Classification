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
    # 定义类别
    classes = ('gangue', 'coal',)
    # 构建模型, 这里图像小网络小，没啥运算量，可以直接用CPU：model = CNN()
    # 如果你以后用transformer之类的模型就别用CPU了，巨慢
    model = CNN().cuda()
    # 加载权重
    model.load_state_dict(torch.load('./model/CNN.pth'))
    # 加载图片
    im = Image.open('coal.bmp')
    im = data_transform(im)  # [C, H, W]
    # 增加batch size维度
    im = torch.unsqueeze(im, dim=0)  # [B, C, H, W]

    model.eval()
    with torch.no_grad():
        # 图像先放到GPU运算完后再取回CPU
        outputs = model(im.cuda()).cpu()
        # 取预测结果概率最大值的索引，再将tensor转为numpy
        predict = torch.max(outputs, dim=1)[1].numpy()
    # 根据索引查询字典对应类别
    print('predict result:', classes[int(predict)])


if __name__ == '__main__':
    main()
