#导入工具包
import numpy
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import dataset
from torchvision import transforms, datasets
from torch.autograd import Variable
from model import CNN



def main():
    # 数据转换
    data_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.17], std=[0.05])
    ])

    # 读取数据，随机分为训练集和验证集
    full_dataset = datasets.ImageFolder('./样本', transform=data_transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    print('训练样本数量：', train_size, '验证集样本数量：', val_size)
    train_data, val_data = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=val_size, shuffle=False)

    val_data_iter = iter(val_loader)
    val_image, val_label = val_data_iter.next()

    model = CNN().cuda()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epoch = 100
    train_set_loss = []
    val_set_loss = []
    train_set_acc = []
    val_set_acc = []
    for ep in range(epoch):  # loop over the dataset multiple times

        running_loss = 0
        trainset_right = 0
        for step, (x, y) in enumerate(train_loader, start=0):
            # get the inputs; data is a list of [inputs, labels]
            inputs = Variable(x).cuda()
            labels = Variable(y).cuda()

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs1 = model(inputs)

            predict_y1 = torch.max(outputs1, dim=1)[1]
            trainset_right += torch.eq(predict_y1, labels).sum().item()

            train_loss = loss_function(outputs1, labels)
            train_loss.backward()
            optimizer.step()

            # validation,这里本来是每n个step输出一次数据，因为通常来讲训练集是成千上万张
            # 但是我们的训练集非常小，我就直接一个step输出一次，这里验证集batchsize=val_size
            # 如果你的验证集数量特别多，建议编写一个evaluate函数用于验证
            running_loss += train_loss.item()
            with torch.no_grad():
                outputs2 = model(val_image.cuda()).cpu()
                predict_y2 = torch.max(outputs2, dim=1)[1]
                val_loss = loss_function(outputs2,predict_y2)
                accuracy = torch.eq(predict_y2, val_label).sum().item() / val_label.size(0)
                print('epoch:[%d/%d] step:[%d/%d]  train_loss: %.3f  val_accuracy: %.3f' %
                        (ep + 1, epoch, step + 1, len(train_loader), train_loss.item(), accuracy))

        train_set_acc.append(trainset_right/train_size)
        loss_per_epoch = running_loss / len(train_loader)
        train_set_loss.append(loss_per_epoch)

        val_set_acc.append(accuracy)
        val_set_loss.append(val_loss)

    print('Finished Training')
    # 可视化
    # 你也可以用vison,tensorbord等可视化工具实时获取损失图像
    epo = numpy.arange(0, len(train_set_acc), 1)
    plt.subplot(2,1,1)
    plt.plot(epo, train_set_loss,color='red',label='train set')
    plt.plot(epo, val_set_loss, color='blue',label='validation set')
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('loss function of train set and validation set')
    plt.subplots_adjust(hspace=0.5)
    plt.subplot(2,1,2)
    plt.plot(epo, train_set_acc,color='red',label='train set')
    plt.plot(epo, val_set_acc, color='blue',label='validation set')
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('accuracy function of train set and validation set')
    plt.show()

    save_path = './model/CNN1.pth'
    torch.save(model.state_dict(), save_path)


if __name__ == '__main__':
    main()
