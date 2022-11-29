#导入工具包
import numpy
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import dataset
from torchvision import transforms, datasets
from model import CNN


def main():
    # 数据转换
    data_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # 随机水平翻转，数据增强
        transforms.Grayscale(num_output_channels=1),    # 灰度图通道设为1，彩色图不要这句话
        transforms.ToTensor(),              # 这个必须要有，转化为tensor。（会自动归一化数据）
        transforms.Normalize(mean=[0.17], std=[0.05])   # 标准化（训练集图片归一化后的均值大概是0.17，标准差0.05），加快收敛，不要影响也不大
    ])

    # 读取数据，随机分为训练集和验证集
    # （不随机划分，训练集和验证集都用datasets.ImageFolder加载就行了，保证路径下包含的文件名为0，1，2...）
    # (ImageFolder有一个__getitem__类，它返回的是数据和标签，暂时不会也没关系，后续很多任务需要自己写dataset，
    # 如果想认真搞深度学习的话一定要掌握）
    full_dataset = datasets.ImageFolder('./样本', transform=data_transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    print('训练样本数量：', train_size, '验证集样本数量：', val_size)

    # random_spilt随机划分数据集
    train_data, val_data = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    # DataLoder把dataset按batchsize打包，我们训练才能从里面逐个取数据，
    # batch_size是一次计算的图片数，一般为2，4，8，16，24，32..因为我们图片不多，尺寸小，模型小，所以运算量很小，我就直接取全部
    # 你自己做其他任务不要这样搞，不然算不动的
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=val_size, shuffle=False)

    # iter(dataloader)返回的是一个迭代器，然后可以使用next()访问。
    # 获取验证集图像和标签用于验证正确率
    val_data_iter = iter(val_loader)
    val_image, val_label = val_data_iter.next()

    # 定义模型，.cuda()表示用GPU计算
    model = CNN().cuda()

    # 损失函数，分类任务一般是交叉熵
    loss_function = nn.CrossEntropyLoss()

    # 优化器，Adam，它考虑了动量和自适应学习率，优化器有很多你可以自己了解了解
    # Aadm和SGD比较常用，有时候SGD效果还好一些
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 迭代次数
    epoch = 100

    # 保存数据
    train_set_loss = []
    val_set_loss = []
    train_set_acc = []
    val_set_acc = []

    # 开始训练
    model.train()
    for ep in range(epoch):  # loop over the dataset multiple times
        running_loss = 0
        trainset_right = 0
        for step, (x, y) in enumerate(train_loader, start=0):
            # 取出的数据也要转到gpu
            inputs = x.cuda()
            labels = y.cuda()
            # 优化器梯度清零
            optimizer.zero_grad()
            # 前向传播
            outputs1 = model(inputs)
            # 统计训练集分类正确个数，用于验证训练集的正确率
            # ouputs1维度为[batchsize,2]，取第1维（类别）最大的索引为类别（0或1）
            predict_y1 = torch.max(outputs1, dim=1)[1]
            trainset_right += torch.eq(predict_y1, labels).sum().item()
            # 计算损失
            train_loss = loss_function(outputs1, labels)
            # 反向传播
            train_loss.backward()
            # 更新优化器
            optimizer.step()

            # 如果你不验证的话，那么至此训练过程就写完了，很简单吧
            # 不管你做什么任务都是这几步，梯度清零，预测结果计算损失，反向传播

            # validation,这里本来是每n个step输出一次数据，因为通常来讲训练集是成千上万张
            # 但是我们的训练集非常小，我就直接一个step输出一次，这里验证集batchsize=val_size
            # 如果你的验证集数量特别多，建议编写一个evaluate函数用于验证
            running_loss += train_loss.item()
            with torch.no_grad():
                # 数据要取回cpu才能绘图
                outputs2 = model(val_image.cuda()).cpu()
                predict_y2 = torch.max(outputs2, dim=1)[1]
                val_loss = loss_function(outputs2, predict_y2)
                accuracy = torch.eq(predict_y2, val_label).sum().item() / val_label.size(0)
                print('epoch:[%d/%d] step:[%d/%d]  train_loss: %.3f  val_loss: %.3f  val_accuracy: %.3f' %
                        (ep + 1, epoch, step + 1, len(train_loader), val_loss.item(), train_loss.item(), accuracy))

        train_set_acc.append(trainset_right/train_size)
        loss_per_epoch = running_loss / len(train_loader)
        train_set_loss.append(loss_per_epoch)

        val_set_acc.append(accuracy)
        val_set_loss.append(val_loss)

    print('Finished Training')
    
    # 保存模型
    save_path = './model/CNN.pth'
    torch.save(model.state_dict(), save_path)
    
    
    # 可视化，为了写论文
    # 你也可以用vison,tensorbord等可视化工具实时获取损失图像
    epo = numpy.arange(0, len(train_set_acc), 1)
    plt.subplot(2, 1, 1)
    plt.plot(epo, train_set_loss, color='red', label='train set')
    plt.plot(epo, val_set_loss, color='blue', label='validation set')
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('loss function of train set and validation set')
    plt.subplots_adjust(hspace=0.5)
    plt.subplot(2, 1, 2)
    plt.plot(epo, train_set_acc,color='red',label='train set')
    plt.plot(epo, val_set_acc, color='blue',label='validation set')
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('accuracy function of train set and validation set')
    plt.show()

if __name__ == '__main__':
    main()
