# for CIFAR-10, CIFAR-100

import numpy as np
import os
import model_loader

import torch
import torch.nn as nn

from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler


def fun():
    torch.multiprocessing.freeze_support()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.empty_cache()

    num_epochs = 50
    batch_size = 16

    # augment *padding *HorizontalFlip
    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        # CIFAR-10 (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        # CIFAR-100 (0.5074, 0.4867, 0.4411), (0.2011, 0.1987, 0.2025)
        # CIFAR-100 scaled 224X224 (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Call CIFAR datasets
    train_dataset = datasets.CIFAR100('./train', train=True, transform=transform_train, download=True)
    val_dataset = datasets.CIFAR100('./train', train=True, transform=transform_test, download=True)
    test_dataset = datasets.CIFAR100('./test', train=False, transform=transform_test, download=True)

    # Split train datasets for validation datasets
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(0.2 * num_train))
    train_idx, val_idx = indices[split:], indices[:split]

    val_sampler = SubsetRandomSampler(val_idx)
    np.random.shuffle(indices)
    train_sampler = SubsetRandomSampler(train_idx)

    # Create DataLoader (split by batch size)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=train_sampler,
                                               num_workers=4)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, sampler=val_sampler,
                                             num_workers=4)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    cnt_progress = len(train_loader) // 30

    model_name = 'resnext50'
    model = model_loader.load(model_name, num_class=100).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    loss_func = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[30, 40], gamma=0.1)

    best_val_loss = float('Inf')
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    for epoch in range(num_epochs):
        model.train()
        total_train_acc = 0
        total_train_loss = 0
        for cnt, (images_, labels_) in enumerate(train_loader):
            images = images_.to(device)
            labels = labels_.to(device)

            predict = model(images)
            loss = loss_func(predict, labels.long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_acc += (predict.argmax(1) == labels).float().mean().item()
            total_train_loss += loss.item()
            # progress bar 구현
            if cnt % cnt_progress == 0:
                print(f'\rEpoch {epoch + 1} [', end='')
                for prog in range(cnt // cnt_progress):
                    print('■', end='')
                for prog in range((len(train_loader) - cnt) // cnt_progress):
                    print('□', end='')
                print(']', end='')

        total_train_loss /= len(train_loader)
        total_train_acc /= len(train_loader)
        train_loss.append(total_train_loss)
        train_acc.append(total_train_acc)
        print(f' - Train Loss: {total_train_loss:.4f}, Train Acc: {total_train_acc:.4f}')

        model.eval()  # 모델을 평가 모드로 설정
        total_val_acc = 0
        total_val_loss = 0
        num_loss = 0
        with torch.no_grad():
            for images_, labels_ in val_loader:
                images = images_.to(device)
                labels = labels_.to(device)

                predict = model(images)
                acc = (predict.argmax(1) == labels).float().mean().item()
                loss = loss_func(predict, labels.long())
                total_val_acc += acc
                total_val_loss += loss.item()

            total_val_loss /= len(val_loader)
            total_val_acc /= len(val_loader)

        val_loss.append(total_val_loss)
        val_acc.append(total_val_acc)
        print(f'Validation Loss: {total_val_loss:.4f}, Validation Acc: {total_val_acc:.4f}')

        if total_val_loss < best_val_loss:
            num_loss = 0
            best_val_loss = total_val_loss
            torch.save(model.state_dict(), 'best_model.pt')
        # else:  # early stopping
        #     num_loss += 1
        #     if num_loss == 5:
        #         print(f'early stopping: epoch {epoch}')
        #         break

        scheduler.step()

    model.load_state_dict(torch.load('best_model.pt'))
    model.eval()

    total_test_acc = 0
    total_test_loss = 0
    with torch.no_grad():
        for images_, labels_ in test_loader:
            images = images_.to(device)
            labels = labels_.to(device)

            predict = model(images)

            acc = (predict.argmax(1) == labels).float().mean().item()
            loss = loss_func(predict, labels.long())

            total_test_acc += acc
            total_test_loss += loss.item()

        total_test_loss /= len(test_loader)
        total_test_acc /= len(test_loader)

    print(f'Test Loss: {total_test_loss:.4f}, Test Acc: {total_test_acc:.4f}')

    train_loss = np.array(train_loss)  # 각 모델의 loss, acc 정보를 저장 (추후 그래프를 위함)
    train_acc = np.array(train_acc)
    val_loss = np.array(val_loss)
    val_acc = np.array(val_acc)
    test_result = np.array([total_test_loss, total_test_acc, num_epochs])

    os.makedirs(f'result_numpy/{model_name}', exist_ok=True)
    np.save(os.path.join(f'result_numpy/{model_name}/train_loss'), train_loss)
    np.save(os.path.join(f'result_numpy/{model_name}/train_acc'), train_acc)
    np.save(os.path.join(f'result_numpy/{model_name}/val_loss'), val_loss)
    np.save(os.path.join(f'result_numpy/{model_name}/val_acc'), val_acc)
    np.save(os.path.join(f'result_numpy/{model_name}/test_result'), test_result)

    model = model.to("cpu")
    torch.cuda.empty_cache()


if __name__ == '__main__':
    fun()
