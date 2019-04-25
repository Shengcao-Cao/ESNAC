import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import options as opt
import os
import time

def init_model(model):
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out',
                                    nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0, 0.01)
            nn.init.constant_(module.bias, 0)
    return model

def test_model(model, dataset):
    model.eval()
    correct = 0
    total = 0
    loader = None
    if hasattr(dataset, 'test_loader'):
        loader = dataset.test_loader
    elif hasattr(dataset, 'val_loader'):
        loader = dataset.val_loader
    else:
        raise NotImplementedError('Unknown dataset!')
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs = inputs.to(opt.device)
            targets = targets.to(opt.device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100.0 * correct / total
    return acc

def train_model_teacher(model_, dataset, save_path, epochs=400, lr=0.1,
                        momentum=0.9, weight_decay=5e-4):
    acc_best = 0
    model_best = None
    model = torch.nn.DataParallel(model_.to(opt.device))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                          weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    for i in range(1, epochs + 1):
        model.train()
        scheduler.step()
        loss_total = 0
        batch_cnt = 0
        for batch_idx, (inputs, targets) in enumerate(dataset.train_loader):
            inputs = inputs.to(opt.device)
            targets = targets.to(opt.device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            loss_total += loss.item()
            batch_cnt += 1
        opt.writer.add_scalar('training/loss', loss_total / batch_cnt, i)
        acc = test_model(model, dataset)
        opt.writer.add_scalar('training/acc', acc, i)
        if acc > acc_best:
            acc_best = acc
            model.module.acc = acc
            model_best = model.module
            torch.save(model_best, save_path)
    return model_best, acc_best

def train_model_student(model_, dataset, save_path, idx,
                        optimization=opt.tr_fu_optimization,
                        epochs=opt.tr_fu_epochs, lr=opt.tr_fu_lr,
                        momentum=opt.tr_fu_momentum,
                        weight_decay=opt.tr_fu_weight_decay,
                        lr_schedule=opt.tr_fu_lr_schedule,
                        from_scratch=opt.tr_fu_from_scratch):
    acc_best = 0
    model_best = None
    model = torch.nn.DataParallel(model_.to(opt.device))
    criterion = nn.CrossEntropyLoss()

    if optimization == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                              weight_decay=weight_decay)
    elif optimization == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr,
                              weight_decay=weight_decay)
    if lr_schedule == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100,
                                              gamma=0.1)
    elif lr_schedule == 'linear':
        batch_cnt = len(dataset.train_loader)
        n_total_exp = epochs * batch_cnt
        lr_lambda = lambda n_exp_seen: 1 - n_exp_seen/n_total_exp
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    if from_scratch:
        init_model(model)

    for i in range(1, epochs + 1):
        model.train()
        if lr_schedule == 'step':
            scheduler.step()
        loss_total = 0
        batch_cnt = 0
        for batch_idx, (inputs, targets) in enumerate(dataset.train_loader):
            inputs = inputs.to(opt.device)
            targets = targets.to(opt.device)
            if lr_schedule == 'linear':
                scheduler.step()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            loss_total += loss.item()
            batch_cnt += 1
        opt.writer.add_scalar('training_%d/loss' % (idx), loss_total / batch_cnt, i)
        acc = test_model(model, dataset)
        opt.writer.add_scalar('training_%d/acc' % (idx), acc, i)
        if acc > acc_best:
            acc_best = acc
            model.module.acc = acc
            model_best = model.module
            torch.save(model_best, save_path)
    return model_best, acc_best

def train_model_search(teacher_, students_, dataset,
                       optimization=opt.tr_se_optimization,
                       epochs=opt.tr_se_epochs, lr=opt.tr_se_lr,
                       momentum=opt.tr_se_momentum,
                       weight_decay=opt.tr_se_weight_decay,
                       lr_schedule=opt.tr_se_lr_schedule,
                       loss_criterion=opt.tr_se_loss_criterion):
    n = len(students_)
    accs_best = [0.0] * n
    students_best = [None] * n
    teacher = torch.nn.DataParallel(teacher_.to(opt.device))
    students = [None] * n

    for j in range(n):
        students[j] = torch.nn.DataParallel(students_[j].to(opt.device))
    if loss_criterion == 'KD':
        criterion = nn.MSELoss()
    elif loss_criterion == 'CE':
        criterion = nn.CrossEntropyLoss()
    if optimization == 'SGD':
        optimizers = [optim.SGD(students[j].parameters(), lr=lr,
                                momentum=momentum, weight_decay=weight_decay)
                        for j in range(n)]
    elif optimization == 'Adam':
        optimizers = [optim.Adam(students[j].parameters(), lr=lr,
                                 weight_decay=weight_decay) for j in range(n)]
    if lr_schedule == 'linear':
        batch_cnt = len(dataset.train_loader)
        n_total_exp = epochs * batch_cnt
        lr_lambda = lambda n_exp_seen: 1 - n_exp_seen/n_total_exp
        schedulers = [optim.lr_scheduler.LambdaLR(optimizers[j],
                                                  lr_lambda=lr_lambda)
                        for j in range(n)]

    for i in range(1, epochs + 1):
        teacher.eval()
        for j in range(n):
            students[j].train()
        loss_total = [0.0] * n
        batch_cnt = 0
        for batch_idx, (inputs, targets) in enumerate(dataset.train_loader):
            inputs = inputs.to(opt.device)

            if loss_criterion == 'KD':
                teacher_outputs = None
                with torch.no_grad():
                    teacher_outputs = teacher(inputs)
            elif loss_criterion == 'CE':
                targets = targets.to(opt.device)

            for j in range(n):
                if lr_schedule == 'linear':
                    schedulers[j].step()
                optimizers[j].zero_grad()
                student_outputs = students[j](inputs)
                if loss_criterion == 'KD':
                    loss = criterion(student_outputs, teacher_outputs)
                elif loss_criterion == 'CE':
                    loss = criterion(student_outputs, targets)
                loss.backward()
                optimizers[j].step()
                loss_total[j] += loss.item()
            batch_cnt += 1
        for j in range(n):
            opt.writer.add_scalar('step_%d/sample_%d_loss' % (opt.i, j),
                                   loss_total[j] / batch_cnt, i)
            acc = test_model(students[j], dataset)
            opt.writer.add_scalar('step_%d/sample_%d_acc' % (opt.i, j), acc, i)
            if acc > accs_best[j]:
                accs_best[j] = acc
                students_best[j] = students[j].module
    return students_best, accs_best