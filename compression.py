import torch
import datasets
import models
from architecture import Architecture
from kernel import Kernel
from record import Record
import acquisition as ac
import graph as gr
import options as opt
import training as tr
import numpy as np
import argparse
from operator import attrgetter
import os
import random
import time
from tensorboardX import SummaryWriter

def seed_everything(seed=127):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

def new_kernels(teacher, record, kernel_n, alpha=opt.co_alpha,
                beta=opt.co_beta, gamma=opt.co_gamma):
    start_time = time.time()
    kernels = []
    for i in range(kernel_n):
        kernel = Kernel(teacher.rep, 0.0)
        indices = []
        for j in range(record.n):
            if random.random() < gamma:
                indices.append(j)
        if len(indices) > 0:
            x = [record.x[i] for i in indices]
            indices = torch.tensor(indices, dtype=torch.long, device=opt.device)
            y = torch.index_select(record.y, 0, indices)
            kernel.add_batch(x, y)
        ma = 0.0
        for j in range(100):
            ll = kernel.opt_step()
            opt.writer.add_scalar('step_%d/kernel_%d_loglikelihood' % (opt.i, i),
                                  ll, j)
            ma = (alpha * ll + (1 - alpha) * ma) if j > 0 else ll
            if j > 5 and abs(ma - ll) < beta:
                break
        kernels.append(kernel)
    opt.writer.add_scalar('compression/kernel_time',
                          time.time() - start_time, opt.i)
    return kernels

def next_samples(teacher, kernels, kernel_n):
    start_time = time.time()
    n = kernel_n
    reps_best, acqs_best, archs_best = [], [], []

    if opt.co_graph_gen == 'get_graph_shufflenet':
        for i in range(n):
            arch, rep, acq = ac.random_search_sfn(teacher, kernels[i])
            archs_best.append(arch)
            reps_best.append(rep)
            acqs_best.append(acq)
            opt.writer.add_scalar('compression/acq', acq, opt.i * n + i - n + 1)
        opt.writer.add_scalar('compression/sampling_time',
                            time.time() - start_time, opt.i)
        return archs_best, reps_best

    else:
        for i in range(n):
            action, rep, acq = ac.random_search(teacher, kernels[i])
            reps_best.append(rep)
            acqs_best.append(acq)
            archs_best.append(teacher.comp_arch(action))
            opt.writer.add_scalar('compression/acq', acq, opt.i * n + i - n + 1)
        opt.writer.add_scalar('compression/sampling_time',
                            time.time() - start_time, opt.i)
        return archs_best, reps_best

def reward(teacher, teacher_acc, students, dataset):
    start_time = time.time()
    n = len(students)
    students_best, students_acc = tr.train_model_search(teacher, students, dataset)
    rs = []
    for j in range(n):
        c = 1.0 - 1.0 * students_best[j].param_n() / teacher.param_n()
        a = 1.0 * students_acc[j] / teacher_acc
        r = c * (2 - c) * a
        opt.writer.add_scalar('compression/compression_score', c,
                              opt.i * n - n + 1 + j)
        opt.writer.add_scalar('compression/accuracy_score', a,
                              opt.i * n - n + 1 + j)
        opt.writer.add_scalar('compression/reward', r,
                              opt.i * n - n + 1 + j)
        rs.append(r)
        students_best[j].comp = c
        students_best[j].acc = students_acc[j]
        students_best[j].reward = r
    opt.writer.add_scalar('compression/evaluating_time',
                          time.time() - start_time, opt.i)
    return students_best, rs

def compression(teacher, dataset, record, step_n=opt.co_step_n,
                kernel_n=opt.co_kernel_n, best_n=opt.co_best_n):

    teacher_acc = tr.test_model(teacher, dataset)
    archs_best = []
    for i in range(1, step_n + 1):
        print ('Search step %d/%d' %(i, step_n))
        start_time = time.time()
        opt.i = i
        kernels = new_kernels(teacher, record, kernel_n)
        students_best, xi = next_samples(teacher, kernels, kernel_n)
        students_best, yi = reward(teacher, teacher_acc, students_best, dataset)
        for j in range(kernel_n):
            record.add_sample(xi[j], yi[j])
            if yi[j] == record.reward_best:
                opt.writer.add_scalar('compression/reward_best', yi[j], i)
        students_best = [student.to('cpu') for student in students_best]
        archs_best.extend(students_best)
        archs_best.sort(key=attrgetter('reward'), reverse=True)
        archs_best = archs_best[:best_n]
        for j, arch in enumerate(archs_best):
            arch.save('%s/arch_%d.pth' % (opt.savedir, j))
        record.save(opt.savedir + '/record.pth')
        opt.writer.add_scalar('compression/step_time',
                              time.time() - start_time, i)

def fully_train(dataset, best_n=opt.co_best_n):
    dataset = getattr(datasets, dataset)()
    for i in range(best_n):
        print ('Fully train student architecture %d/%d' %(i+1, best_n))
        model = torch.load('%s/arch_%d.pth' % (opt.savedir, i))
        tr.train_model_student(model, dataset,
                               '%s/fully_%d.pth' % (opt.savedir, i), i)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Learnable Embedding Space for Efficient Neural Architecture Compression')

    parser.add_argument('--network', type=str, default='resnet34',
                        help='resnet18/resnet34/vgg19/shufflenet')
    parser.add_argument('--dataset', type=str, default='cifar100',
                        help='cifar10/cifar100')
    parser.add_argument('--suffix', type=str, default='0', help='0/1/2/3...')
    parser.add_argument('--device', type=str, default='cuda', help='cpu/cuda')

    args = parser.parse_args()

    seed_everything()

    assert args.network in ['resnet18', 'resnet34', 'vgg19', 'shufflenet']
    assert args.dataset in ['cifar10', 'cifar100']

    if args.network in ['resnet18', 'resnet34']:
        opt.co_graph_gen = 'get_graph_resnet'
    elif args.network == 'vgg19':
        opt.co_graph_gen = 'get_graph_vgg'
    elif args.network == 'shufflenet':
        opt.co_graph_gen = 'get_graph_shufflenet'

    if args.dataset == 'cifar10':
        opt.dataset = 'CIFAR10Val'
    elif args.dataset == 'cifar100':
        opt.dataset = 'CIFAR100Val'

    opt.device = args.device

    opt.model = './models/pretrained/%s_%s.pth' % (args.network, args.dataset)
    opt.savedir = './save/%s_%s_%s' % (args.network, args.dataset, args.suffix)
    opt.writer = SummaryWriter('./runs/%s_%s_%s' % (args.network, args.dataset,
                                                  args.suffix))
    assert not(os.path.exists(opt.savedir)), 'Overwriting existing files!'

    print ('Start compression. Please check the TensorBoard log in the folder ./runs/%s_%s_%s.'%
                                                    (args.network, args.dataset, args.suffix))

    model = torch.load(opt.model).to(opt.device)
    teacher = Architecture(*(getattr(gr, opt.co_graph_gen)(model)))
    dataset = getattr(datasets, opt.dataset)()
    record = Record()
    compression(teacher, dataset, record)
    fully_train(dataset=opt.dataset[:-3])
