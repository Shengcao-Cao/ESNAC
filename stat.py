from compression import *

if __name__ == '__main__':
    teacher_accs = {
        'vgg19_cifar100_'       : 73.71,
        'resnet18_cifar100_'    : 78.68,
        'resnet34_cifar100_'    : 78.71,
        'shufflenet_cifar100_'  : 71.14,
        'vgg19_cifar10_'        : 93.91,
        'resnet18_cifar10_'     : 95.24,
        'resnet34_cifar10_'     : 95.57,
        'shufflenet_cifar10_'   : 90.87,
    }
    path_list = [
        './save/resnet34_cifar100_0',
    ]

    for path in path_list:
        if not(os.path.exists(path)):
            print('Path \'%s\' does not exist!' % (path))
            continue
        print(path)
        teacher_acc = 0.0
        for key, val in teacher_accs.items():
            if path.find(key) != -1:
                teacher_acc = val
        if teacher_acc == 0.0:
            print('Teacher acc not given!')
            continue
        else:
            print('Teacher acc:', teacher_acc)

        # architecture index, number of parameters, compression ratio, compression times, accuracy before & after fully training, f(x) before & after fully training
        print('Index\t#Params\tRatio\tTimes\tAcc before\tAcc after\tf(x) before\tf(x) after')
        best_index = 0
        best_reward2 = 0

        for i in range(opt.co_best_n):
            arch_path = '%s/arch_%d.pth' % (path, i)
            if not(os.path.exists(arch_path)):
                continue
            arch = torch.load(arch_path)
            param_n = arch.param_n()
            reward1 = arch.reward
            comp1 = arch.comp
            comp2 = 1.0 / (1.0 - comp1)
            acc1 = arch.acc

            arch_path = '%s/fully_%d.pth' % (path, i)
            if not(os.path.exists(arch_path)):
                continue
            arch = torch.load(arch_path)
            acc2 = arch.acc
            reward2 = arch.comp * (2 - arch.comp) * acc2 / teacher_acc

            print('%d\t%d\t%.4f\t%.2f\t%.2f\t\t%.2f\t\t%.4f\t\t%.4f' % (i, param_n, comp1, comp2, acc1, acc2, reward1, reward2))
            if reward2 > best_reward2:
                best_index = i
                best_reward2 = reward2

        print('The best is arch %d with f(x) = %.4f' % (best_index, best_reward2))