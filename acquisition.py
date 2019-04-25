import torch
from architecture import Architecture
from kernel import Kernel
import options as opt

def get_rep_acq(teacher, kernel, action):
    rep = teacher.comp_rep(action)
    acq = kernel.acquisition(rep)
    return rep, acq

def random_search(teacher, kernel, search_n=opt.ac_search_n):
    action_best, rep_best, acq_best = None, None, -1.0
    for i in range(search_n):
        action = teacher.comp_action_rand()
        rep, acq = get_rep_acq(teacher, kernel, action)
        if acq > acq_best:
            action_best, rep_best, acq_best = action, rep, acq
    return action_best, rep_best, acq_best

def random_search_sfn(teacher, kernel, search_n=opt.ac_search_n):
    arch_best, rep_best, acq_best = None, None, -1.0
    for i in range(search_n):
        arch = teacher.comp_arch_rand_sfn()
        rep = arch.rep
        acq = kernel.acquisition(rep)
        if acq > acq_best:
            arch_best, rep_best, acq_best = arch, rep, acq
    return arch_best, rep_best, acq_best