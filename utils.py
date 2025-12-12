import torch
import torch.nn.functional as F

########################################################################################
#Model training utilities
########################################################################################

def freeze_modules(model, module):
    for name, p in model.named_parameters():
        if module in name:
            p.requires_grad = False

def unfreeze_modules(model, module='all'):
    if module == 'all':
        for p in model.parameters():
            p.requires_grad = True
    else:
        for name, p in model.named_parameters():
            if module in name:
                p.requires_grad = True

# Learning-rate finder
def lr_find(Trainer, net):
    lr_finder = Trainer.tuner.lr_find(net)
    print(lr_finder.results)
    fig = lr_finder.plot(suggest=True)
    fig.show()
    new_lr = lr_finder.suggestion()
    return new_lr


