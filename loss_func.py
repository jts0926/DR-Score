import torch
from torch import nn
#import torch.nn.functional as F
#from torch.autograd import Variable

class Regularization(object):
    def __init__(self, order, weight_decay):
        """
        params
        order: (int) norm order number
        weight_decay: (float) weight decay rate
        """
        super(Regularization, self).__init__()
        self.order = order
        self.weight_decay = weight_decay

    def __call__(self, model):
        """
        params:
        model: (torch.nn.Module object)
        return:
        reg_loss: (torch.Tensor) the regularization(self.order) loss
        """
        reg_loss = 0
        for name, w in model.named_parameters():
            if 'weight' in name:
                reg_loss = reg_loss + torch.norm(w, p=self.order)
        reg_loss = self.weight_decay * reg_loss
        return reg_loss
        
class NegativeLogLikelihood(nn.Module):
    def __init__(self, L2_reg):
        super(NegativeLogLikelihood, self).__init__()
        self.reg = Regularization(order=2, weight_decay=L2_reg)

    def forward(self, risk_pred, y, e, model):
        """
        @params: risk_pred: 预测的生存期/风险函数，即cox回归指数项上的结果，注意该数据与实际生存期间的正负关系（比如风险函数与生存期为法相关系）   shape: (N,1)
        @params: y: 真实事件终止事件（可能为右删失数据，也有可能为真实事件终止）    shape:(N,1)
        @params: e: event indicator， 1-事件终止； 0-右删失     shape:(N,1)
        """
        # If the batch contains no events, return 0 loss
        if e.sum() == 0:
            return torch.tensor(0.0).to(risk_pred.device)
            
        mask = torch.ones(y.shape[0], y.shape[0], device='cuda:0')     
        mask[(y-y.T) > 0] = 0             
        exp_loss = torch.exp(risk_pred.T) * mask
        log_loss = torch.log((exp_loss.sum(dim=1))/(mask.sum(dim=1)))
        e = e.reshape(-1)
        neg_log_loss = -torch.sum((risk_pred.T - log_loss) * e) / torch.sum(e) 

        #print('neg_log_loss',neg_log_loss)
        l2_loss = self.reg(model)
        #print('l2_loss',l2_loss)
        loss = neg_log_loss + l2_loss

        return loss
