import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict


class Added_BCEWithLogitsLoss(nn.Module):
    def __init__(self,top_k_percent_pixels=None,
    hard_example_mining_step=100000):
        super(Added_BCEWithLogitsLoss,self).__init__()
        self.top_k_percent_pixels = top_k_percent_pixels
        if top_k_percent_pixels is not None:
            assert(top_k_percent_pixels>0 and top_k_percent_pixels<1)
        self.hard_example_mining_step=hard_example_mining_step
        if self.top_k_percent_pixels==None:
            self.bceloss = nn.BCEWithLogitsLoss(reduction='mean')
        else:
            self.bceloss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self,dic_tmp,y,step):
        final_loss = 0
        for seq_name in dic_tmp.keys():
            pred_logits = dic_tmp[seq_name]
            gts = y[seq_name]
            if self.top_k_percent_pixels==None:
                final_loss+= self.bceloss(pred_logits,gts)
            else:
            # Only compute the loss for top k percent pixels.
        # First, compute the loss for all pixels. Note we do not put the loss
        # to loss_collection and set reduction = None to keep the shape.
                num_pixels = float(pred_logits.size(2)*pred_logits.size(3))
                pred_logits = pred_logits.view(-1,pred_logits.size(1),pred_logits.size(2)*pred_logits.size(3))
                gts = gts.view(-1,gts.size(1),gts.size(2)*gts.size(3))
                pixel_losses = self.bceloss(pred_logits,gts)
                if self.hard_example_mining_step==0:
                    top_k_pixels=int(self.top_k_percent_pixels*num_pixels)
                else:
                    ratio = min(1.0,step/float(self.hard_example_mining_step))
                    top_k_pixels = int((ratio*self.top_k_percent_pixels+(1.0-ratio))*num_pixels)
                _,top_k_indices = torch.topk(pixel_losses,k=top_k_pixels,dim=2)

                final_loss += nn.BCEWithLogitsLoss(weight=top_k_indices,reduction='mean')(pred_logits,gts)
        return final_loss

class Added_CrossEntropyLoss(nn.Module):
    def __init__(self,top_k_percent_pixels=None,
    hard_example_mining_step=100000):
        super(Added_CrossEntropyLoss,self).__init__()
        self.top_k_percent_pixels = top_k_percent_pixels
        if top_k_percent_pixels is not None:
            assert(top_k_percent_pixels>0 and top_k_percent_pixels<1)
        self.hard_example_mining_step=hard_example_mining_step
        if self.top_k_percent_pixels==None:
            self.celoss = nn.CrossEntropyLoss(ignore_index=255,reduction='mean')
        else:
            self.celoss = nn.CrossEntropyLoss(ignore_index=255,reduction='none')


    def forward(self,dic_tmp,y,step):
        final_loss = 0
        for seq_name in dic_tmp.keys():
            pred_logits = dic_tmp[seq_name]
            gts = y[seq_name]
            if self.top_k_percent_pixels==None:
                final_loss+= self.celoss(pred_logits,gts)
            else:
            # Only compute the loss for top k percent pixels.
        # First, compute the loss for all pixels. Note we do not put the loss
        # to loss_collection and set reduction = None to keep the shape.
                num_pixels = float(pred_logits.size(2)*pred_logits.size(3))
                pred_logits = pred_logits.view(-1,pred_logits.size(1),pred_logits.size(2)*pred_logits.size(3))
                gts = gts.view(-1,gts.size(1)*gts.size(2))
                pixel_losses = self.celoss(pred_logits,gts)
                if self.hard_example_mining_step==0:
                    top_k_pixels=int(self.top_k_percent_pixels*num_pixels)
                else:
                    ratio = min(1.0,step/float(self.hard_example_mining_step))
                    top_k_pixels = int((ratio*self.top_k_percent_pixels+(1.0-ratio))*num_pixels)
                top_k_loss,top_k_indices = torch.topk(pixel_losses,k=top_k_pixels,dim=1)

                final_loss += torch.mean(top_k_loss)
        return final_loss


class AddedEdge_CrossEntropyLoss(nn.Module):
    def __init__(self,top_k_percent_pixels=None,
    hard_example_mining_step=100000):
        super(AddedEdge_CrossEntropyLoss,self).__init__()
        self.top_k_percent_pixels = top_k_percent_pixels
        if top_k_percent_pixels is not None:
            assert(top_k_percent_pixels>0 and top_k_percent_pixels<1)
        self.hard_example_mining_step=hard_example_mining_step
        self.celoss=None
#        if self.top_k_percent_pixels==None:
#            self.celoss = nn.CrossEntropyLoss(ignore_index=255,reduction='mean')
#        else:
#            self.celoss = nn.CrossEntropyLoss(ignore_index=255,reduction='none')


    def forward(self,pred_logits,gts,step):
        pos_num = torch.sum(gts == 1, dtype=torch.float)
        neg_num = torch.sum(gts == 0, dtype=torch.float)

        weight_pos = neg_num / (pos_num + neg_num)
        weight_neg = pos_num / (pos_num + neg_num)
        weights = torch.tensor([weight_neg, weight_pos])
        if self.top_k_percent_pixels==None:
            sig_pred_logits=torch.sigmoid(pred_logits)
#            self.celoss=nn.CrossEntropyLoss(weight=weights.cuda(),ignore_index=255,reduction='mean')
            self.bceloss=nn.BCEWithLogitsLoss(pos_weight=weight_pos.cuda(),reduction='mean')
            if torch.sum(gts)==0:
                dcloss=0
            else:
                dcloss = (torch.sum(sig_pred_logits*sig_pred_logits)+torch.sum(gts*gts))/(torch.sum(2*sig_pred_logits*gts)+1e-5)
            final_loss= 0.1*self.bceloss(pred_logits,gts)+dcloss
            #final_loss= self.celoss(pred_logits,gts.long())
        else:
            self.celoss=nn.CrossEntropyLoss(weight=weights.cuda(),ignore_index=255,reduction='none')
            #self.celoss=nn.CrossEntropyLoss(ignore_index=255,reduction='none')
        # Only compute the loss for top k percent pixels.
           # rst, compute the loss for all pixels. Note we do not put the loss
            #loss_collection and set reduction = None to keep the shape.
            num_pixels = float(pred_logits.size(2)*pred_logits.size(3))
            pred_logits = pred_logits.view(-1,pred_logits.size(1),pred_logits.size(2)*pred_logits.size(3))
            gts = gts.view(-1,gts.size(2)*gts.size(3))
            pixel_losses = self.celoss(pred_logits,gts)
            if self.hard_example_mining_step==0:
                top_k_pixels=int(self.top_k_percent_pixels*num_pixels)
            else:
                ratio = min(1.0,step/float(self.hard_example_mining_step))
                top_k_pixels = int((ratio*self.top_k_percent_pixels+(1.0-ratio))*num_pixels)
            top_k_loss,top_k_indices = torch.topk(pixel_losses,k=top_k_pixels,dim=1)

            final_loss = torch.mean(top_k_loss)
        return final_loss


class DC_and_topk_loss(nn.Module):
    def __init__(self, ce_kwargs, aggregate="sum"):
        super(DC_and_topk_loss, self).__init__()
        self.aggregate = aggregate
        self.ce = TopKLoss(**ce_kwargs)
        self.dc = DiceLoss()

    def forward(self, net_output, target):
        # net_output, target: (b, 1, h, w) with value range[0,1]
        # modified from https://github.com/JunMa11/SegLoss, since the net_output in my project is (b, 1, h, w) with range(0,1), 
        # the Crossentropy loss in the original code is replaced with torch.log + NLLLoss
        dc_loss = self.dc(net_output, target)

        pred_cat = torch.cat((1-net_output, net_output), dim=1)
        if self.ce.k != 0:
            ce_loss = self.ce(pred_cat, target)
        else:
            ce_loss = 0

        if self.aggregate == "sum":
            result = ce_loss + dc_loss
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later?)

        return result, ce_loss, dc_loss

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.epsilon = 1e-5
    
    def forward(self, predict, target):
        assert predict.size() == target.size()
        num = predict.size(0)
        
        pre = predict.view(num, -1)
        tar = target.view(num, -1)
        
        intersection = (pre * tar).sum(-1).sum() 
        union = (pre + tar).sum(-1).sum()
        
        score = 1 - 2 * (intersection + self.epsilon) / (union + self.epsilon)
        
        return score

class CrossentropyND(torch.nn.NLLLoss):
    """
    Network has to have NO NONLINEARITY!
    """
    def forward(self, inp, target):
        target = target.long()
        num_classes = inp.size()[1]

        i0 = 1
        i1 = 2

        while i1 < len(inp.shape): # this is ugly but torch only allows to transpose two axes at once
            inp = inp.transpose(i0, i1)
            i0 += 1
            i1 += 1

        inp = inp.contiguous()
        inp = inp.view(-1, num_classes)

        target = target.view(-1,)

        inp = torch.log(inp+1e-4)

        return super(CrossentropyND, self).forward(inp, target)

class TopKLoss(CrossentropyND):
    """
    Network has to have NO LINEARITY!
    """
    def __init__(self, weight=None, ignore_index=-100, k=10):
        self.k = k
        super(TopKLoss, self).__init__(weight=weight, ignore_index=ignore_index, reduction='none')

    def forward(self, inp, target):
        target = target[:, 0].long()
        res = super(TopKLoss, self).forward(inp, target)
        num_voxels = np.prod(res.shape)
        res, _ = torch.topk(res.view((-1, )), int(num_voxels * self.k / 100), sorted=False)
        return res.mean()


class LossComputer:
    def __init__(self, para):
        super().__init__()
        self.para = para
        self.bce = DC_and_topk_loss()

    def compute(self, data, it):
        losses = defaultdict(int)

        b, s, _, _, _ = data['gt'].shape
        selector = data.get('selector', None)

        for i in range(1, s):
            # Have to do it in a for-loop like this since not every entry has the second object
            # Well it's not a lot of iterations anyway
            for j in range(b):
                if selector is not None and selector[j][1] > 0.5:
                    loss, p = self.bce(data['logits_%d'%i][j:j+1], data['cls_gt'][j:j+1,i], it)
                else:
                    loss, p = self.bce(data['logits_%d'%i][j:j+1,:2], data['cls_gt'][j:j+1,i], it)

                losses['loss_%d'%i] += loss / b
                losses['p'] += p / b / (s-1)

            losses['total_loss'] += losses['loss_%d'%i]

        return losses

def dice_loss(input, target, smooth=1e-4):
    iflat = input.view(-1).float()
    tflat = target.view(-1).float()
    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection + smooth)/(iflat.sum() + tflat.sum() + smooth))

class DC_and_topk_loss_3d(nn.Module):
    def __init__(self, topk=10):
        super(DC_and_topk_loss_3d, self).__init__()
        self.k = topk
        self.NLL = nn.NLLLoss(reduction="none")
        self.dice_loss = DiceLoss()

    def forward(self, net_output, target):
        dc_loss = self.dice_loss(net_output, target)
        ce_loss = self.topk_loss(net_output, target)
        
        result = ce_loss + dc_loss

        return result, ce_loss, dc_loss
    
    def topk_loss(self, net_output, target, smooth=1e-4):
        target = target.squeeze(1).long()
        pred_cat = torch.cat((1-net_output, net_output), dim=1)
        log_pred = torch.log(pred_cat+smooth)
        nll_loss = self.NLL(log_pred, target)
        num_voxels = np.prod(nll_loss.shape)
        results, _ = torch.topk(nll_loss.view((-1, )), int(num_voxels * self.k / 100), sorted=False)
        return results.mean()