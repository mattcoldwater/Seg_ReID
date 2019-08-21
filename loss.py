from torch.nn import CrossEntropyLoss
from torch.nn.modules import loss
from util.Loss_Func import TripletLoss, FocalLoss, TripCosLoss

class Loss_MGN(loss._Loss):
    def __init__(self):
        super(Loss_MGN, self).__init__()

    def forward(self, outputs, labels, show=True):
        cross_entropy_loss = CrossEntropyLoss()
        triplet_loss = TripletLoss(margin=1.2)

        Triplet_Loss = [triplet_loss(output, labels) for output in outputs[1:4]]
        Triplet_Loss = sum(Triplet_Loss) / len(Triplet_Loss)

        CrossEntropy_Loss = [cross_entropy_loss(output, labels) for output in outputs[4:]]
        CrossEntropy_Loss = sum(CrossEntropy_Loss) / len(CrossEntropy_Loss)

        loss_sum = Triplet_Loss + 2 * CrossEntropy_Loss

        if show:
            print('\rtotal loss:%.2f  Triplet_Loss:%.2f  CrossEntropy_Loss:%.2f' % (
                loss_sum.data.cpu().numpy(),
                Triplet_Loss.data.cpu().numpy(),
                CrossEntropy_Loss.data.cpu().numpy()),
                    end=' ')
        return loss_sum

class Loss_CGN(loss._Loss):
    def __init__(self):
        super(Loss_CGN, self).__init__()

    def forward(self, outputs, labels, show=True):
        cross_entropy_loss = CrossEntropyLoss()
        CrossEntropy_Loss = [cross_entropy_loss(output, labels) for output in outputs[-1]] # outputs [feat, predictions]
        CrossEntropy_Loss = sum(CrossEntropy_Loss) / len(CrossEntropy_Loss)

        loss_sum = CrossEntropy_Loss

        if show:
            print('total loss: {:.2f}'.format(loss_sum.data.cpu().numpy()))
        return loss_sum

class Loss_Resnet(loss._Loss):
    def __init__(self):
        super(Loss_Resnet, self).__init__()

    def forward(self, outputs, labels, show=True):
        cross_entropy_loss = CrossEntropyLoss()
        triplet_loss = TripletLoss(margin=1.2)

        Triplet_Loss = triplet_loss(outputs[0], labels)

        CrossEntropy_Loss = cross_entropy_loss(outputs[1], labels)

        loss_sum = 2*Triplet_Loss + CrossEntropy_Loss

        if show:
            print('\rtotal loss:%.2f  Triplet_Loss:%.2f  CrossEntropy_Loss:%.2f' % (
                loss_sum.data.cpu().numpy(),
                Triplet_Loss.data.cpu().numpy(),
                CrossEntropy_Loss.data.cpu().numpy()),
                    end=' ')
        return loss_sum

class Loss_SN(loss._Loss):
    def __init__(self):
        super(Loss_SN, self).__init__()

    def forward(self, outputs, labels, show=True):
        cross_entropy_loss = CrossEntropyLoss()
        focal_loss = FocalLoss(gamma=5)

        CrossEntropy_Loss = cross_entropy_loss(outputs[0], labels[0])
        Focal_Loss = focal_loss(outputs[1], labels[1])

        loss_sum = CrossEntropy_Loss + Focal_Loss

        if show:
            print('\rtotal loss:%.2f  CrossEntropy_Loss:%.2f  Focal_Loss:%.2f' % (
                loss_sum.data.cpu().numpy(),
                CrossEntropy_Loss.data.cpu().numpy(),
                Focal_Loss.data.cpu().numpy()),
                    end=' ')
        return loss_sum

class Loss_FPN(loss._Loss):
    def __init__(self):
        super(Loss_FPN, self).__init__()

    def forward(self, outputs, labels, show=True):
        focal_loss = FocalLoss()

        Focal_Loss = [focal_loss(output, labels) for output in outputs]
        loss_sum = sum(Focal_Loss) / len(Focal_Loss)

        if show:
            print('\rtotal loss:%.2f' % (
                loss_sum.data.cpu().numpy()),
                    end=' ')
        return loss_sum

class Loss_AN(loss._Loss):
    def __init__(self):
        super(Loss_AN, self).__init__()

    def forward(self, outputs, labels, show=True):
        cross_entropy_loss = CrossEntropyLoss()
        triplet_loss = TripCosLoss(m=1.2)

        Triplet_Loss = triplet_loss(outputs[0], labels)

        CrossEntropy_Loss = cross_entropy_loss(outputs[1], labels)

        loss_sum = 2*Triplet_Loss + CrossEntropy_Loss

        if show:
            print('\rtotal loss:%.2f  Triplet_Loss:%.2f  CrossEntropy_Loss:%.2f' % (
                loss_sum.data.cpu().numpy(),
                Triplet_Loss.data.cpu().numpy(),
                CrossEntropy_Loss.data.cpu().numpy()),
                    end=' ')
        return loss_sum

class Loss_Segnet(loss._Loss):
    def __init__(self):
        super(Loss_Segnet, self).__init__()

    def forward(self, outputs, labels, show=True):
        cross_entropy_loss = CrossEntropyLoss()
        triplet_loss = TripletLoss(margin=1.2)

        Triplet_Loss = [triplet_loss(output, labels) for output in outputs[1]]
        Triplet_Loss = sum(Triplet_Loss) / len(Triplet_Loss)

        CrossEntropy_Loss = [cross_entropy_loss(output, labels) for output in outputs[2]]
        CrossEntropy_Loss = sum(CrossEntropy_Loss) / len(CrossEntropy_Loss)

        loss_sum = Triplet_Loss + 2 * CrossEntropy_Loss

        if show:
            print('\rtotal loss:%.2f  Triplet_Loss:%.2f  CrossEntropy_Loss:%.2f' % (
                loss_sum.data.cpu().numpy(),
                Triplet_Loss.data.cpu().numpy(),
                CrossEntropy_Loss.data.cpu().numpy()),
                    end=' ')
        return loss_sum