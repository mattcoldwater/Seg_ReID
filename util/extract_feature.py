import torch
from opt import opt

def extract_feature(model, loader):
    torch.cuda.empty_cache()
    features = torch.FloatTensor()

    for (inputs, labels) in loader:

        ff = torch.FloatTensor(inputs.size(0), 2048).zero_()
        labels = labels.to(opt.device)
        for i in range(2):
            if i == 1:
                inputs = inputs.index_select(3, torch.arange(inputs.size(3) - 1, -1, -1).long())
            input_img = inputs.to(opt.device)
            outputs = model(input_img, labels)

            f = outputs[0].data.cpu()
            ff = ff + f

        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        features = torch.cat((features, ff), 0)
    return features

def extract_feature_SN(model, loader):
    torch.cuda.empty_cache()
    if opt.model_name == 'FPN':
        features = [torch.FloatTensor().to(opt.device),]*5
    else:
        features = torch.FloatTensor().to(opt.device)

    for (inputs, labels) in loader:
        inputs = inputs.to(opt.device)
        outputs = model.extract_feature(inputs)
        if opt.model_name == 'FPN':
            for i in range(5):
                features[i] = torch.cat((features[i], outputs[i]), 0)
        else:
            features = torch.cat((features, outputs), 0)
    return features  