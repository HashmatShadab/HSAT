import timm

def timm_wideresnet50_2(pretrained=False, **kwargs):
    model = timm.create_model('wide_resnet50_2', pretrained=pretrained, num_classes=0)
    return model

def timm_resnet50(pretrained=False, **kwargs):
    model = timm.create_model('resnet50', pretrained=pretrained, num_classes=0)
    return model

def timm_resnetv2_50(pretrained=False, **kwargs):
    model = timm.create_model('resnetv2_50', pretrained=pretrained, num_classes=0)
    return model