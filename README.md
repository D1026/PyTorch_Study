# PyTorch_Study
1、torch.Tensor.sum(dim=1, keepdim= True/False)
    keepdim=True, dim这一维变为 1
            False, dim这一维消失
    
    exsample:
    print(a.shape)
    print(a.sum(dim=0, keepdim=True).shape)
    print(a.sum(dim=0, keepdim=False).shape)
    output:
    torch.Size([2, 3, 4])
    torch.Size([1, 3, 4])
    torch.Size([3, 4])