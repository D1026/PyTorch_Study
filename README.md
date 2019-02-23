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
    
2、矩阵乘法
①   torch.mm(mat1, mat2, out=None) → Tensor
    This function does not broadcast. 
    For broadcasting matrix products, see torch.matmul().
②   torch.matmul(tensor1, tensor2, out=None) → Tensor
    Matrix product of two tensors.