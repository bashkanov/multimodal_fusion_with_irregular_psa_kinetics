import torch
from models.image_model import NNUNetEncoder

# from  DeepLearningExamples/PyTorch/Segmentation/nnUNet

if __name__ == "__main__":

    filters = [8, 16, 32, 64, 128]
    kernels = [[3, 3, 1], [3, 3, 1], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
    strides = [[1, 1, 1], [2, 2, 1], [2, 2, 1], [2, 2, 2], [2, 2, 1]]

    from torchsummary import summary

    model = NNUNetEncoder(
    # model = NNUNetEncoderDAFT(
        in_channels=3,
        num_output_classes=4,
        kernels=kernels,
        strides=strides,
        dimension=3,
        residual=True,
        filters=filters,
        normalization_layer='instance',
        negative_slope=0.01,
        depth_wise_conv_levels=0,
        dropout_rate=0.1,
        ndim_non_img=23,
        bottleneck_dim=int(23/2),
    )

    summary(model, (3, 140, 140, 24), batch_size=-1, device='cpu')
    print(model)
    test_input, test_input_tab = torch.rand((4, 3, 140, 140, 24)), torch.rand((4, 23))
    test_ouput = model(test_input)
    # test_ouput = model((test_input, test_input_tab))

    print(test_ouput.shape)



# after removing the last block
# Total params: 292,820
# Trainable params: 292,820
# Non-trainable params: 0
# ---------------------------
