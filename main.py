import torch

from models.fusion import get_model
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser


def get_main_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    arg = parser.add_argument
   
    arg("--discriminator_net", default="nnunet_enc_2FC", help="Image model. Default: %(default)s")
    arg("--time_model_net", default="mTAN_RNN", help="Time sequence model. Default: %(default)s")
    arg("--fusion_mode", default="enc_mtan_concat_trainable", 
        choices=["enc_mtan_emb", "enc_mtan_concat", "enc_mtan_emb_trainable", "enc_mtan_concat_trainable",
                 "concat_tabular", "transformer_encoder", "None"], help="Fusion method. Default: %(default)s")
    arg("--film_location", type=int, default=0, help="location of FiLM when a Film-based model is trained. Default: %(default)s")
    arg("--daft_bottleneck_factor", type=float, default=7.0, help="Reduction factor in a Film-based model. Default: %(default)s")
    arg("--daft_scale", choices=["enabled", "disabled"], default="enabled", help="scaling in film. Default: %(default)s")
    arg("--daft_shift", choices=["enabled", "disabled"], default="enabled", help="shifting in film. Default: %(default)s")
    arg("--daft_activation",choices=["linear", "tanh", "sigmoid"], default="linear", help="activation in film. Default: %(default)s")       
    arg("--num_classes", default=7, type=int, help="Number classes")
    arg("--activation_function", default="ReLU", type=str, help="Activation function used for transformer blocks")
    arg("--norm", default="LayerNorm", type=str, help="Normalization approach used for transformer blocks")
    arg("--dropout_rate", default=0.1, type=float, help="Shutoff probability for each neuron using dropout in transformer blocks")      
    arg("--n_basefilters", default=2, type=int, help="Number or first basefilters for nnUnet-Encoder")
    arg("--in_channels", default=3, type=int, help="Number or input channels for nnUnet-Encoder")
    arg("--num_features_ts", default=32, type=int, help="Number or features in sequential data")
    arg("--pos_freq", default=1000.0, type=float, help="Pos enc freq")
    arg("--pe_type", default="learned", type=str, help="Positional encoding in mTAN [pe,learned]")
    arg("--emb_dim", default=32, type=int, help="Embedding dimension of transformer block")
    arg("--emb_time", default=64, type=int, help="Embedding dimension of time encoding")
    arg("--num_heads", default=1, type=int, help="Number of heads in use in MultiHead SelfAttention mechanism")
    arg("--batch_size", default=4, type=int, help="Number of heads in use in MultiHead SelfAttention mechanism")
    arg("--return_hidden", action="store_true")
    arg("--device", type=str, default="cpu")
    
    return parser.parse_args()

    
if __name__ == "__main__":
    args = get_main_args()
    
    args.tabular_size = args.num_features_ts
    if args.fusion_mode in ['concat_tabular']:
        args.tabular_size = args.num_features_ts
    if args.fusion_mode in ['enc_mtan_concat', 'enc_mtan_concat_trainable']:
        args.tabular_size = args.emb_dim + args.num_features_ts
    elif args.fusion_mode in ['transformer_encoder', 'enc_mtan_emb', 'enc_mtan_emb_trainable']:
        args.tabular_size = args.emb_dim
    
    model = get_model(args=args)
    print(model)
    
    rand_input_image  = torch.rand((args.batch_size, args.in_channels, 140, 140, 24))
    # [batch, time, features]
    rand_observed_data = torch.rand((args.batch_size, 100, args.num_features_ts)) 
    rand_masks = torch.rand((args.batch_size, 100, args.num_features_ts)).bool().int()
    rand_observed_tp = torch.rand((args.batch_size, 100))
 
    outputs, outputs_tab = model(rand_input_image, rand_observed_data[:, 0, :], torch.cat((rand_observed_data, rand_masks), 2), rand_observed_tp)
    print(f"Main fusion target: {outputs.shape}, Auxialiry supervision: {outputs_tab.shape}")
