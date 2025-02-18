import torch
import torch.nn as nn

from .sequential_baseline import SimpleRNN
from .mTAND_model import enc_mtan_classif, enc_mtan_classif_transformer
from .image_model import NNUNetEncoder, NNUNetEncoderMCM, NNUNetEncoder1FC, NNUNetEncoder2FC, NNUNetEncoderDAFT


def get_image_model(args):
    
    if "nnunet" in args.discriminator_net:
        
        class_dict = {
            "nnunet_enc": NNUNetEncoder,
            "nnunet_enc_mlpcatmlp": NNUNetEncoderMCM,
            "nnunet_enc_1FC": NNUNetEncoder1FC,
            "nnunet_enc_2FC": NNUNetEncoder2FC,
            "nnunet_enc_DAFT": NNUNetEncoderDAFT,
        }
        
        def get_nnunet_enc(args):
            if args.n_basefilters == 2: 
                filters = [2, 4, 8, 16, 32]
            elif args.n_basefilters == 4: 
                filters = [4, 8, 16, 32, 64]
            elif args.n_basefilters == 6: 
                filters = [6, 12, 24, 48, 64]
            elif args.n_basefilters == 8: 
                filters = [8, 16, 32, 64, 128]
            
            kernels = [[3, 3, 1], [3, 3, 1], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
            strides = [[1, 1, 1], [2, 2, 1], [2, 2, 1], [2, 2, 2], [2, 2, 1]]

            model = class_dict[args.discriminator_net](
                in_channels=args.in_channels,
                num_output_classes=(args.num_classes if args.num_classes > 2 else 1),
                kernels=kernels,
                strides=strides,
                dimension=3,
                residual=True,
                filters=filters,
                normalization_layer='instance',
                negative_slope=0.01,
                depth_wise_conv_levels=0,
                dropout_rate=0.1,
                ndim_non_img=args.tabular_size,
                bottleneck_dim=int(filters[-1]/2),
                film_location=args.film_location,
            )
            return model 
        
        return get_nnunet_enc(args)
    else:
        raise Warning(f"Image model not found {args.discriminator_net}")
    
    
def get_sequence_model(args):
    
    if args.time_model_net == "mTAN_transformer":
        return enc_mtan_classif_transformer(
            input_dim=args.num_features_ts, 
            query=torch.linspace(0, 1., 128), 
            nhidden=args.emb_dim, 
            embed_time=args.emb_time,
            num_heads=args.num_heads,
            learn_emb=args.pe_type == "learned",
            freq=args.pos_freq,
            device=args.device,
            num_classes=args.num_classes,
            return_hidden=args.return_hidden,
            ).to(args.device)
    
    if "mTAN" in args.time_model_net:
        encoder_type = args.time_model_net.split("_")[-1]
        return enc_mtan_classif(
            input_dim=args.num_features_ts, 
            query=torch.linspace(0, 1., 128), 
            nhidden=args.emb_dim, 
            embed_time=args.emb_time,
            num_heads=args.num_heads,
            learn_emb=args.pe_type == "learned",
            freq=args.pos_freq,
            device=args.device,
            num_classes=args.num_classes,
            return_hidden=args.return_hidden,
            encoder_type=encoder_type,
            ).to(args.device)
    
    if args.time_model_net in ['RNN', 'GRU', 'LSTM']:
        return SimpleRNN(input_dim=args.num_features_ts,
                         nhidden=args.emb_dim, 
                         output_size=args.num_classes,
                         encoder_type=args.time_model_net, 
                         device=args.device,
                         append_missing="mask" in args.seq_aux_features, 
                         append_timestamps="time" in args.seq_aux_features, 
                         )
    
    raise Warning(f"Image model not found {args.discriminator_net}")

        
def get_model(args):
    
    if args.fusion_mode in ["concat_tabular", "enc_mtan_emb", "enc_mtan_concat", "enc_mtan_emb_trainable", "enc_mtan_concat_trainable"]:
        class FusedModel(nn.Module):
            def __init__(self):
                super(FusedModel, self).__init__()
                
                # make it explicit here
                args.return_hidden = True

                self.seq_model = get_sequence_model(args)
                self.image_model = get_image_model(args)
                self.fusion_mode = args.fusion_mode
        
            def forward(self, image, tabular, observed_data, times):
                tabular = tabular.to(dtype=image.dtype)
                output_seq, emb = self.seq_model(observed_data, times)
                
                if 'enc_mtan' in self.fusion_mode:
                    if 'enc_mtan_concat' in self.fusion_mode:
                        fused_features = torch.cat((emb, tabular), dim=-1)
                        fused_features = fused_features.to(dtype=image.dtype)
                        output = self.image_model((image, fused_features))
                    elif 'enc_mtan_emb' in self.fusion_mode:
                        output = self.image_model((image, emb))
                
                elif 'concat_tabular' in self.fusion_mode:
                    output = self.image_model((image, tabular))
                else:    
                    raise Warning(f"fusion_mode is not known: {self.fusion_mode}")
                            
                return output, output_seq
        model = FusedModel()
        
    elif "mTAN" in args.time_model_net or args.time_model_net in ['RNN', 'GRU', 'LSTM']:
        model = get_sequence_model(args)
  
    elif args.discriminator_net in ["resnet", "nnunet_enc"]:
        model = get_image_model(args)

    else:
        raise Warning(f"Found no implementation for {args.discriminator_net}, {args.time_model_net}, or {args.fusion_mode}")
            
    return model