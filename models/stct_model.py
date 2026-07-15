from .vct_gan_pp_model import VCTGANPPModel


class STCTModel(VCTGANPPModel):
    """STCT: structure-aware token contrastive translation.

    This is the paper-facing model entry.  The implementation reuses the
    VCT-GAN++ training components, but sets the default objective to the STCT
    main line: ViT token contrastive translation + SCL + TRC + TTC.  The
    saliency/perception auxiliary term is disabled by default because it was
    unstable in the current ablation results.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser = VCTGANPPModel.modify_commandline_options(parser, is_train)
        parser.set_defaults(
            dataset_mode="unaligned_double",
            atten_layers="1,3,5",
            lambda_D_ViT=1.0,
            lambda_GAN=1.0,
            lambda_global=5.0,
            lambda_NCE=0.2,
            lambda_structure=1.0,
            lambda_anti_hallucination=0.5,
            lambda_temporal_token=1.0,
            lambda_temporal_luma=0.5,
            lambda_perception=0.0,
            structure_mode="full",
            recover_mode="full",
            temporal_mode="full",
        )
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        if getattr(opt, "lambda_perception", 0.0) <= 0.0:
            self.loss_names = [name for name in self.loss_names if name != "perception"]
