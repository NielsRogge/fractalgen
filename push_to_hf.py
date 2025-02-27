import argparse

from models import fractalgen
import torch


def push_to_hf(args):
    # Create fractal generative model
    model = fractalgen.__dict__[args.model](
        label_drop_prob=args.label_drop_prob,
        class_num=args.class_num,
        attn_dropout=args.attn_dropout,
        proj_dropout=args.proj_dropout,
        guiding_pixel=args.guiding_pixel,
        num_conds=args.num_conds,
        r_weight=args.r_weight,
        grad_checkpointing=args.grad_checkpointing
    )

    # show example
    state_dict = torch.load("pretrained_models/fractalmar_base_in256/checkpoint-last.pth")
    model.load_state_dict(state_dict)
    model.push_to_hub("nielsr/fractalgen-base-in256")

    # reload
    model = fractalgen.FractalGen.from_pretrained("nielsr/fractalgen-base-in256")

    # show example
    # model.sample(cond_list=[], num_iter_list=[], cfg=1.0, cfg_schedule=0.0, temperature=1.0, filter_threshold=0.0, fractal_level=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model parameters
    parser.add_argument("--model", type=str, default="fractalmar_base_in256")
    parser.add_argument("--label_drop_prob", type=float, default=0.1)

    # Optimizer parameters
    parser.add_argument('--grad_checkpointing', action='store_true')
    
    # Fractal generator parameters
    parser.add_argument('--guiding_pixel', action='store_true',
                        help='Use guiding pixels')
    parser.add_argument('--num_conds', type=int, default=1,
                        help='Number of conditions to use')
    parser.add_argument('--r_weight', type=float, default=5.0,
                        help='Loss weight on the red channel')
    parser.add_argument('--attn_dropout', type=float, default=0.1,
                        help='Attention dropout rate')
    parser.add_argument('--proj_dropout', type=float, default=0.1,
                        help='Projection dropout rate')
    
    # Dataset parameters
    parser.add_argument('--class_num', default=1000, type=int)

    args = parser.parse_args()
    push_to_hf(args)