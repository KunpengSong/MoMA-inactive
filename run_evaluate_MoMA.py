import torch
import os
from pytorch_lightning import seed_everything
from torchvision.utils import save_image
import torch.utils.checkpoint
from dataset_lib.dataset_eval_MoMA import Dataset_evaluate_MoMA
from model_lib.modules import MoMA_main_modal
from model_lib.utils import parse_args

#LLM2

def main():
    args = parse_args()
    
    args.device = torch.device("cuda", 0)

    args.load_MLLM = "/common/users/ks1418/paper_experiments_users/intern_bytedance/dreamcast/output/llava_ckpts/model_state-00050000.th" #fine tuned llava ckpt
    args.load_attn_adapters = '/common/users/ks1418/paper_experiments_users/intern_bytedance/MoMA_codebase/checkpoints/self_cross_attn_adapters.th'

    moMA_main_modal = MoMA_main_modal(args)
    moMA_main_modal.to(args.device, dtype=torch.bfloat16)

    seed_everything(0)
    rgb_path = "/common/users/ks1418/paper_experiments_users/intern_bytedance/MoMA_codebase/example_images/myImages/3.jpg"
    mask_path = "/common/users/ks1418/paper_experiments_users/intern_bytedance/MoMA_codebase/example_images/myImages/3_mask.jpg"
    output_path = "/common/users/ks1418/paper_experiments_users/intern_bytedance/MoMA_codebase/output"
    
    subject = "car"
    prompt = "A car in autumn with falling leaves."
    
    batch = Dataset_evaluate_MoMA(rgb_path, prompt, subject, mask_path,moMA_main_modal)
    
    moMA_main_modal.moMA_generator.set_scale('self',1.0)
    moMA_main_modal.moMA_generator.set_scale('cross',1.0)

    for sample_id in range(3):
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True):
            with torch.no_grad(): 
                llava_emb = moMA_main_modal.forward_MLLM(batch).clone().detach()
                img2,mask = moMA_main_modal.moMA_generator.generate_with_MoMA(batch,llava_emb=llava_emb,seed=sample_id+0,device=args.device)                            
                
                save_image(torch.cat([(batch['image'].cpu()+1)/2.0,(img2.cpu()+1)/2.0,mask.cpu()],dim=0),f"{output_path}/{subject}_{prompt}_{sample_id}_new.jpg")
                moMA_main_modal.reset()

    print("done")
main()


