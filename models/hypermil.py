import torch
import torch.nn as nn
import torch.nn.functional as F
from conch.open_clip_custom import create_model_from_pretrained, tokenize, get_tokenizer
from models.infonce import InfoNCEHP, hyper_sim
from models.conch_custom import TextEncoder, PromptLearner
import os
from PIL import Image
from PIL import PngImagePlugin
import math
from models import lorentz

PngImagePlugin.MAX_TEXT_CHUNK = 1024*2**20

conch_model, preprocess = create_model_from_pretrained('conch_ViT-B-16', "hf_hub:MahmoodLab/conch", hf_auth_token="your_token")
conch_model = conch_model.to("cuda")

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            # ref from huggingface
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.Linear):
            # ref from clam
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class HyperAGG(nn.Module):
    def __init__(self,input_dim,act='gelu',bias=False,dropout=0.25,rrt=None):
        super(HyperAGG, self).__init__()
        self.L = 512 #512
        self.D = 128 #128
        self.K = 1

        if act.lower() == 'gelu':
            self.feature = [nn.LayerNorm(input_dim),
                        nn.GELU(),
                        nn.Linear(input_dim, 512)]
            self.feature += [nn.LayerNorm(input_dim),
                             nn.GELU()]
        else:
            self.feature = [
                # nn.ReLU(),
                # nn.Dropout(0.25),
                nn.Linear(input_dim, 512)]
            self.feature += [nn.ReLU()]

        if dropout:
            self.feature += [nn.Dropout(0.25)]

        self.feature = nn.Sequential(*self.feature)

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )
                
        self.apply(initialize_weights)
        
    def forward(self, x):
        feature = self.feature(x)
         
        A = self.attention(feature) # N,1
        
        A = torch.transpose(A, -1, -2)  # BX1xN (1,N)
        A = F.softmax(A, dim=-1)  # softmax over N : patch attention
        M = torch.bmm(A, feature)  # BXKxL (N,1,512) : slide_feat

        return M

class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 num_inner: int = 0,
                 pre_norm: bool = False, device: torch.device = None,
                 **kwargs):
        super().__init__()

        self.pre_norm = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.GELU(),
        ) if pre_norm else nn.Identity()

        self._real_output_dim = output_size

        self.fc1 = nn.Linear(input_size, hidden_size)

        blocks = []
        for _ in range(num_inner):
            blocks.append(nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                # nn.Dropout(0.25), ######
                nn.Linear(hidden_size, hidden_size),
            ))
        self.blocks = nn.ModuleList(blocks)

        self.final = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            # nn.Dropout(0.25),
            nn.Linear(hidden_size, output_size),
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_norm(x)
        x = self.fc1(x)
        for block in self.blocks:
            x = x + block(x)
        x = self.final(x)

        return x

class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        ratio = 0.2
        feature = self.fc(x)
        x = ratio * feature + (1-ratio) * x
        return x

class Projection(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, drop=0.25):
        super(Projection, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim) 
        self.relu = nn.ReLU()     
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_dim, output_dim) 
        # self.drop2 = nn.Dropout(drop)
        
    def forward(self, x):
        x = self.fc1(x)      
        x = self.relu(x)     
        x = self.drop1(x)
        x = self.fc2(x)    
        # x = self.drop2(x)
        return x

class HyperMIL(nn.Module):
    def __init__(self, input_dim=512, n_classes=2, dropout=0.25, act='relu', rrt=None, n_ctx=None, n_flp=0, num_patch_prompt=0, task=None, fold=None, exp_code=None, train=True, base_mil=None, slide_align=1):
        super(HyperMIL, self).__init__()
        self.slide_align = int(slide_align)
        if base_mil == 'abmil':
            from models.abmil import DAttention
            self.mil = DAttention(n_classes=n_classes,use_classifier=False)
    
        # ################## Project before Hyperbolic ##############
        self.patch_feat_proj = Projection(input_dim, input_dim, input_dim)
        self.region_feat_proj = Projection(input_dim, input_dim, input_dim)
        self.slide_feat_proj = Projection(input_dim, input_dim, input_dim)
        self.txt_feat_proj = Projection(input_dim, input_dim, input_dim)
        
        self.patch2region = HyperAGG(input_dim,act='relu')
        self.region2slide = HyperAGG(input_dim,act='relu')
    
        self.apply(initialize_weights)
        
        ################## Hyperbolic Parameters ##################
        # Initialize curvature parameter. Hyperboloid curvature will be `-curv`.
        curv_init=1.0
        learn_curv=True
        embed_dim = input_dim
        
        # Initialize a learnable logit scale parameter.
        self.logit_scale = nn.Parameter(torch.tensor(1 / 0.07).log())
        
        self.curv = nn.Parameter(
            torch.tensor(curv_init).log(), requires_grad=learn_curv
        )
        # When learning the curvature parameter, restrict it in this interval to
        # prevent training instability.
        self._curv_minmax = {
            "max": math.log(curv_init * 10),
            "min": math.log(curv_init / 10),
        }

        # Learnable scalars to ensure that image/text features have an expected
        # unit norm before exponential map (at initialization).
        self.patch_alpha = nn.Parameter(torch.tensor(embed_dim**-0.5).log())
        self.region_alpha = nn.Parameter(torch.tensor(embed_dim**-0.5).log())
        self.slide_alpha = nn.Parameter(torch.tensor(embed_dim**-0.5).log())
        self.textual_alpha = nn.Parameter(torch.tensor(embed_dim**-0.5).log())
        ################## Hyperbolic Parameters ##################
        if task == 'BRCA':
            from models.prompts import brca_prompts
            prompts = brca_prompts()
            self.vis_concepts = None
            self.region_top_k = 10
            self.patch_top_k_thre = [0.25,0.3]
            self.patch_top_k = 100
            self.region_top_k_inv = 1
        elif task == 'BRCA_HER2':
            from models.prompts import brca_her2_prompts
            prompts = brca_her2_prompts()
            visual_prompts_path = './prmpts/brca_her2'
            vis_feats_pos = []
            vis_feats_neg = []
            for file in os.listdir(visual_prompts_path+'/pos'):
                image = Image.open(visual_prompts_path+'/pos/'+file)
                image = preprocess(image).unsqueeze(0).to('cuda')
                with torch.inference_mode():
                    image_embs = conch_model.encode_image(image, proj_contrast=True, normalize=True)
                vis_feats_pos.append(image_embs)
            for file in os.listdir(visual_prompts_path+'/neg'):
                image = Image.open(visual_prompts_path+'/neg/'+file)
                image = preprocess(image).unsqueeze(0).to('cuda')
                with torch.inference_mode():
                    image_embs = conch_model.encode_image(image, proj_contrast=True, normalize=True)
                vis_feats_neg.append(image_embs)
            vis_feats_pos = torch.concat(vis_feats_pos, dim=0)
            vis_feats_neg = torch.concat(vis_feats_neg, dim=0)
            self.vis_concepts = torch.stack([vis_feats_neg, vis_feats_pos], dim=0)
            self.region_top_k = 10
            self.region_top_k_inv = 1
            self.patch_top_k_thre = [0.625,0.525]  
            self.patch_top_k = 50
        elif task == 'LUAD_EGFR':
            from models.prompts import luad_egfr_prompts
            prompts = luad_egfr_prompts()
            visual_prompts_path = './prmpts/luad_egfr'
            vis_feats_pos = []
            vis_feats_neg = []
            for file in os.listdir(visual_prompts_path+'/pos'):
                image = Image.open(visual_prompts_path+'/pos/'+file)
                image = preprocess(image).unsqueeze(0).to('cuda')
                with torch.inference_mode():
                    image_embs = conch_model.encode_image(image, proj_contrast=True, normalize=True)
                vis_feats_pos.append(image_embs)
            for file in os.listdir(visual_prompts_path+'/neg'):
                image = Image.open(visual_prompts_path+'/neg/'+file)
                image = preprocess(image).unsqueeze(0).to('cuda')
                with torch.inference_mode():
                    image_embs = conch_model.encode_image(image, proj_contrast=True, normalize=True)
                vis_feats_neg.append(image_embs)
            vis_feats_pos = torch.concat(vis_feats_pos, dim=0)
            vis_feats_neg = torch.concat(vis_feats_neg, dim=0)
            self.vis_concepts = torch.stack([vis_feats_neg, vis_feats_pos], dim=0)
            self.region_top_k = 10
            self.region_top_k_inv = 1
            self.patch_top_k_thre = [0.625,0.525]  
            self.patch_top_k = 50
        elif task == 'NSCLC':
            from models.prompts import nsclc_prompts
            prompts = nsclc_prompts()
            self.vis_concepts = None
            self.region_top_k = 10
            self.patch_top_k_thre = [0.25,0.25] 
            self.patch_top_k = 50
            self.region_top_k_inv = 1
     
       
        
        self.ori_feats = []
        for i in range(len(prompts)):
            tokenized_templates = tokenize(texts=prompts[i], tokenizer=get_tokenizer())
            self.ori_feats.append(conch_model.encode_text(tokenized_templates.to('cuda')).detach())
        self.ori_feats = [feat.to("cuda") for feat in self.ori_feats]
        

        conch_text = conch_model.text
        self.prompt_learner_patch = nn.ModuleList([PromptLearner(prompt, conch_text, n_ctx, n_flp, num_patch_prompt, is_shared=False) for prompt in prompts])
        self.prompt_learner_region = nn.ModuleList([PromptLearner(prompt, conch_text, n_ctx, n_flp, num_patch_prompt, is_shared=False) for prompt in prompts])
        self.prompt_learner_global = nn.ModuleList([PromptLearner(prompt, conch_text, n_ctx, n_flp, num_patch_prompt, is_shared=False) for prompt in prompts])
        self.tokenized_prompts_patch = [learner.tokenized_prompts for learner in self.prompt_learner_patch]
        self.tokenized_prompts_region = [learner.tokenized_prompts for learner in self.prompt_learner_region]
        self.tokenized_prompts_global = [learner.tokenized_prompts for learner in self.prompt_learner_global]
        self.text_encoder = TextEncoder(conch_text)
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        self.infonce_hp_loss = InfoNCEHP(reduction='mean',sim_mode='hyper_angle')
        self.infonce_hp_dis_loss = InfoNCEHP(reduction='mean',sim_mode='hyper_dis')

    def hyper_proj(self, x, alpha):
        x_hp = x * alpha.exp()
        with torch.autocast(x_hp.device.type, dtype=torch.float32):
            x_hp = lorentz.exp_map0(x_hp, self.curv.exp())
        return x_hp
    
    def encode_text(self, tokens: torch.Tensor, project: bool):
        """
        Args:
            tokens: List of tensors, each containing text tokens. Tensors may have
                variable length (they will be padded internally).
            project: Lift features from the encoder onto the Hyperboloid.
        """

        # Get Euclidean features from the encoder (without L2 normalization).
        text_feats = self.txt_feat_proj(tokens)

        if project:
            text_feats_hp = self.hyper_proj(text_feats, self.textual_alpha)
        else:
            text_feats_hp = None
            
        return text_feats, text_feats_hp
    
    def encode_slide(self, slide_feats: torch.Tensor, project: bool):
        """
        Args:
            tokens: List of tensors, each containing text tokens. Tensors may have
                variable length (they will be padded internally).
            project: Lift features from the encoder onto the Hyperboloid.
        """

        # Get Euclidean features from the encoder (without L2 normalization).
        slide_feats = self.slide_feat_proj(slide_feats)

        if project:
            slide_feats_hp = self.hyper_proj(slide_feats, self.slide_alpha)
        else:
            slide_feats_hp = None
            
        return slide_feats, slide_feats_hp
    def encode_patch(self, patch_feats: torch.Tensor, project: bool):
        """
        Args:
            tokens: List of tensors, each containing text tokens. Tensors may have
                variable length (they will be padded internally).
            project: Lift features from the encoder onto the Hyperboloid.
        """

        # Get Euclidean features from the encoder (without L2 normalization).
        patch_feats = self.patch_feat_proj(patch_feats)

        if project:
            patch_feats_hp = self.hyper_proj(patch_feats, self.patch_alpha)
        else:
            patch_feats_hp = None
            
        return patch_feats, patch_feats_hp

    def encode_region(self, region_feats: torch.Tensor, project: bool):
        """
        Args:
            tokens: List of tensors, each containing text tokens. Tensors may have
                variable length (they will be padded internally).
            project: Lift features from the encoder onto the Hyperboloid.
        """

        # Get Euclidean features from the encoder (without L2 normalization).
        region_feats = self.region_feat_proj(region_feats)

        if project:
            region_feats_hp = self.hyper_proj(region_feats, self.region_alpha)
        else:
            region_feats_hp = None
            
        return region_feats, region_feats_hp
    def encode_visual(self, visual_feats: torch.Tensor, project: bool):
        """
        Args:
            tokens: List of tensors, each containing text tokens. Tensors may have
                variable length (they will be padded internally).
            project: Lift features from the encoder onto the Hyperboloid.
        """

        # Get Euclidean features from the encoder (without L2 normalization).
        visual_feats = self.visual_feat_proj(visual_feats)

        if project:
            visual_feats_hp = self.hyper_proj(visual_feats, self.visual_alpha)
        else:
            visual_feats_hp = None
            
        return visual_feats, visual_feats_hp
    
    def reparameterise(self, mu, logvar):
        epsilon = torch.randn_like(mu)
        return mu + epsilon * torch.exp(logvar / 2)

    def kl_loss(self, mu, logvar):
        kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        kl_mean = torch.mean(kl_div)
        return kl_mean

    def compute_exclude_loss(self, x, y, _curv, alpha = 1.0, beta=0.8, class_angle=torch.pi):
        # x should exclude y
        _angle = lorentz.oxy_angle(x, y, _curv)
        _aperture = lorentz.half_aperture(x, _curv)
                
        object_angle = lorentz.oxy_angle(x,y, _curv) + lorentz.oxy_angle(y,x, _curv) - torch.pi
        
        factor_a = alpha * (torch.clamp(_aperture/_angle - 1,max=3)).exp()
        
        factor_b = 1
        
        enclusion_loss = factor_a * torch.clamp(_aperture - factor_b * _angle, min=0)
        return enclusion_loss
    
    def compute_entail_loss(self, x, y, _curv, alpha=1.0):
        # x should entail y
        _angle = lorentz.oxy_angle(x, y, _curv)
        _aperture = lorentz.half_aperture(x, _curv)
        factor_a = (torch.clamp(_angle/_aperture - 1,max=3)).exp()
        entailment_loss = factor_a *torch.clamp(_angle - alpha*_aperture, min=0)
        return entailment_loss
    def compute_pos_loss(self, x, y, _curv, margin=0.0):
        x_dis = lorentz.hyperbolic_distance_to_origin(x,_curv)
        y_dis = lorentz.hyperbolic_distance_to_origin(y,_curv)
        
        position_loss = torch.clamp(x_dis + margin - y_dis,min=0)
                
        return position_loss / 2
    
    def forward(self, x1, x2=None, label=None, return_attn=False,no_norm=False, pretrain=False, club=None, coords=None):
        N, k1, k2, D = x1.shape
        ori_patch_feats = x1.reshape(N,-1,D)
        ori_region_feats = x2.reshape(N,1,D)

        results_dict = {}
        
        ######## Hyperbolic Forward #############
        self.curv.data = torch.clamp(self.curv.data, **self._curv_minmax)
        _curv = self.curv.exp()

        # Clamp scaling factors such that they do not up-scale the feature norms.
        # Once `exp(scale) = 1`, they can simply be removed during inference.
        self.patch_alpha.data = torch.clamp(self.patch_alpha.data, max=0.0)
        self.region_alpha.data = torch.clamp(self.region_alpha.data, max=0.0)
        # self.visual_alpha.data = torch.clamp(self.visual_alpha.data, max=0.0)
        self.textual_alpha.data = torch.clamp(self.textual_alpha.data, max=0.0)
        self.slide_alpha.data = torch.clamp(self.slide_alpha.data, max=0.0)
        
        ori_text_feats = []
        for ori_feat in self.ori_feats:
            ori_text_feats.append(ori_feat.mean(dim=0))
        ori_text_feats = torch.stack(ori_text_feats, dim=0)
        
        ori_text_feats, ori_text_feats_hp = self.encode_text(ori_text_feats, project=True)
        
        patch_prompts_patch = []
        patch_prompts_region = []
        patch_prompts_global = []
        
        for i in range(len(self.prompt_learner_patch)):
            patch_prompts_patch.append(self.prompt_learner_patch[i]())
        for i in range(len(self.prompt_learner_region)):
            patch_prompts_region.append(self.prompt_learner_region[i]())
        for i in range(len(self.prompt_learner_global)):
            patch_prompts_global.append(self.prompt_learner_global[i]())
            
        patch_tokenized_prompts_patch = self.tokenized_prompts_patch
        patch_tokenized_prompts_region = self.tokenized_prompts_region
        patch_tokenized_prompts_global = self.tokenized_prompts_global
        
        self.feats_patch = [self.text_encoder(prompt, tokenized_prompt) for prompt, tokenized_prompt in zip(patch_prompts_patch, patch_tokenized_prompts_patch)]
        self.feats_region = [self.text_encoder(prompt, tokenized_prompt) for prompt, tokenized_prompt in zip(patch_prompts_region, patch_tokenized_prompts_region)]
        self.feats_global = [self.text_encoder(prompt, tokenized_prompt) for prompt, tokenized_prompt in zip(patch_prompts_global, patch_tokenized_prompts_global)]
                
        text_patch_feats, text_patch_feats_hp = self.encode_text(torch.stack([feat_patch.mean(dim=0) for feat_patch in self.feats_patch],dim=0),project=True)
        text_region_feats, text_region_feats_hp = self.encode_text(torch.stack([feat_region.mean(dim=0) for feat_region in self.feats_region],dim=0),project=True)
        text_feats, text_feats_hp = self.encode_text(torch.stack([feat_global.mean(dim=0) for feat_global in self.feats_global],dim=0),project=True)

        ori_image_feats = torch.cat([ori_patch_feats,ori_region_feats],dim=1).reshape(1,-1,D)
        
        patch_feats, patch_feats_hp = self.encode_patch(ori_patch_feats, project=True)
        
        #####################v3##################
        region_feats = self.patch2region(patch_feats)
        region_feats = region_feats.squeeze(1)
        region_feats_hp = self.hyper_proj(region_feats.unsqueeze(1),self.region_alpha)
        slide_feats = self.region2slide(region_feats.squeeze(1).unsqueeze(0)).squeeze(1)
        slide_feats_hp = self.hyper_proj(slide_feats,self.slide_alpha)
        ##############################################
                
        if label is not None:
            ###################################
            if self.vis_concepts is None:
                patch_sim = torch.stack([ori_patch_feats @ ori_feat for ori_feat in self.ori_feats[label[0]]]).mean(dim=0)
            else:
                patch_sim = torch.stack([ori_patch_feats @ ori_feat for ori_feat in self.vis_concepts[label[0]]]).mean(dim=0)

            # temperature = 0.05
            patch_sim_flatten = patch_sim.view(-1)
            
            patch_top_k = min(self.patch_top_k,patch_sim_flatten.size(0))
            patch_sim_value, patch_sim_ind = torch.topk(patch_sim_flatten, patch_top_k)
            
            region_sim = patch_sim.max(dim=-1)[0]
            
            region_top_k = min(self.region_top_k,region_sim.size(0))

            region_sim_value, region_sim_ind = torch.topk(region_sim, region_top_k)
         
            region_feats_hp_flatten = region_feats_hp.reshape(-1,D)[region_sim_ind]
            patch_feats_hp_flatten = patch_feats_hp.reshape(-1,D)[patch_sim_ind]
            
            ####################################
            if not slide_feats_hp.requires_grad: 
                results_dict[f'slide_feats_hp'] = slide_feats_hp
                results_dict[f'region_feats_hp'] = region_feats_hp_flatten  
                results_dict[f'patch_feats_hp'] = patch_feats_hp_flatten 
                results_dict[f'text_region_feats_hp'] = text_region_feats_hp
                results_dict[f'text_patch_feats_hp'] = text_patch_feats_hp
                results_dict[f'text_feats_hp'] = text_feats_hp
            with torch.autocast(ori_image_feats.device.type, dtype=torch.float32):
      
                self.logit_scale.data = torch.clamp(self.logit_scale.data, max=4.6052)
                _scale = self.logit_scale.exp() 

                pos_feats_hp = text_feats_hp[label]
                neg_feats_hp = text_feats_hp[~label]
                slide_info_loss = self.infonce_hp_loss(slide_feats_hp, pos_feats_hp, neg_feats_hp, scale=_scale,_curv=_curv)
                slide_info_loss += self.infonce_hp_loss(pos_feats_hp, slide_feats_hp, neg_feats_hp, scale=_scale,_curv=_curv)
               
                ########################################################################
                pos_region_feats_hp = text_region_feats_hp[label]
                neg_region_feats_hp = text_region_feats_hp[~label]
                
                region_info_loss = self.infonce_hp_loss(region_feats_hp_flatten, pos_region_feats_hp, neg_region_feats_hp, scale=_scale,_curv=_curv)  #!!  
                region_info_loss += self.infonce_hp_loss(pos_region_feats_hp, region_feats_hp_flatten, neg_region_feats_hp, scale=_scale,_curv=_curv)  #!!  
                
                # ########################################################################
                pos_patch_feats_hp = text_patch_feats_hp[label]
                neg_patch_feats_hp = text_patch_feats_hp[~label]
                
                patch_info_loss = self.infonce_hp_loss(patch_feats_hp_flatten, pos_patch_feats_hp, neg_patch_feats_hp, scale=_scale,_curv=_curv)  #!!  
                patch_info_loss += self.infonce_hp_loss(pos_patch_feats_hp, patch_feats_hp_flatten, neg_patch_feats_hp, scale=_scale,_curv=_curv)  #!!  

                info_loss = 1/3*(slide_info_loss+region_info_loss+patch_info_loss)
                
                ##########################################################################
                class_angle = lorentz.oxy_angle(text_feats_hp[label], text_feats_hp[~label], _curv) + lorentz.oxy_angle(text_feats_hp[~label], text_feats_hp[label], _curv) - torch.pi
                contrastive_loss = info_loss 
                
                # Hyperbolic entailment loss: text should entail matching image.         

                entailment_loss = []
                position_loss = []
                entailment_loss.append(self.compute_entail_loss(slide_feats_hp, region_feats_hp.view(-1,D), _curv, alpha=1.0).view(-1).mean(dim=-1,keepdim=True))
                entailment_loss.append(self.compute_entail_loss(region_feats_hp, patch_feats_hp, _curv, alpha=1.0).view(-1).mean(dim=-1,keepdim=True))
                entailment_loss.append(self.compute_entail_loss(text_feats_hp[label], slide_feats_hp, _curv, alpha=0.8).view(-1).mean(dim=-1,keepdim=True))
                entailment_loss.append(self.compute_entail_loss(text_region_feats_hp[label], region_feats_hp_flatten, _curv, alpha=1.0).view(-1).mean(dim=-1,keepdim=True))
                entailment_loss.append(self.compute_entail_loss(text_patch_feats_hp[label], patch_feats_hp_flatten, _curv, alpha=1.0).view(-1).mean(dim=-1,keepdim=True))
                entailment_loss.append(self.compute_entail_loss(text_feats_hp, text_region_feats_hp, _curv, alpha=0.8).view(-1).mean(dim=-1,keepdim=True))  
                entailment_loss.append(self.compute_entail_loss(text_region_feats_hp, text_patch_feats_hp, _curv, alpha=0.8).view(-1).mean(dim=-1,keepdim=True)) 
                entailment_loss.append(self.compute_entail_loss(text_patch_feats_hp[label], slide_feats_hp, _curv, alpha=0.8).view(-1).mean(dim=-1,keepdim=True)) 
                entailment_loss.append(self.compute_entail_loss(text_region_feats_hp[label], slide_feats_hp, _curv, alpha=0.8).view(-1).mean(dim=-1,keepdim=True))  

                position_loss.append(self.compute_pos_loss(slide_feats_hp, region_feats_hp.view(-1,D), _curv, margin=0.05).view(-1).mean(dim=-1,keepdim=True))
                position_loss.append(self.compute_pos_loss(region_feats_hp, patch_feats_hp, _curv, margin=0.05).view(-1).mean(dim=-1,keepdim=True))
                position_loss.append(self.compute_pos_loss(text_patch_feats_hp[label], patch_feats_hp.view(-1,D), _curv, margin=1.0).view(-1).mean(dim=-1,keepdim=True))
                position_loss.append(self.compute_pos_loss(text_patch_feats_hp[label], region_feats_hp.view(-1,D), _curv, margin=1.0).view(-1).mean(dim=-1,keepdim=True))
                position_loss.append(self.compute_pos_loss(text_patch_feats_hp[label], slide_feats_hp.view(-1,D), _curv, margin=1.0).view(-1).mean(dim=-1,keepdim=True))
                position_loss.append(self.compute_pos_loss(text_feats_hp, text_region_feats_hp, _curv, margin=0.05).view(-1).mean(dim=-1,keepdim=True))  
                position_loss.append(self.compute_pos_loss(text_region_feats_hp, text_patch_feats_hp, _curv, margin=0.05).view(-1).mean(dim=-1,keepdim=True))  
                
                entailment_loss =  torch.cat(entailment_loss,dim=0).mean()  
                position_loss = torch.cat(position_loss,dim=0).mean()
                
                exclusion_loss = []
                
                region_class_angle = lorentz.oxy_angle(text_region_feats_hp[label], text_region_feats_hp[~label], _curv) + lorentz.oxy_angle(text_region_feats_hp[~label], text_region_feats_hp[label], _curv) - torch.pi
                patch_class_angle = lorentz.oxy_angle(text_patch_feats_hp[label], text_patch_feats_hp[~label], _curv) + lorentz.oxy_angle(text_patch_feats_hp[~label], text_patch_feats_hp[label], _curv) - torch.pi
                
                exclusion_loss.append(self.compute_exclude_loss(text_feats_hp[~label], text_region_feats_hp[label], _curv, alpha = 1.0, beta = 0.8, class_angle=class_angle).view(-1).mean(dim=-1,keepdim=True)) 
                exclusion_loss.append(self.compute_exclude_loss(text_feats_hp[label], text_region_feats_hp[~label], _curv, alpha = 1.0, beta = 0.8, class_angle=class_angle).view(-1).mean(dim=-1,keepdim=True)) 
                
                exclusion_loss.append(self.compute_exclude_loss(text_region_feats_hp[~label], text_patch_feats_hp[label], _curv, alpha = 1.0, beta = 0.8, class_angle=region_class_angle).view(-1).mean(dim=-1,keepdim=True)) 
                exclusion_loss.append(self.compute_exclude_loss(text_region_feats_hp[label], text_patch_feats_hp[~label], _curv, alpha = 1.0, beta = 0.8, class_angle=region_class_angle).view(-1).mean(dim=-1,keepdim=True)) 
                
                exclusion_loss.append(self.compute_exclude_loss(text_patch_feats_hp[~label], slide_feats_hp, _curv, alpha = 1.0, beta = 0.8, class_angle=patch_class_angle).view(-1).mean(dim=-1,keepdim=True)) #这个需要大一点！
                exclusion_loss.append(self.compute_exclude_loss(text_region_feats_hp[~label], slide_feats_hp, _curv, alpha = 1.0, beta = 0.8, class_angle=region_class_angle).view(-1).mean(dim=-1,keepdim=True)) 
                exclusion_loss.append(self.compute_exclude_loss(text_feats_hp[~label], slide_feats_hp, _curv, alpha = 1.0, beta = 0.8, class_angle=class_angle).view(-1).mean(dim=-1,keepdim=True)) 
                
                
                exclusion_loss =  torch.cat(exclusion_loss,dim=0).mean()
                
                results_dict[f'contrastive_loss'] = 1*contrastive_loss
                results_dict[f'exclusion_loss'] = 1*exclusion_loss
                results_dict[f'entailment_loss'] = 10*entailment_loss
        
    
        x_out_hp = -lorentz.pairwise_dist(slide_feats_hp, text_feats_hp, _curv)
      
        self.logit_scale.data = torch.clamp(self.logit_scale.data, max=4.6052)
        _scale = self.logit_scale.exp()
      
        x_out = x_out_hp
        
        Y_hat = torch.argmax(x_out_hp) # : pred_label
        
        Y_prob = F.softmax(x_out_hp*_scale) # : prob
   
        return x_out, Y_hat, Y_prob, results_dict