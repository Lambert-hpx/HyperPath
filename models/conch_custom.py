import torch
from torch import nn
from conch.open_clip_custom import create_model_from_pretrained, tokenize, get_tokenizer
from torch.nn import functional as F

tokenizer = get_tokenizer()

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.ln_final.weight.dtype
        # self.attn_mask = clip_model.attn_mask

        # self.cls_emb = nn.Parameter(torch.empty(768))
        # nn.init.normal_(self.cls_emb, std=0.01)
        self.cls_emb = clip_model.cls_emb
        # self.conch_text = clip_model

        self.register_buffer('attn_mask', self.build_attention_mask(), persistent=False)
        # self.ctx = nn.Parameter(torch.Tensor(10, 10))

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(128, 128)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def _repeat(self, t, N: int):
        return t.reshape(1, 1, -1).repeat(N, 1, 1)
    
    def build_cls_mask(self, text, cast_dtype: torch.dtype):
        cls_mask = (text != 0).unsqueeze(1)
        cls_mask = F.pad(cls_mask, (1, 0, cls_mask.shape[2], 0), value=1.0)
        additive_mask = torch.empty(cls_mask.shape, dtype=cast_dtype, device=cls_mask.device)
        additive_mask.fill_(0)
        additive_mask.masked_fill_(~cls_mask, float("-inf"))
        additive_mask = torch.repeat_interleave(additive_mask, 12, 0)
        return additive_mask

    def forward(self, prompts, tokenized_prompts):
        ##
        prompts = prompts[:, :-1]
        seq_len = prompts.shape[1]
        attn_mask = self.attn_mask
        seq_len += 1

        prompts = torch.cat([prompts, self._repeat(self.cls_emb, prompts.shape[0])], dim=1)

        cls_mask = self.build_cls_mask(tokenized_prompts[:, :-1], self.dtype)
        attn_mask = attn_mask[None, :seq_len, :seq_len] + cls_mask[:, :seq_len, :seq_len]
        ##

        x = prompts + self.positional_embedding[:seq_len].type(self.dtype)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=attn_mask) #.last_hidden_state
        x = x.permute(1, 0, 2)  # LND -> NLD

        pooled, tokens = x[:, -1], x[:, :-1]
        pooled = self.ln_final(pooled)
        x = pooled @ self.text_projection

        x = F.normalize(x, dim=-1)
        # x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        
        # text = text[:, :-1]
        # text_latent, token_emb = self.conch_text(text)
        # x = F.normalize(text_latent, dim=-1)

        return x

# class TextEncoder(nn.Module):
#     def __init__(self, clip_model):
#         super().__init__()
#         self.text_model = clip_model

#     def forward(self, prompts, tokenized_prompts, normalize=True, embed_cls=True):
#         prompts = prompts[:, :-1] if embed_cls else prompts # make space for CLS token
#         text_latent, token_emb = self.text(text)
#         text_latent = F.normalize(text_latent, dim=-1) if normalize else text_latent
    
class PromptLearner(nn.Module):
    # def __init__(self, cfg, classnames, clip_model):
    def __init__(self, classnames, clip_model, n_ctx, n_flp=0, num_patch_prompt=0, is_shared=False):
        super().__init__()
        n_cls = len(classnames)
        # clip_model = clip_model.to('cuda')
        
        self.n_flp = n_flp
        self.num_patch_prompt = num_patch_prompt

        # ===============
        n_ctx = n_ctx # cfg.TRAINER.COOP.N_CTX
        # ctx_init = "" # cfg.TRAINER.COOP.CTX_INIT
        # ===============

        dtype = clip_model.ln_final.weight.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        # clip_imsize = 224

        # ===============
        # cfg_imsize = 224 #cfg.INPUT.SIZE[0]
        # ===============
        
        # assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if not is_shared:
            print("Initializing class-specific contexts")
            ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            if n_flp>0 and num_patch_prompt>0:
                flp_vectors = torch.empty(int(n_cls/num_patch_prompt)*n_flp, 75, ctx_dim, dtype=dtype)
            
        else:
            print("Initializing a generic context")
            if n_flp>0 and num_patch_prompt>0:
                flp_vectors = torch.empty(75, ctx_dim, dtype=dtype)
                
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            
        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * n_ctx)
        
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        # self.ctx2 = nn.Parameter(ctx_vectors)  # to be optimized
        
        classnames = [name.replace("_", " ") for name in classnames]
        # name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        name_lens = [(tokenize(texts=[name], tokenizer=tokenizer) > 0).sum() for name in classnames]
        # prompts = [prompt_prefix + " " + name + "." for name in classnames]
        prompts = [prompt_prefix + " " + name for name in classnames]
        

        tokenized_prompts = torch.cat([tokenize(texts=[p], tokenizer=tokenizer) for p in prompts]).to('cuda')
        
        with torch.no_grad():

            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        

        if n_flp>0 and num_patch_prompt>0:
            nn.init.normal_(flp_vectors, std=0.02)
            self.flp = nn.Parameter(flp_vectors)
            flp_prefix = " ".join(["X"] * 75)
            tokenized_flp = torch.cat([tokenize(texts=[flp_prefix+"."], tokenizer=tokenizer) for _ in range(int(n_cls/num_patch_prompt)*n_flp)]).to('cuda')
            
            with torch.no_grad():
                embedding_flp = clip_model.token_embedding(tokenized_flp).type(dtype)
            
            self.register_buffer("flp_token_prefix", embedding_flp[:, :1, :])  # SOS
            self.register_buffer("flp_token_suffix", embedding_flp[:, 1 + 75 :, :])  # CLS, EOS
            
            tokenized_prompts_ = []
            for i in range(n_cls):
                if i%num_patch_prompt==0:
                    cur_i_ = int(i/num_patch_prompt)
                    # print(tokenized_flp.shape)
                    tokenized_prompts_.append(tokenized_flp[cur_i_:cur_i_+n_flp])
                tokenized_prompts_.append(tokenized_prompts[i].unsqueeze(0))
            self.tokenized_prompts = torch.cat(tokenized_prompts_, dim=0).to('cuda')
            # print("aaaa")
        else:
            self.tokenized_prompts = tokenized_prompts
        
        
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.name_lens = name_lens
        
        # ===============
        self.class_token_position = "pre"
        # ===============

    def forward(self):
        ctx = self.ctx

        
        if self.n_flp>0 and self.num_patch_prompt>0:
            flp_prefix = self.flp_token_prefix
            flp_suffix = self.flp_token_suffix
            flp = self.flp
            if flp.dim() == 2:
                flp = flp.unsqueeze(0).expand(int(self.n_cls/self.num_patch_prompt)*self.n_flp, -1, -1)
            
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        half_n_ctx = self.n_ctx // 2
        
        # ==== 
        if self.class_token_position == "pre":
            prompts = []
            for i in range(self.n_cls):
                if self.n_flp>0 and self.num_patch_prompt>0:
                    if i%self.num_patch_prompt==0:

                        cur_i_ = int(i/self.num_patch_prompt)
                    
                        flp_i = flp[cur_i_: cur_i_+self.n_flp, :, :]
                        flp_prefix_i = flp_prefix[cur_i_: cur_i_+self.n_flp, :, :]
                        flp_suffix_i = flp_suffix[cur_i_: cur_i_+self.n_flp, :, :]
                        
                        prompt_flp = torch.cat(
                                [
                                    flp_prefix_i,
                                    flp_i,
                                    flp_suffix_i
                                ],
                                dim=1,
                                ) 
                        
                        prompts.append(
                            prompt_flp
                        )
                # tokenized_prompts.append(clip.tokenize(prompts[cur_i]))
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                # ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                # ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]

                prompt_i = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        # ctx_i_half1,
                        ctx_i,
                        class_i,   # (1, name_len, dim)
                        # ctx_i_half2,
                        # ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim) 
                    ],
                    dim=1,
                )
                prompts.append(prompt_i)
            prompts = torch.cat(prompts, dim=0)

        return prompts
