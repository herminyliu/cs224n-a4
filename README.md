## CS224N-a4 2024 Spring Solution and Code Architecture Explanation

Stanford course CS224N assignment 4 coding part solution, **spring 2024**. For detailed infomation on the course and this assignment, see https://web.stanford.edu/class/cs224n/. 

# Code Overview

The assignment 4 coding part does the job of implementing a miniGPT model to perform question-answering task. It is not a general-purpose model which can answer a lot of quesitons, this model focus on and limited to answering the question on where was some famous people born.

# Pretrain Finetune Evaluation

In the *run.py* there are three mode: pretrain, finetune and evaluate. **Pretrain** part is the most effect comsuming part which use the data named *wiki.txt*. The txt file *wiki.txt* contains facts on wikipedia including where were some people born. **Finetune** part use the params obtained in the pretrain part(*vanilla.pretrain.params*) and use an additional data *birth_places_train.tsv* to finetune it. This file is more organised than *wiki.txt* and is in a question-answering format. **Evaluate** part use the params obtained in the finetune part(*vanilla.finetune.params*) and does the evaluation job on the data *birth_test_inputs*.

# Two different position embedding

The model has two ways of position-embedding: **vanilla** and **rope**.This refer to the different modes in the model. The former mode vanilla uses classic embedding and the latter mode rope use RoPE embedding, the rest of model detail remain the same. The pretrained/finetuned params file are also uploaded and named accordingly, which were trained on my local GPU. The accuracy of vanilla mode is over 19%, and the rope mode should over 30% according to the handout.

For the detail on the classic position embedding, refer to the part in the repository:
  if not config.rope:
      self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
  position_embeddings = self.pos_emb[:, :t, :]  # each position maps to a (learnable) vector
  x_input = token_embeddings + position_embeddings

For the detail on RoPE(Rotary Position Embedding), see the paper *RoFormer: Enhanced Transformer with Rotary Position Embedding* https://arxiv.org/abs/2104.09864.

# Remaining Fatal BUG

**Something is wrong on the rope position-embedding part, DO NOT implement directly(run the run.py file with rope option)**

The rope position-embedding part in *attention.py* must has something wrong, since vanilla part position embedding method works well but when switch to rope position embedding the output in evaluation part is garbled. There are already plenty of repositories about CS224N assignment solution on github. ***However, it seems the RoPE position embedding first came out on 2024 Spring since I have not found RoPE position embedding part on older repositories on github.*** If someone find way to fix my code, please kindly fork this repository and contact me. THANKS!

