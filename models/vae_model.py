import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils.train_util import mean_with_lens, max_with_lens, generate_length_mask

from models.word_model import CaptionModel
from models.utils import repeat_tensor
import models.text_encoder as text_encoder

class VAEModel(CaptionModel):

    def __init__(self, Audioencoder: nn.Module, Textdecoder: nn.Module, **kwargs):
        super().__init__(Audioencoder, Textdecoder, **kwargs)
        self.qnet = getattr(text_encoder, kwargs["posterior_model"])(
            word_dim = Textdecoder.embed_size,
            embed_size = Textdecoder.embed_size,
            vocab_size = Textdecoder.vocab_size,
            **kwargs["posterior_args"]
        )
        self.pnet = getattr(text_encoder, kwargs["prior_model"])(
            word_dim = Textdecoder.embed_size,
            audiofeats_size = Textdecoder.embed_size,
            embed_size = Textdecoder.embed_size,
            vocab_size= Textdecoder.vocab_size,
            **kwargs["prior_args"]
        )
        
        if Textdecoder.embed_size !=Audioencoder.embed_size:
            self.ln = nn.Linear(Audioencoder.embed_size,Textdecoder.embed_size)
            nn.init.xavier_uniform_(self.ln.weight)
    
    def stepwise_forward(self, encoded, caps, cap_lens,**kwargs):
        """Step-by-step decoding"""
        if cap_lens is not None: # scheduled sampling training
            max_length = max(cap_lens) - 1
        else: # inference
            max_length = kwargs.get("max_length", self.max_length)
        decoder_input = {}
        output = {}
        self.prepare_output(encoded, output, max_length)
        # start sampling
        for t in range(max_length):
            self.decode_step(decoder_input, encoded, caps, output, t, **kwargs)
            if caps is None: # decide whether to stop when sampling
                unfinished_t = output["seqs"][:, t] != self.end_idx
                if t == 0:
                    unfinished = unfinished_t
                else:
                    unfinished *= unfinished_t
                output["seqs"][:, t][~unfinished] = self.end_idx
                if unfinished.sum() == 0:
                    break
        return output

    def forward(self, *input, **kwargs):
            """
            an encoder first encodes audio feature into an embedding sequence, obtaining `encoded`: {
                audio_embeds: [N, enc_mem_max_len, enc_mem_size]
                state: rnn style hidden states, [num_dire * num_layers, N, hs_enc]
                audio_embeds_lens: [N,] 
            }
            """
            if len(input) == 4:
                feats, feat_lens, caps, cap_lens = input
                encoded = self.encoder(feats, feat_lens)
                if self.encoder.embed_size!=  self.decoder.embed_size:
                    encoded["audio_embeds"]=self.ln(encoded["audio_embeds"])
     
                qnetout = self.qnet(caps,cap_lens,encoded["audio_embeds"],encoded["audio_embeds_lens"]) 
                qnetout["q_means"] = qnetout["q_means"].to(encoded["audio_embeds"].device)
                qnetout["q_logs"] = qnetout["q_logs"].to(encoded["audio_embeds"].device)
                qnetout["q_z"] = qnetout["q_z"].to(encoded["audio_embeds"].device)
                encoded.update(qnetout)
                output = self.train_forward(encoded, caps, cap_lens, **kwargs)
            elif len(input) == 2:
                feats, feat_lens = input
                encoded = self.encoder(feats, feat_lens)
                if self.encoder.embed_size!=  self.decoder.embed_size:
                    encoded["audio_embeds"]= self.ln(encoded["audio_embeds"])
                output = self.inference_forward(encoded, **kwargs)
            else:
                raise Exception("Number of input should be either 4 (feats, feat_lens, caps, cap_lens) or 2 (feats, feat_lens)")

            return output

    def prepare_output(self, encoded, output, max_length):
        N = encoded["audio_embeds"].size(0)
        output["seqs"] = torch.empty(N, max_length, dtype=torch.long).fill_(self.end_idx)
        output["logits"] = torch.empty(N, max_length, self.vocab_size).to(encoded["audio_embeds"].device)

        output["outputs"] = torch.empty(output["seqs"].size(0), max_length, self.decoder.embed_size).to(encoded["audio_embeds"].device)
        output["sampled_logprobs"] = torch.zeros(output["seqs"].size(0), max_length)
        
        
        output["attn_weights"] = torch.empty(output["seqs"].size(0), max(encoded["audio_embeds_lens"]), max_length)

        if  hasattr(self.pnet,'gmm_kernel'):
            output["p_means"] = torch.empty(self.pnet.gmm_kernel,output["seqs"].size(0),max_length,self.pnet.embed_size)
            output["p_logs"] = torch.empty(self.pnet.gmm_kernel,output["seqs"].size(0),max_length,self.pnet.embed_size)
            
        else:
            output["p_means"] = torch.empty(output["seqs"].size(0),max_length,self.pnet.embed_size)
            output["p_logs"] = torch.empty(output["seqs"].size(0),max_length,self.pnet.embed_size)


        output["p_z"] = torch.empty(output["seqs"].size(0),max_length,self.pnet.embed_size)
        output["rnn_input"] = torch.empty(output["seqs"].size(0),max_length,3*self.pnet.embed_size)
        if "q_means" in encoded:
            output["q_means"] = encoded["q_means"]
            output["q_logs"] = encoded["q_logs"]
            output["q_z"] = encoded["q_z"]
        return output
  
    def decode_step(self,  decoder_input, encoded, caps, output, t, **kwargs):
        """Decoding operation of timestep t"""

        self.prepare_decoder_input(decoder_input, encoded, caps, output, t, **kwargs)
        
        output_pnet_t = self.pnet(decoder_input["word"],decoder_input["enc_mem"],decoder_input["hiddens_state"],decoder_input["last_z"],decoder_input["enc_mem_lens"])
        
        #prepare decoder
        if caps is not None:
            decoder_input["z"] = output["q_z"][:,t,:]
            if kwargs["dis_ratio"] ==0:
                decoder_input["z"] = output["q_z"][:,t,:]
            elif torch.rand(1)<= kwargs["dis_ratio"]:
                decoder_input["z"] = output_pnet_t["z"]
        else: 
            decoder_input["z"] = output_pnet_t["z"]

        output_t = self.decoder(**decoder_input)
        output_t.update(output_pnet_t)

        logits_t = output_t["logits"].squeeze(1)
        sampled = self.sample_next_word(logits_t,**kwargs)
        
        self.stepwise_process_step(output, output_t, t, sampled)

    def prepare_decoder_input(self, decoder_input, encoded, caps, output, t, **kwargs):

        decoder_input["enc_mem"] = encoded["audio_embeds"]
        decoder_input["enc_mem_lens"] = encoded["audio_embeds_lens"]
        decoder_input["audio_embeds_pooled"] = encoded["audio_embeds_pooled"]
        ###############
        # determine input word
        ################
        if caps is not None and random.random() < kwargs["ss_ratio"]: # training, scheduled sampling
            word = caps[:, t].long()
        else:
            if t == 0:
                word = torch.tensor([self.start_idx,] * output["seqs"].size(0)).long()
            else:
                word = output["seqs"][:, t-1]
        # word: [N,]
        decoder_input["word"] = word.unsqueeze(1)

        ################
        # prepare rnn state
        ################
        if t == 0:
            decoder_input["state"] = self.decoder.init_hidden(output["seqs"].size(0)).to(encoded["audio_embeds"].device)
            decoder_input["hiddens_state"] = self.pnet.init_hidden(output["seqs"].size(0),encoded["audio_embeds"].device)
            decoder_input["last_z"] = torch.zeros(output["seqs"].size(0),self.decoder.embed_size).to(encoded["audio_embeds"].device)
        if t > 0:
            decoder_input["state"] = output["state"]
            decoder_input["hiddens_state"] = output["hiddens_state"]
            decoder_input["last_z"] = output["last_z"]

        return decoder_input

    def stepwise_process_step(self, output, output_t, t, sampled):

        output["logits"][:, t, :] = output_t["logits"].squeeze(1)
        output["outputs"][:, t, :] = output_t["output"].squeeze(1)
        output["seqs"][:, t] = sampled["w_t"]
        output["sampled_logprobs"][:, t] = sampled["probs"]

        if len(output["p_means"].shape) == 4:
            output["p_means"][:,:, t, :] = output_t["mean"].squeeze(1)
            output["p_logs"][:,:, t, :] = output_t["log"].squeeze(1)
        else:
            output["p_means"][:, t, :] = output_t["mean"].squeeze(1)
            output["p_logs"][:,t, :] = output_t["log"].squeeze(1)
        output["p_z"][:, t, :] = output_t["z"].squeeze(1)
        output["rnn_input"] [:, t, :] = output_t["rnn_input"].squeeze(1)
        if "state" in output_t:
            output["state"] = output_t["state"]
            output["hiddens_state"] = output_t["hiddens_state"]
            output["attn_weights"][:, :, t] = output_t["weights"]
        output["last_z"] = output_t['z']
        
    def train_forward(self, encoded, caps, cap_lens,**kwargs):
        # Bahdanau attention only supports step-by-step implementation, so we implement forward in 
        # step-by-step manner whether in training or evaluation
        output = self.stepwise_forward(encoded, caps, cap_lens,**kwargs)
        if "embedding_lens" in kwargs and kwargs["embedding_lens"] !=self.decoder.model.hidden_size :
            output["outputs"] = self.output_transform(output["outputs"])
        
        return output

    def inference_forward(self, encoded, **kwargs):
        # optional sampling keyword arguments
        method = kwargs.get("method", "greedy")
        # print(method)
        max_length = kwargs.get("max_length", self.max_length)
        if method == "beam":
            beam_size = kwargs.get("beam_size", 3)
            return self.beam_search(encoded, max_length, beam_size)
        elif method == "dbs":
            beam_size = kwargs.get("beam_size", 5)
            group_size = kwargs.get("group_size",5)
            diversity_lambda =  kwargs.get("diversity_lambda",0.5)
            temperature = kwargs.get('temperature', 1.0)
            group_nbest = kwargs.get("group_nbest",True)
            return self.diverse_beam_search(encoded, max_length, beam_size,group_size,diversity_lambda,temperature,group_nbest)
        return self.stepwise_forward(encoded, None, None, **kwargs)

    def beam_search(self, encoded, max_length, beam_size):
        output = {}
        self.prepare_output(encoded, output, max_length)
 
        # instance by instance beam seach
        for i in range(encoded["audio_embeds"].size(0)):
            output_i = {}
            self.prepare_beamsearch_output(output_i, beam_size, encoded, max_length)
            decoder_input = {}
            for t in range(max_length):
                if len(output_i["done_beams"]) >= beam_size:
                    break
                output_t = self.beamsearch_step(decoder_input, encoded, output_i, i, t, beam_size)

                logits_t = output_t["logits"].squeeze(1)
                logprobs_t = torch.log_softmax(logits_t, dim=1)
                logprobs_t = output_i["top_k_logprobs"].unsqueeze(1).expand_as(logprobs_t) + logprobs_t
                top_k_logprobs, top_k_words = logprobs_t.view(-1).topk(beam_size, 0, True, True)
                output_i["top_k_logprobs"] = top_k_logprobs
                
                output_i["prev_word_inds"] = torch.div(top_k_words, self.vocab_size, rounding_mode='trunc')
                output_i["next_word_inds"] = top_k_words % self.vocab_size  # [beam_size,]
                if t == 0:
                    output_i["seqs"] = output_i["next_word_inds"].unsqueeze(1)
                else:
                    output_i["seqs"] = torch.cat([output_i["seqs"][output_i["prev_word_inds"]], 
                                                  output_i["next_word_inds"].unsqueeze(1)], dim=1)

                self.beamsearch_process_step(output_i, output_t)
     
            self.beamsearch_process(output, output_i, i)
            
        return output
        
    def prepare_beamsearch_output(self, output, beam_size, encoded, max_length):
        output["top_k_logprobs"] = torch.zeros(beam_size).to(encoded["audio_embeds"].device)

        output["attn_weights"] = torch.empty(beam_size, max(encoded["audio_embeds_lens"]), max_length)
        
        output["p_means"] = torch.empty(beam_size,max_length,self.pnet.embed_size)
        output["p_logs"] = torch.empty(beam_size,max_length,self.pnet.embed_size)
        output["p_z"] = torch.empty(beam_size,max_length,self.pnet.embed_size)
    
        output["done_beams"] = []
    def beamsearch_step(self, decoder_input, encoded, output, i, t, beam_size):
        self.prepare_beamsearch_decoder_input(decoder_input, encoded, output, i, t, beam_size)
        output_pnet_t = self.pnet(decoder_input["word"],decoder_input["enc_mem"],decoder_input["hiddens_state"],decoder_input["last_z"],decoder_input["enc_mem_lens"])
        decoder_input["z"] = output_pnet_t["z"]
        output_t = self.decoder(**decoder_input)
        output_t.update(output_pnet_t)
        if "weights" in output_t:
            output["attn_weights"][:, :, t] = output_t["weights"]
        return output_t

    def prepare_beamsearch_decoder_input(self, decoder_input, encoded, output, i, t, beam_size):
        
        if t == 0:
            decoder_input["enc_mem"] =  encoded["audio_embeds"][i].unsqueeze(0).repeat(beam_size, 1, 1)
            decoder_input["enc_mem_lens"] = encoded["audio_embeds_lens"][i].repeat(beam_size)
            decoder_input["audio_embeds_pooled"] = encoded["audio_embeds_pooled"][i].unsqueeze(0).repeat(beam_size, 1)

            decoder_input["state"] = self.decoder.init_hidden(beam_size).to(encoded["audio_embeds"].device)
            decoder_input["hiddens_state"] = self.pnet.init_hidden(beam_size,encoded["audio_embeds"].device)
            decoder_input["last_z"] = torch.zeros(beam_size,self.decoder.embed_size).to(encoded["audio_embeds"].device)

            w_t = torch.tensor([self.start_idx,] * beam_size).long()
        else:
            w_t = output["next_word_inds"]

            decoder_input["state"] = output["state"][:, output["prev_word_inds"], :].contiguous()
            if output["hiddens_state"] != None:
                hiddens_state_0 = output["hiddens_state"][0][:, output["prev_word_inds"], :].contiguous()
                hiddens_state_1 = output["hiddens_state"][1][:, output["prev_word_inds"], :].contiguous()
                decoder_input["hiddens_state"] =(hiddens_state_0,hiddens_state_1)
            decoder_input["last_z"] = output["last_z"][output["prev_word_inds"], :].contiguous()

        decoder_input["word"] = w_t.unsqueeze(1)

    def beamsearch_process_step(self, output, output_t):
        
        if "state" in output_t:
            output["state"] = output_t["state"]
            output["hiddens_state"] = output_t["hiddens_state"]
            output["attn_weights"] = output["attn_weights"][output["prev_word_inds"], :, :]
        
        output["last_z"] = output_t["z"]
        
    
    def beamsearch_process(self,  output, output_i, i):
        output["seqs"][i] = output_i["seqs"][0].unsqueeze(0)
        
        if "attn_weights" in output_i:
            output["attn_weights"][i] = output_i["attn_weights"][0]# done_beams = sorted(output_i["done_beams"], key=lambda x: -x["score"])

    def prepare_dbs_decoder_input(self,encoded,decoder_input,t,i,bdash,divm, output_i):
        local_time = t - divm
        if t == 0:
            decoder_input["enc_mem"] =  encoded["audio_embeds"][i].unsqueeze(0).repeat(bdash, 1, 1)
            decoder_input["enc_mem_lens"] = encoded["audio_embeds_lens"][i].repeat(bdash)
            decoder_input["audio_embeds_pooled"] = encoded["audio_embeds_pooled"][i].unsqueeze(0).repeat(bdash, 1)
        ###############
        # determine input word
        ################
        if local_time == 0:
            word = torch.tensor([self.start_idx,] * bdash).long()
            decoder_input["state"] = self.decoder.init_hidden(bdash).to(encoded["audio_embeds"].device)
            decoder_input["hiddens_state"] = self.pnet.init_hidden(bdash,encoded["audio_embeds"].device)
            decoder_input["last_z"] = torch.zeros(bdash,self.decoder.embed_size).to(encoded["audio_embeds"].device)
        else:
            word = output_i["next_word"][divm]
        decoder_input["word"] = word.unsqueeze(1)

        if local_time > 0:
 
            decoder_input["state"] = output_i["state"][divm][
                :, output_i["prev_words_beam"][divm], :].contiguous()
            hiddens_state_0 = output_i["hiddens_state"][divm][0][:, output_i["prev_words_beam"][divm], :].contiguous()
            hiddens_state_1 = output_i["hiddens_state"][divm][1][:, output_i["prev_words_beam"][divm], :].contiguous()
            decoder_input["hiddens_state"] =(hiddens_state_0,hiddens_state_1)
            decoder_input["last_z"] = output_i["last_z"][divm][output_i["prev_words_beam"][divm], :].contiguous()
         
        return decoder_input
    def dbs_process_step(self, output_i, output_t):
        divm = output_t["divm"]
        output_i["state"][divm] = output_t["state"]
        output_i["hiddens_state"][divm] = output_t["hiddens_state"]
        output_i["last_z"][divm] = output_t['z']

    def dbs_step(self, encoded, decoder_input,output_i, i, t, bdash,divm):
        
        self.prepare_dbs_decoder_input(encoded,decoder_input,t,i,bdash,divm, output_i)
        output_pnet_t = self.pnet(decoder_input["word"],decoder_input["enc_mem"],decoder_input["hiddens_state"],decoder_input["last_z"],decoder_input["enc_mem_lens"])
        decoder_input["z"] = output_pnet_t["z"]
        output_t = self.decoder(**decoder_input)
        output_t.update(output_pnet_t)

        return output_t

    def prepare_dbs_output(self,output_i, bdash, encoded,group_size, max_length):
        output_i["prev_words_beam"] = [None for _ in range(group_size)]
        output_i["next_word"] = [None for _ in range(group_size)]
        output_i["state"] = [None for _ in range(group_size)]
        output_i["hiddens_state"] = [None for _ in range(len(output_i["prev_words_beam"]))]
        output_i["last_z"] = [None for _ in range(len(output_i["prev_words_beam"]))]
        output_i["attn_weights"] = [torch.empty(bdash, max(encoded["audio_embeds_lens"]), max_length) for _ in range(len(output_i["prev_words_beam"]))]

class SimpleVAEModel(CaptionModel):

    def __init__(self, Audioencoder: nn.Module, Textdecoder: nn.Module, **kwargs):
        super().__init__(Audioencoder, Textdecoder, **kwargs)
        self.qnet = getattr(text_encoder, kwargs["posterior_model"])(
            word_dim = Textdecoder.embed_size,
            embed_size = Textdecoder.embed_size,
            vocab_size = Textdecoder.vocab_size,
            **kwargs["posterior_args"]
        )

        if Textdecoder.embed_size !=Audioencoder.embed_size:
            self.ln = nn.Linear(Audioencoder.embed_size,Textdecoder.embed_size)
            nn.init.xavier_uniform_(self.ln.weight)
    
    def stepwise_forward(self, encoded, caps, cap_lens,**kwargs):
        """Step-by-step decoding"""
        if cap_lens is not None: # scheduled sampling training
            max_length = max(cap_lens) - 1
        else: # inference
            max_length = kwargs.get("max_length", self.max_length)
        decoder_input = {}
        output = {}
        self.prepare_output(encoded, output, max_length)
        # start sampling
        for t in range(max_length):
            self.decode_step(decoder_input, encoded, caps, output, t, **kwargs)
            if caps is None: # decide whether to stop when sampling
                unfinished_t = output["seqs"][:, t] != self.end_idx
                if t == 0:
                    unfinished = unfinished_t
                else:
                    unfinished *= unfinished_t
                output["seqs"][:, t][~unfinished] = self.end_idx
                if unfinished.sum() == 0:
                    break

        return output

    def forward(self, *input, **kwargs):
            """
            an encoder first encodes audio feature into an embedding sequence, obtaining `encoded`: {
                audio_embeds: [N, enc_mem_max_len, enc_mem_size]
                state: rnn style hidden states, [num_dire * num_layers, N, hs_enc]
                audio_embeds_lens: [N,] 
            }
            """
            if len(input) == 4:
                feats, feat_lens, caps, cap_lens = input
                encoded = self.encoder(feats, feat_lens)
                if self.encoder.embed_size!=  self.decoder.embed_size:
                    encoded["audio_embeds"]=self.ln(encoded["audio_embeds"])
                qnetout = self.qnet(caps,cap_lens)
                encoded.update(qnetout)
                output = self.train_forward(encoded, caps, cap_lens, **kwargs)
            elif len(input) == 2:
                feats, feat_lens = input
                encoded = self.encoder(feats, feat_lens)
                if self.encoder.embed_size!=  self.decoder.embed_size:
                    encoded["audio_embeds"]= self.ln(encoded["audio_embeds"])
                output = self.inference_forward(encoded, **kwargs)
            else:
                raise Exception("Number of input should be either 4 (feats, feat_lens, caps, cap_lens) or 2 (feats, feat_lens)")

            return output

    def prepare_output(self, encoded, output, max_length):
        N = encoded["audio_embeds"].size(0)
        output["seqs"] = torch.empty(N, max_length, dtype=torch.long).fill_(self.end_idx)
        output["logits"] = torch.empty(N, max_length, self.vocab_size).to(encoded["audio_embeds"].device)

        output["outputs"] = torch.empty(output["seqs"].size(0), max_length, self.decoder.embed_size).to(encoded["audio_embeds"].device)
        output["sampled_logprobs"] = torch.zeros(output["seqs"].size(0), max_length)
        
        
        output["attn_weights"] = torch.empty(output["seqs"].size(0), max(encoded["audio_embeds_lens"]), max_length)

        output["p_means"] = torch.empty(output["seqs"].size(0),self.decoder.embed_size)
        output["p_logs"] = torch.empty(output["seqs"].size(0),self.decoder.embed_size)
        output["p_z"] = torch.empty(output["seqs"].size(0),self.decoder.embed_size)
        
        if "q_means" in encoded:
            output["q_means"] = encoded["q_means"]
            output["q_logs"] = encoded["q_logs"]
            output["q_z"] = encoded["q_z"]
        return output
  
    def decode_step(self,  decoder_input, encoded, caps, output, t, **kwargs):
        """Decoding operation of timestep t"""

        self.prepare_decoder_input(decoder_input, encoded, caps, output, t, **kwargs)
        #prepare decoder
        if caps is not None:
            decoder_input["z"] = output["q_z"]

        output_t = self.decoder(**decoder_input)

        logits_t = output_t["logits"].squeeze(1)
        sampled = self.sample_next_word(logits_t,**kwargs)
        
        self.stepwise_process_step(output, output_t, t, sampled)

    def prepare_decoder_input(self, decoder_input, encoded, caps, output, t, **kwargs):

        decoder_input["enc_mem"] = encoded["audio_embeds"]
        decoder_input["enc_mem_lens"] = encoded["audio_embeds_lens"]
        decoder_input["audio_embeds_pooled"] = encoded["audio_embeds_pooled"]
        ###############
        # determine input word
        ################
        if caps is not None and random.random() < kwargs["ss_ratio"]: # training, scheduled sampling
            word = caps[:, t].long()
        else:
            if t == 0:
                word = torch.tensor([self.start_idx,] * output["seqs"].size(0)).long()
            else:
                word = output["seqs"][:, t-1]
        decoder_input["word"] = word.unsqueeze(1)

        ################
        # prepare rnn state
        ################
        if t == 0:
            decoder_input["state"] = self.decoder.init_hidden(output["seqs"].size(0)).to(encoded["audio_embeds"].device)

            output["p_means"] = torch.zeros(encoded["audio_embeds"].size(0), self.encoder.embed_size).to(encoded["audio_embeds"].device)
            output["p_logs"] = torch.full((encoded["audio_embeds"].size(0),  self.encoder.embed_size),0.5).to(encoded["audio_embeds"].device)
            z = torch.randn(output["p_means"].shape).to(encoded["audio_embeds"].device) * torch.exp(.5 * output["p_logs"]) + output["p_means"]
            output["p_z"] = z
            decoder_input["z"] =  output["p_z"]

        if t > 0:
            decoder_input["state"] = output["state"]

        return decoder_input

    def stepwise_process_step(self, output, output_t, t, sampled):

        output["logits"][:, t, :] = output_t["logits"].squeeze(1)
        output["outputs"][:, t, :] = output_t["output"].squeeze(1)
        output["seqs"][:, t] = sampled["w_t"]
        output["sampled_logprobs"][:, t] = sampled["probs"]

        if "state" in output_t:
            output["state"] = output_t["state"]
            output["attn_weights"][:, :, t] = output_t["weights"]
        
    def train_forward(self, encoded, caps, cap_lens,**kwargs):
        # Bahdanau attention only supports step-by-step implementation, so we implement forward in 
        # step-by-step manner whether in training or evaluation
        output = self.stepwise_forward(encoded, caps, cap_lens,**kwargs)
        if "embedding_lens" in kwargs and kwargs["embedding_lens"] !=self.decoder.model.hidden_size :
            output["outputs"] = self.output_transform(output["outputs"])
        
        return output

    def inference_forward(self, encoded, **kwargs):
        # optional sampling keyword arguments
        method = kwargs.get("method", "greedy")
        max_length = kwargs.get("max_length", self.max_length)
        if method == "beam":
            beam_size = kwargs.get("beam_size", 3)
            return self.beam_search(encoded, max_length, beam_size)
        elif method == "dbs":
            beam_size = kwargs.get("beam_size", 5)
            group_size = kwargs.get("group_size",5)
            diversity_lambda =  kwargs.get("diversity_lambda",0.5)
            temperature = kwargs.get('temperature', 1.0)
            group_nbest = kwargs.get("group_nbest",True)
            return self.diverse_beam_search(encoded, max_length, beam_size,group_size,diversity_lambda,temperature,group_nbest)
        return self.stepwise_forward(encoded, None, None, **kwargs)

    def beam_search(self, encoded, max_length, beam_size):
        output = {}
        self.prepare_output(encoded, output, max_length)
  
        # instance by instance beam seach
        for i in range(encoded["audio_embeds"].size(0)):
            output_i = {}
            self.prepare_beamsearch_output(output_i, beam_size, encoded, max_length)
            decoder_input = {}
            for t in range(max_length):
                output_t = self.beamsearch_step(decoder_input, encoded, output_i, i, t, beam_size)
                logits_t = output_t["logits"].squeeze(1)
                logprobs_t = torch.log_softmax(logits_t, dim=1)
                logprobs_t = output_i["top_k_logprobs"].unsqueeze(1).expand_as(logprobs_t) + logprobs_t

                top_k_logprobs, top_k_words = logprobs_t.view(-1).topk(beam_size, 0, True, True)
                output_i["top_k_logprobs"] = top_k_logprobs
                
                output_i["prev_word_inds"] = torch.div(top_k_words, self.vocab_size, rounding_mode='floor')
                output_i["next_word_inds"] = top_k_words % self.vocab_size  # [beam_size,]
                if t == 0:
                    output_i["seqs"] = output_i["next_word_inds"].unsqueeze(1)
                else:
                    output_i["seqs"] = torch.cat([output_i["seqs"][output_i["prev_word_inds"]], 
                                                  output_i["next_word_inds"].unsqueeze(1)], dim=1)
                self.beamsearch_process_step(output_i, output_t)
            self.beamsearch_process(output, output_i, i)
        return output
        
    def prepare_beamsearch_output(self, output, beam_size, encoded, max_length):
        output["top_k_logprobs"] = torch.zeros(beam_size).to(encoded["audio_embeds"].device)

        output["attn_weights"] = torch.empty(beam_size, max(encoded["audio_embeds_lens"]), max_length)
        
        output["p_means"] = torch.empty(beam_size,self.decoder.embed_size)
        output["p_logs"] = torch.empty(beam_size,self.decoder.embed_size)
        output["p_z"] = torch.empty(beam_size,self.decoder.embed_size)
    
    
    def beamsearch_step(self, decoder_input, encoded, output, i, t, beam_size):
        self.prepare_beamsearch_decoder_input(decoder_input, encoded, output, i, t, beam_size)
        output_t = self.decoder(**decoder_input)
        if "weights" in output_t:
            output["attn_weights"][:, :, t] = output_t["weights"]
        return output_t

    def prepare_beamsearch_decoder_input(self, decoder_input, encoded, output, i, t, beam_size):
        
        if t == 0:
            decoder_input["enc_mem"] =  encoded["audio_embeds"][i].unsqueeze(0).repeat(beam_size, 1, 1)
            decoder_input["enc_mem_lens"] = encoded["audio_embeds_lens"][i].repeat(beam_size)
            decoder_input["audio_embeds_pooled"] = encoded["audio_embeds_pooled"][i].unsqueeze(0).repeat(beam_size, 1)

            decoder_input["state"] = self.decoder.init_hidden(beam_size).to(encoded["audio_embeds"].device)
            
            output["p_means"] = torch.zeros(beam_size, self.encoder.embed_size).to(encoded["audio_embeds"].device)
            output["p_logs"] = torch.full((beam_size,  self.encoder.embed_size),1).to(encoded["audio_embeds"].device)
            output["p_z"] = torch.randn(output["p_means"].shape).to(encoded["audio_embeds"].device) * torch.exp(.5 * output["p_logs"]) + output["p_means"]

            decoder_input["z"] =  output["p_z"]


            w_t = torch.tensor([self.start_idx,] * beam_size).long()
        else:
            w_t = output["next_word_inds"]

            decoder_input["state"] = output["state"][:, output["prev_word_inds"], :].contiguous()

        decoder_input["word"] = w_t.unsqueeze(1)

    def beamsearch_process_step(self, output, output_t):
        
        if "state" in output_t:
            output["state"] = output_t["state"]
            output["attn_weights"] = output["attn_weights"][output["prev_word_inds"], :, :]
        
    
    def beamsearch_process(self,  output, output_i, i):
        output["seqs"][i] = output_i["seqs"][0].unsqueeze(0)

        if "attn_weights" in output_i:
            output["attn_weights"][i] = output_i["attn_weights"][0]

    def prepare_dbs_decoder_input(self,encoded,decoder_input,t,i,bdash,divm, output_i):
        local_time = t - divm
        # print(local_time)
        if t == 0:
            decoder_input["enc_mem"] =  encoded["audio_embeds"][i].unsqueeze(0).repeat(bdash, 1, 1)
            decoder_input["enc_mem_lens"] = encoded["audio_embeds_lens"][i].repeat(bdash)
            decoder_input["audio_embeds_pooled"] = encoded["audio_embeds_pooled"][i].unsqueeze(0).repeat(bdash, 1)
        ###############
        # determine input word
        ################
        if local_time == 0:
            word = torch.tensor([self.start_idx,] * bdash).long()
            decoder_input["state"] = self.decoder.init_hidden(bdash).to(encoded["audio_embeds"].device)
            decoder_input["hiddens_state"] = self.pnet.init_hidden(bdash,encoded["audio_embeds"].device)
            decoder_input["last_z"] = torch.zeros(bdash,self.decoder.embed_size).to(encoded["audio_embeds"].device)
        else:
            word = output_i["next_word"][divm]
        decoder_input["word"] = word.unsqueeze(1)

        if local_time > 0:
 
            decoder_input["state"] = output_i["state"][divm][
                :, output_i["prev_words_beam"][divm], :].contiguous()
            hiddens_state_0 = output_i["hiddens_state"][divm][0][:, output_i["prev_words_beam"][divm], :].contiguous()
            hiddens_state_1 = output_i["hiddens_state"][divm][1][:, output_i["prev_words_beam"][divm], :].contiguous()
            decoder_input["hiddens_state"] =(hiddens_state_0,hiddens_state_1)
            decoder_input["last_z"] = output_i["last_z"][divm][output_i["prev_words_beam"][divm], :].contiguous()
         
        return decoder_input
    def dbs_process_step(self, output_i, output_t):
        divm = output_t["divm"]
        output_i["state"][divm] = output_t["state"]
        output_i["hiddens_state"][divm] = output_t["hiddens_state"]
        output_i["last_z"][divm] = output_t['z']

    def dbs_step(self, encoded, decoder_input,output_i, i, t, bdash,divm):
        
        self.prepare_dbs_decoder_input(encoded,decoder_input,t,i,bdash,divm, output_i)
        output_pnet_t = self.pnet(decoder_input["word"],decoder_input["enc_mem"],decoder_input["hiddens_state"],decoder_input["last_z"],decoder_input["enc_mem_lens"])
        decoder_input["z"] = output_pnet_t["z"]
        output_t = self.decoder(**decoder_input)
        output_t.update(output_pnet_t)

        return output_t

    def prepare_dbs_output(self,output_i, bdash, encoded,group_size, max_length):
        output_i["prev_words_beam"] = [None for _ in range(group_size)]
        output_i["next_word"] = [None for _ in range(group_size)]
        output_i["state"] = [None for _ in range(group_size)]
        output_i["hiddens_state"] = [None for _ in range(len(output_i["prev_words_beam"]))]
        output_i["last_z"] = [None for _ in range(len(output_i["prev_words_beam"]))]
        output_i["attn_weights"] = [torch.empty(bdash, max(encoded["audio_embeds_lens"]), max_length) for _ in range(len(output_i["prev_words_beam"]))]

class Hybrid_VAEModel(CaptionModel):

    def __init__(self, Audioencoder: nn.Module, Textdecoder: nn.Module, **kwargs):
        super().__init__(Audioencoder, Textdecoder, **kwargs)
        self.qnet = getattr(text_encoder, kwargs["posterior_model"])(
            word_dim = Textdecoder.embed_size,
            embed_size = Textdecoder.embed_size,
            vocab_size = Textdecoder.vocab_size,
            **kwargs["posterior_args"]
        )

        self.pnet = getattr(text_encoder, kwargs["prior_model"])(
            word_dim = Textdecoder.embed_size,
            audiofeats_size = Textdecoder.embed_size,
            embed_size = Textdecoder.embed_size,
            vocab_size= Textdecoder.vocab_size,
            **kwargs["prior_args"]
        )
        
        self.mean_log_out = nn.Linear(Textdecoder.embed_size, 2*Textdecoder.embed_size)
        
        if Textdecoder.embed_size !=Audioencoder.embed_size:
            self.ln = nn.Linear(Audioencoder.embed_size,Textdecoder.embed_size)
            nn.init.xavier_uniform_(self.ln.weight)
        nn.init.xavier_uniform_(self.mean_log_out.weight)

    def stepwise_forward(self, encoded, caps, cap_lens,**kwargs):
        """Step-by-step decoding"""
        if cap_lens is not None: # scheduled sampling training
            max_length = max(cap_lens) - 1
        else: # inference
            max_length = kwargs.get("max_length", self.max_length)
        decoder_input = {}
        output = {}
        self.prepare_output(encoded, output, max_length)
        # start sampling
        for t in range(max_length):
            self.decode_step(decoder_input, encoded, caps, output, t, **kwargs)
            if caps is None: # decide whether to stop when sampling
                unfinished_t = output["seqs"][:, t] != self.end_idx
                if t == 0:
                    unfinished = unfinished_t
                else:
                    unfinished *= unfinished_t
                output["seqs"][:, t][~unfinished] = self.end_idx
                if unfinished.sum() == 0:
                    break

        if caps is not None:
            hidden_mean = mean_with_lens(output["outputs"], cap_lens-1)
            hidden_max = max_with_lens(output["outputs"], cap_lens-1)
            hidden = hidden_mean + hidden_max
            mean_log_out = self.mean_log_out(hidden)
            
            output["p_means_utt"] =  mean_log_out
            output["p_logs_utt"] = None
        return output

    def forward(self, *input, **kwargs):
            """
            an encoder first encodes audio feature into an embedding sequence, obtaining `encoded`: {
                audio_embeds: [N, enc_mem_max_len, enc_mem_size]
                state: rnn style hidden states, [num_dire * num_layers, N, hs_enc]
                audio_embeds_lens: [N,] 
            }
            """
            if len(input) == 4:
                feats, feat_lens, caps, cap_lens = input
                encoded = self.encoder(feats, feat_lens)
                if self.encoder.embed_size!=  self.decoder.embed_size:
                    encoded["audio_embeds"]=self.ln(encoded["audio_embeds"])
                qnetout = self.qnet(caps,cap_lens) 
                qnetout["q_means"] = qnetout["q_means"].to(encoded["audio_embeds"].device)
                qnetout["q_logs"] = qnetout["q_logs"].to(encoded["audio_embeds"].device)
                qnetout["q_z"] = qnetout["q_z"].to(encoded["audio_embeds"].device)
                encoded.update(qnetout)
                output = self.train_forward(encoded, caps, cap_lens, **kwargs)
            elif len(input) == 2:
                feats, feat_lens = input
                encoded = self.encoder(feats, feat_lens)
                if self.encoder.embed_size!=  self.decoder.embed_size:
                    encoded["audio_embeds"]= self.ln(encoded["audio_embeds"])
                output = self.inference_forward(encoded, **kwargs)
            else:
                raise Exception("Number of input should be either 4 (feats, feat_lens, caps, cap_lens) or 2 (feats, feat_lens)")

            return output

    def prepare_output(self, encoded, output, max_length):
        N = encoded["audio_embeds"].size(0)
        output["seqs"] = torch.empty(N, max_length, dtype=torch.long).fill_(self.end_idx)
        output["logits"] = torch.empty(N, max_length, self.vocab_size).to(encoded["audio_embeds"].device)

        output["outputs"] = torch.empty(output["seqs"].size(0), max_length, self.decoder.embed_size).to(encoded["audio_embeds"].device)
        output["sampled_logprobs"] = torch.zeros(output["seqs"].size(0), max_length)
        
        
        output["attn_weights"] = torch.empty(output["seqs"].size(0), max(encoded["audio_embeds_lens"]), max_length)

        if  hasattr(self.pnet,'gmm_kernel'):
            output["p_means"] = torch.empty(self.pnet.gmm_kernel,output["seqs"].size(0),max_length,self.pnet.embed_size)
            output["p_logs"] = torch.empty(self.pnet.gmm_kernel,output["seqs"].size(0),max_length,self.pnet.embed_size)
            
        else:
            output["p_means"] = torch.empty(output["seqs"].size(0),max_length,self.pnet.embed_size)
            output["p_logs"] = torch.empty(output["seqs"].size(0),max_length,self.pnet.embed_size)


        output["p_z"] = torch.empty(output["seqs"].size(0),max_length,self.pnet.embed_size)
        
        if "q_means" in encoded:
            output["q_means"] = encoded["q_means"]
            output["q_logs"] = encoded["q_logs"]
            output["q_z"] = encoded["q_z"]
            output["q_means_utt"] = encoded["q_means_utt"]
            output["q_logs_utt"] = encoded["q_logs_utt"]
        return output
  
    def decode_step(self,  decoder_input, encoded, caps, output, t, **kwargs):
        """Decoding operation of timestep t"""

        self.prepare_decoder_input(decoder_input, encoded, caps, output, t, **kwargs)
        
        output_pnet_t = self.pnet(decoder_input["word"],decoder_input["enc_mem"],decoder_input["hiddens_state"],decoder_input["last_z"],decoder_input["enc_mem_lens"])
        
        #prepare decoder
        if caps is not None:
            decoder_input["z"] = output["q_z"][:,t,:]
            if kwargs["dis_ratio"] ==0:
                # print("yes!")
                decoder_input["z"] = output["q_z"][:,t,:]
            elif torch.rand(1)<= kwargs["dis_ratio"]:
                decoder_input["z"] = output_pnet_t["z"]
        else: 
            decoder_input["z"] = output_pnet_t["z"]

        output_t = self.decoder(**decoder_input)
        output_t.update(output_pnet_t)

        logits_t = output_t["logits"].squeeze(1)
        sampled = self.sample_next_word(logits_t,**kwargs)
        
        self.stepwise_process_step(output, output_t, t, sampled)

    def prepare_decoder_input(self, decoder_input, encoded, caps, output, t, **kwargs):

        decoder_input["enc_mem"] = encoded["audio_embeds"]
        decoder_input["enc_mem_lens"] = encoded["audio_embeds_lens"]
        decoder_input["audio_embeds_pooled"] = encoded["audio_embeds_pooled"]
        ###############
        # determine input word
        ################
        if caps is not None and random.random() < kwargs["ss_ratio"]: # training, scheduled sampling
            word = caps[:, t].long()
        else:
            if t == 0:
                word = torch.tensor([self.start_idx,] * output["seqs"].size(0)).long()
            else:
                word = output["seqs"][:, t-1]
        # word: [N,]
        decoder_input["word"] = word.unsqueeze(1)

        ################
        # prepare rnn state
        ################
        if t == 0:
            decoder_input["state"] = self.decoder.init_hidden(output["seqs"].size(0)).to(encoded["audio_embeds"].device)
            decoder_input["hiddens_state"] = self.pnet.init_hidden(output["seqs"].size(0),encoded["audio_embeds"].device)
            decoder_input["last_z"] = torch.zeros(output["seqs"].size(0),self.decoder.embed_size).to(encoded["audio_embeds"].device)
        if t > 0:
            decoder_input["state"] = output["state"]
            decoder_input["hiddens_state"] = output["hiddens_state"]
            decoder_input["last_z"] = output["last_z"]

        return decoder_input

    def stepwise_process_step(self, output, output_t, t, sampled):

        output["logits"][:, t, :] = output_t["logits"].squeeze(1)
        output["outputs"][:, t, :] = output_t["output"].squeeze(1)
        output["seqs"][:, t] = sampled["w_t"]
        output["sampled_logprobs"][:, t] = sampled["probs"]

        if len(output["p_means"].shape) == 4:
            output["p_means"][:,:, t, :] = output_t["mean"].squeeze(1)
            output["p_logs"][:,:, t, :] = output_t["log"].squeeze(1)
        else:
            output["p_means"][:, t, :] = output_t["mean"].squeeze(1)
            output["p_logs"][:,t, :] = output_t["log"].squeeze(1)
        output["p_z"][:, t, :] = output_t["z"].squeeze(1)

        if "state" in output_t:
            output["state"] = output_t["state"]
            output["hiddens_state"] = output_t["hiddens_state"]
            output["attn_weights"][:, :, t] = output_t["weights"]
        output["last_z"] = output_t['z']
        
    def train_forward(self, encoded, caps, cap_lens,**kwargs):
        # Bahdanau attention only supports step-by-step implementation, so we implement forward in 
        # step-by-step manner whether in training or evaluation
        output = self.stepwise_forward(encoded, caps, cap_lens,**kwargs)
        if "embedding_lens" in kwargs and kwargs["embedding_lens"] !=self.decoder.model.hidden_size :
            output["outputs"] = self.output_transform(output["outputs"])
        
        return output

    def inference_forward(self, encoded, **kwargs):
        # optional sampling keyword arguments
        method = kwargs.get("method", "greedy")
        max_length = kwargs.get("max_length", self.max_length)
        if method == "beam":
            beam_size = kwargs.get("beam_size", 3)
            return self.beam_search(encoded, max_length, beam_size)
        elif method == "dbs":
            beam_size = kwargs.get("beam_size", 5)
            group_size = kwargs.get("group_size",5)
            diversity_lambda =  kwargs.get("diversity_lambda",0.5)
            temperature = kwargs.get('temperature', 1.0)
            group_nbest = kwargs.get("group_nbest",True)
            return self.diverse_beam_search(encoded, max_length, beam_size,group_size,diversity_lambda,temperature,group_nbest)
        return self.stepwise_forward(encoded, None, None, **kwargs)

    def beam_search(self, encoded, max_length, beam_size):
        output = {}
        self.prepare_output(encoded, output, max_length)

        # instance by instance beam seach
        for i in range(encoded["audio_embeds"].size(0)):
   
            output_i = {}
            self.prepare_beamsearch_output(output_i, beam_size, encoded, max_length)
            decoder_input = {}
            for t in range(max_length):
                output_t = self.beamsearch_step(decoder_input, encoded, output_i, i, t, beam_size)
                logits_t = output_t["logits"].squeeze(1)
                logprobs_t = torch.log_softmax(logits_t, dim=1)

                logprobs_t = output_i["top_k_logprobs"].unsqueeze(1).expand_as(logprobs_t) + logprobs_t
                top_k_logprobs, top_k_words = logprobs_t.view(-1).topk(beam_size, 0, True, True)
                output_i["top_k_logprobs"] = top_k_logprobs
                
                output_i["prev_word_inds"] = torch.div(top_k_words, self.vocab_size, rounding_mode='trunc')
                output_i["next_word_inds"] = top_k_words % self.vocab_size  # [beam_size,]
                if t == 0:
                    output_i["seqs"] = output_i["next_word_inds"].unsqueeze(1)
                else:
                    output_i["seqs"] = torch.cat([output_i["seqs"][output_i["prev_word_inds"]], 
                                                  output_i["next_word_inds"].unsqueeze(1)], dim=1)

                self.beamsearch_process_step(output_i, output_t)

            self.beamsearch_process(output, output_i, i)
            
        return output
        
    def prepare_beamsearch_output(self, output, beam_size, encoded, max_length):
        output["top_k_logprobs"] = torch.zeros(beam_size).to(encoded["audio_embeds"].device)

        output["attn_weights"] = torch.empty(beam_size, max(encoded["audio_embeds_lens"]), max_length)
        
        output["p_means"] = torch.empty(beam_size,max_length,self.pnet.embed_size)
        output["p_logs"] = torch.empty(beam_size,max_length,self.pnet.embed_size)
        output["p_z"] = torch.empty(beam_size,max_length,self.pnet.embed_size)
    
        output["done_beams"] = []
    def beamsearch_step(self, decoder_input, encoded, output, i, t, beam_size):
        self.prepare_beamsearch_decoder_input(decoder_input, encoded, output, i, t, beam_size)

        output_pnet_t = self.pnet(decoder_input["word"],decoder_input["enc_mem"],decoder_input["hiddens_state"],decoder_input["last_z"],decoder_input["enc_mem_lens"])
        decoder_input["z"] = output_pnet_t["z"]

        output_t = self.decoder(**decoder_input)

        output_t.update(output_pnet_t)
        if "weights" in output_t:
            output["attn_weights"][:, :, t] = output_t["weights"]
        return output_t

    def prepare_beamsearch_decoder_input(self, decoder_input, encoded, output, i, t, beam_size):
        
        if t == 0:
            decoder_input["enc_mem"] =  encoded["audio_embeds"][i].unsqueeze(0).repeat(beam_size, 1, 1)
            decoder_input["enc_mem_lens"] = encoded["audio_embeds_lens"][i].repeat(beam_size)
            decoder_input["audio_embeds_pooled"] = encoded["audio_embeds_pooled"][i].unsqueeze(0).repeat(beam_size, 1)

            decoder_input["state"] = self.decoder.init_hidden(beam_size).to(encoded["audio_embeds"].device)
            decoder_input["hiddens_state"] = self.pnet.init_hidden(beam_size,encoded["audio_embeds"].device)
            decoder_input["last_z"] = torch.zeros(beam_size,self.decoder.embed_size).to(encoded["audio_embeds"].device)

            w_t = torch.tensor([self.start_idx,] * beam_size).long()
        else:
            w_t = output["next_word_inds"]

            decoder_input["state"] = output["state"][:, output["prev_word_inds"], :].contiguous()
            if output["hiddens_state"] != None:
                hiddens_state_0 = output["hiddens_state"][0][:, output["prev_word_inds"], :].contiguous()
                hiddens_state_1 = output["hiddens_state"][1][:, output["prev_word_inds"], :].contiguous()
                decoder_input["hiddens_state"] =(hiddens_state_0,hiddens_state_1)
            decoder_input["last_z"] = output["last_z"][output["prev_word_inds"], :].contiguous()

        decoder_input["word"] = w_t.unsqueeze(1)

    def beamsearch_process_step(self, output, output_t):
        
        if "state" in output_t:
            output["state"] = output_t["state"]
            output["hiddens_state"] = output_t["hiddens_state"]
            output["attn_weights"] = output["attn_weights"][output["prev_word_inds"], :, :]
        
        output["last_z"] = output_t["z"]
        
    
    def beamsearch_process(self,  output, output_i, i):
        done_beams = sorted(output_i["done_beams"], key=lambda x: -x["score"])
        done_beams = done_beams[:output["seqs"].shape[1]]
        for out_idx, done_beam in enumerate(done_beams):
            seq = done_beam["seq"]
            output["seqs"][i][out_idx, :len(seq)] = seq
        output["seqs"][i] = output_i["seqs"][0].unsqueeze(0)

        if "attn_weights" in output_i:
            output["attn_weights"][i] = output_i["attn_weights"][0]

    def prepare_dbs_decoder_input(self,encoded,decoder_input,t,i,bdash,divm, output_i):

        local_time = t - divm
        if t == 0:
            decoder_input["enc_mem"] =  encoded["audio_embeds"][i].unsqueeze(0).repeat(bdash, 1, 1)
            decoder_input["enc_mem_lens"] = encoded["audio_embeds_lens"][i].repeat(bdash)
            decoder_input["audio_embeds_pooled"] = encoded["audio_embeds_pooled"][i].unsqueeze(0).repeat(bdash, 1)
        ###############
        # determine input word
        ################
        if local_time == 0:
            word = torch.tensor([self.start_idx,] * bdash).long()
            decoder_input["state"] = self.decoder.init_hidden(bdash).to(encoded["audio_embeds"].device)
            decoder_input["hiddens_state"] = self.pnet.init_hidden(bdash,encoded["audio_embeds"].device)
            decoder_input["last_z"] = torch.zeros(bdash,self.decoder.embed_size).to(encoded["audio_embeds"].device)
        else:
            word = output_i["next_word"][divm]
        decoder_input["word"] = word.unsqueeze(1)

        if local_time > 0:
 
            decoder_input["state"] = output_i["state"][divm][
                :, output_i["prev_words_beam"][divm], :].contiguous()
            hiddens_state_0 = output_i["hiddens_state"][divm][0][:, output_i["prev_words_beam"][divm], :].contiguous()
            hiddens_state_1 = output_i["hiddens_state"][divm][1][:, output_i["prev_words_beam"][divm], :].contiguous()
            decoder_input["hiddens_state"] =(hiddens_state_0,hiddens_state_1)
            decoder_input["last_z"] = output_i["last_z"][divm][output_i["prev_words_beam"][divm], :].contiguous()
         
        return decoder_input
    def dbs_process_step(self, output_i, output_t):
        divm = output_t["divm"]
        output_i["state"][divm] = output_t["state"]
        output_i["hiddens_state"][divm] = output_t["hiddens_state"]
        output_i["last_z"][divm] = output_t['z']

    def dbs_step(self, encoded, decoder_input,output_i, i, t, bdash,divm):
        
        self.prepare_dbs_decoder_input(encoded,decoder_input,t,i,bdash,divm, output_i)
        output_pnet_t = self.pnet(decoder_input["word"],decoder_input["enc_mem"],decoder_input["hiddens_state"],decoder_input["last_z"],decoder_input["enc_mem_lens"])
        decoder_input["z"] = output_pnet_t["z"]
        output_t = self.decoder(**decoder_input)
        output_t.update(output_pnet_t)

        return output_t

    def prepare_dbs_output(self,output_i, bdash, encoded,group_size, max_length):
        output_i["prev_words_beam"] = [None for _ in range(group_size)]
        output_i["next_word"] = [None for _ in range(group_size)]
        output_i["state"] = [None for _ in range(group_size)]
        output_i["hiddens_state"] = [None for _ in range(len(output_i["prev_words_beam"]))]
        output_i["last_z"] = [None for _ in range(len(output_i["prev_words_beam"]))]
        output_i["attn_weights"] = [torch.empty(bdash, max(encoded["audio_embeds_lens"]), max_length) for _ in range(len(output_i["prev_words_beam"]))]


