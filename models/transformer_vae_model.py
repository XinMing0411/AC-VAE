import imp
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from models.word_model import CaptionModel
from models.utils import repeat_tensor
import models.text_encoder as text_encoder
from models.vae_model import VAEModel
class TransVAEModel(VAEModel):
    def __init__(self, Audioencoder: nn.Module, Textdecoder: nn.Module, **kwargs):
        super(TransVAEModel,self).__init__(Audioencoder, Textdecoder, **kwargs)

    def prepare_decoder_input(self, decoder_input, encoded, caps, output, t, **kwargs):
        # super(TransVAEModel,self).prepare_decoder_input(decoder_input, encoded, caps, output, t, **kwargs)
        decoder_input["enc_mem"] = encoded["audio_embeds"]
        decoder_input["enc_mem_lens"] = encoded["audio_embeds_lens"]
        decoder_input["audio_embeds_pooled"] = encoded["audio_embeds_pooled"]
        
        ###############
        # determine input word
        ################
        if caps is not None and random.random() < kwargs["ss_ratio"]: # training, scheduled sampling
            words = caps[:, :t+1]
        else:
            if t == 0:
                words = torch.tensor([self.start_idx,] * output["seqs"].size(0)).unsqueeze(1).long()
            else:
                words = torch.cat((torch.tensor([self.start_idx,] * output["seqs"].size(0)).unsqueeze(1).long(), output["seqs"][:, :t]), dim=-1)
        # word: [N,]
        decoder_input["words"] = words
        decoder_input["caps_padding_mask"] = (words == self.pad_idx).to(encoded["audio_embeds"].device)

        ################
        # prepare state
        ################
        if t == 0:
            decoder_input["state"] = None
            decoder_input["hiddens_state"] = None
            decoder_input["last_z"] = torch.zeros(output["seqs"].size(0),self.decoder.embed_size).unsqueeze(1).to(encoded["audio_embeds"].device)
        if t > 0:
            decoder_input["last_z"] = torch.cat((torch.zeros(output["seqs"].size(0),self.decoder.embed_size).unsqueeze(1), output["p_z"][:, :t,:]), dim=1)

        return decoder_input

    def decode_step(self,  decoder_input, encoded, caps, output, t, **kwargs):
        """Decoding operation of timestep t"""

        self.prepare_decoder_input(decoder_input, encoded, caps, output, t, **kwargs)
        output_pnet_t = self.pnet(**decoder_input)
        #prepare decoder
        if caps is not None:
            decoder_input["z"] = output["q_z"][:,:t+1,:]
        else: 
            output["p_means"][:, t, :] = output_pnet_t["mean"].squeeze(1)
            output["p_logs"][:, t, :] = output_pnet_t["log"].squeeze(1)
            output["p_z"][:, t, :] = output_pnet_t["z"].squeeze(1)

            decoder_input["z"] = output["p_z"][:,:t+1, :]

        output_t = self.decoder(**decoder_input)
        output_t.update(output_pnet_t)

        logits_t = output_t["logits"].squeeze(1)

        sampled = self.sample_next_word(logits_t,**kwargs)
        
        self.stepwise_process_step(output, output_t, t, sampled)

    def beam_search(self, encoded, max_length, beam_size):
        output = {}
        self.prepare_output(encoded, output, max_length)
        # output["seqs"] = torch.empty(encoded["audio_embeds"].size(0), beam_size, max_length, dtype=torch.long).fill_(self.end_idx)

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
                    
                    output_i["p_z"] =  output_i["p_z"][output_i["prev_word_inds"],:,:]
                self.beamsearch_process_step(output_i, output_t)
            self.beamsearch_process(output, output_i, i)
        return output
        
    def prepare_beamsearch_decoder_input(self, decoder_input, encoded, output, i, t, beam_size):
        
        if t == 0:
            
            decoder_input["enc_mem_lens"] = encoded["audio_embeds_lens"][i].repeat(beam_size)
            decoder_input["enc_mem"] =  encoded["audio_embeds"][i, :encoded["audio_embeds_lens"][i]].unsqueeze(0).repeat(beam_size, 1, 1)
            decoder_input["audio_embeds_pooled"] = encoded["audio_embeds_pooled"][i].unsqueeze(0).repeat(beam_size, 1)

            decoder_input["state"] = None
            decoder_input["hiddens_state"] = None
            # decoder_input["last_z"] = torch.zeros(output["seqs"].size(0),self.decoder.embed_size).unsqueeze(1).to(encoded["audio_embeds"].device)
            decoder_input["last_z"] = torch.zeros(beam_size,self.decoder.embed_size).unsqueeze(1).to(encoded["audio_embeds"].device)

            words = torch.tensor([self.start_idx,] * beam_size).unsqueeze(1).long().to(encoded["audio_embeds"].device)
        else:
            words = torch.cat((torch.tensor([self.start_idx,] * beam_size).unsqueeze(1).long().to(encoded["audio_embeds"].device),output["seqs"]), dim=1)
            decoder_input["last_z"] = torch.cat((torch.zeros(beam_size,self.decoder.embed_size).unsqueeze(1), output["p_z"][:,:t,:]), dim=1).contiguous()

        decoder_input["words"] = words
        decoder_input["caps_padding_mask"] = (words == self.pad_idx).to(encoded["audio_embeds"].device)

    def beamsearch_step(self, decoder_input, encoded, output, i, t, beam_size):
        self.prepare_beamsearch_decoder_input(decoder_input, encoded, output, i, t, beam_size)
        output_pnet_t = self.pnet(**decoder_input)
        
        output["p_means"][:, t, :] = output_pnet_t["mean"].squeeze(1)
        output["p_logs"][:, t, :] = output_pnet_t["log"].squeeze(1)
        output["p_z"][:, t, :] = output_pnet_t["z"].squeeze(1)
        
        decoder_input["z"] = output["p_z"][:,:t+1, :]
        
        output_t = self.decoder(**decoder_input)
        output_t.update(output_pnet_t)
        
        return output_t