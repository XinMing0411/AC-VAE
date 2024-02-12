import torch
import torch.nn as nn

from models.word_model import CaptionModel

class Seq2SeqAttention(nn.Module):

    def __init__(self, hs_enc, hs_dec, attn_size):
        """
        Args:
            hs_enc: encoder hidden size
            hs_dec: decoder hidden size
            attn_size: attention vector size
        """
        super(Seq2SeqAttention, self).__init__()
        self.h2attn = nn.Linear(hs_enc + hs_dec, attn_size)
        self.v = nn.Parameter(torch.randn(attn_size))
        nn.init.kaiming_uniform_(self.h2attn.weight)

    def forward(self, h_dec, h_enc, src_lens):
        """
        Args:
            h_dec: decoder hidden state, [N, hs_dec]
            h_enc: encoder hiddens/outputs, [N, src_max_len, hs_enc]
            src_lens: source (encoder input) lengths, [N, ]
        """
        N = h_enc.size(0)
        src_max_len = h_enc.size(1)
        h_dec = h_dec.unsqueeze(1).repeat(1, src_max_len, 1) # [N, src_max_len, hs_dec]

        attn_input = torch.cat((h_dec, h_enc), dim=-1)
        attn_out = torch.tanh(self.h2attn(attn_input)) # [N, src_max_len, attn_size]

        v = self.v.repeat(N, 1).unsqueeze(1) # [N, 1, attn_size]
        # score = torch.bmm(v, attn_out.permute(0, 2, 1)).squeeze(1) # [N, src_max_len]
        score = (v@attn_out.permute(0, 2, 1)).squeeze(1) # [N, src_max_len]

        idxs = torch.arange(src_max_len).repeat(N).view(N, src_max_len)
        mask = (idxs < src_lens.view(-1, 1)).to(h_dec.device)

        score = score.masked_fill(mask == 0, -1e10)
        weights = torch.softmax(score, dim=-1) # [N, src_max_len]
        # ctx = torch.bmm(weights.unsqueeze(1), h_enc).squeeze(1) # [N, hs_enc]
        ctx = (weights.unsqueeze(1)@h_enc).squeeze(1) # [N, hs_enc]

        return ctx, weights


class Seq2SeqAttnModel(CaptionModel):

    def __init__(self, encoder, decoder, **kwargs):
                #add 
        super(Seq2SeqAttnModel, self).__init__(encoder, decoder, **kwargs)
        if "embedding_lens" in kwargs and kwargs["embedding_lens"] != self.decoder.model.hidden_size :
            self.output_transform = nn.Linear(decoder.model.hidden_size,kwargs["embedding_lens"])
            nn.init.kaiming_uniform_(self.output_transform.weight)


    def train_forward(self, encoded, caps, cap_lens, **kwargs):
        # Bahdanau attention only supports step-by-step implementation, so we implement forward in 
        # step-by-step manner whether in training or evaluation
        output = self.stepwise_forward(encoded, caps, cap_lens, **kwargs)
        if "embedding_lens" in kwargs and kwargs["embedding_lens"] !=self.decoder.model.hidden_size :
            output["outputs"] = self.output_transform(output["outputs"])
        
        return output

    def prepare_output(self, encoded, output, max_length):
        super(Seq2SeqAttnModel, self).prepare_output(encoded, output, max_length)
        attn_weights = torch.empty(output["seqs"].size(0), max(encoded["audio_embeds_lens"]), max_length)
        output["attn_weights"] = attn_weights
        # output["audio_embeds_pooled"] = encoded["audio_embeds_pooled"]


    def prepare_decoder_input(self, decoder_input, encoded, caps, output, t, **kwargs):
        super(Seq2SeqAttnModel, self).prepare_decoder_input(decoder_input, encoded, caps, output, t, **kwargs)
        if t == 0:
            decoder_input["enc_mem"] = encoded["audio_embeds"]
            decoder_input["enc_mem_lens"] = encoded["audio_embeds_lens"]
            if encoded["state"] is None:
                state = self.decoder.init_hidden(output["seqs"].size(0))
                state = state.to(encoded["audio_embeds"].device)
                decoder_input["state"] = state

        # decoder_input: { "word": ..., "state": ..., "enc_mem": ..., "enc_mem_lens": ... }

    def stepwise_process_step(self, output, output_t, t, sampled):
        super(Seq2SeqAttnModel, self).stepwise_process_step(output, output_t, t, sampled)
        output["attn_weights"][:, :, t] = output_t["weights"]

    def prepare_beamsearch_output(self, output, beam_size, encoded, max_length):
        super(Seq2SeqAttnModel, self).prepare_beamsearch_output(output, beam_size, encoded, max_length)
        output["attn_weights"] = torch.empty(beam_size, max(encoded["audio_embeds_lens"]), max_length)
        output["done_beams"] = []
    def prepare_beamsearch_decoder_input(self, decoder_input, encoded, output, i, t, beam_size):
        super(Seq2SeqAttnModel, self).prepare_beamsearch_decoder_input(decoder_input, encoded, output, i, t, beam_size)
        if t == 0:
            enc_mem = encoded["audio_embeds"][i]
            decoder_input["enc_mem"] = enc_mem.unsqueeze(0).repeat(beam_size, 1, 1)
            enc_mem_lens = encoded["audio_embeds_lens"][i]
            decoder_input["enc_mem_lens"] = enc_mem_lens.repeat(beam_size)
            if decoder_input["state"] is None:
                decoder_input["state"] = self.decoder.init_hidden(beam_size)
                decoder_input["state"] = decoder_input["state"].to(decoder_input["enc_mem"].device)

    def beamsearch_step(self, decoder_input, encoded, output, i, t, beam_size):
        output_t = super(Seq2SeqAttnModel, self).beamsearch_step(decoder_input, encoded, output, i, t, beam_size)
        output["attn_weights"][:, :, t] = output_t["weights"]
        return output_t

    def beamsearch_process_step(self, output, output_t):
        super().beamsearch_process_step(output, output_t)
        output["attn_weights"] = output["attn_weights"][output["prev_word_inds"], :, :]

    def beamsearch_process(self, output, output_i, i):
        done_beams = sorted(output_i["done_beams"], key=lambda x: -x["score"])
        for out_idx, done_beam in enumerate(done_beams) :
            seq = done_beam["seq"]
            # print(out_idx)
            if out_idx<output["seqs"].shape[1]:
                output["seqs"][i][out_idx, :len(seq)] = seq 

        # output["seqs"][i] = output_i["seqs"][0]
        # output["seqs"][i] = output_i["seqs"].unsqueeze(0)
        output["attn_weights"][i] = output_i["attn_weights"].unsqueeze(0)
        # output["attn_weights"][i] = output_i["attn_weights"][0]


class Seq2SeqAttnEnsemble():

    def __init__(self, models, max_length=20) -> None:
        self.models = models
        self.max_length = max_length
        self.end_idx = models[0].end_idx
    
    def inference(self, feats, feat_lens):
        encoded = []
        for model in self.models:
            encoded.append(model.encoder(feats, feat_lens))

        N = feats.size(0)
        output_seqs = torch.empty(N, self.max_length, dtype=torch.long).fill_(self.end_idx)
        
