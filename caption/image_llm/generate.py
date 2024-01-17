# coding=utf-8
import torch
import torch.nn.functional as F
import numpy as np
from image_llm.models.generate_opt import generate as opt_generate


@torch.no_grad()
def generate_beam(model, tokenizer, beam_size: int = 5, prompt=None, embed=None,
                    entry_length=67, temperature=1., stop_token: str = '.'):
    model.eval()
    device = next(model.parameters()).device

    if hasattr(model, "llm") and 'opt' in model.llm.model_name:
        output_texts = opt_generate(model.llm.llm, tokenizer, prompt="", query_embeds=embed, num_beams=beam_size, device=device)
        import pdb; pdb.set_trace()
        return output_texts

    stop_token_index = tokenizer.encode(stop_token)[0]

    tokens = None
    scores = None
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)

    if embed is not None:
        generated = embed
    else:
        if tokens is None:
            tokens = torch.tensor(tokenizer.encode(prompt))
            tokens = tokens.unsqueeze(0).to(device)
            generated = model.llm.input_token_embedding()(tokens)

    for i in range(entry_length):
        outputs = model.llm(inputs_embeds=generated)
        logits = outputs.logits
        logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
        logits = logits.softmax(-1).log()

        if scores is None:
            scores, next_tokens = logits.topk(beam_size, -1)
            generated = generated.expand(beam_size, *generated.shape[1:])
            next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
            if tokens is None:
                tokens = next_tokens
            else:
                tokens = tokens.expand(beam_size, *tokens.shape[1:])
                tokens = torch.cat((tokens, next_tokens), dim=1)
        else:
            logits[is_stopped] = -float(np.inf)
            logits[is_stopped, 0] = 0
            scores_sum = scores[:, None] + logits
            seq_lengths[~is_stopped] += 1

            scores_sum_average = scores_sum / seq_lengths[:, None]
            scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(beam_size, -1)
            # next_tokens_source = next_tokens // scores_sum.shape[1]
            next_tokens_source = torch.div(next_tokens, scores_sum.shape[1], rounding_mode='floor')
            seq_lengths = seq_lengths[next_tokens_source]
            next_tokens = next_tokens % scores_sum.shape[1]
            next_tokens = next_tokens.unsqueeze(1)
            tokens = tokens[next_tokens_source]
            tokens = torch.cat((tokens, next_tokens), dim=1)
            generated = generated[next_tokens_source]
            scores = scores_sum_average * seq_lengths
            is_stopped = is_stopped[next_tokens_source]
        
        next_token_embed = model.llm.input_token_embedding()(next_tokens.squeeze()).view(generated.shape[0], 1, -1)
        generated = torch.cat((generated, next_token_embed), dim=1)

        is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
        if is_stopped.all():
            break

    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [tokenizer.decode(output[:int(length)], skip_special_tokens=True) for output, length in zip(output_list, seq_lengths)]
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]

    return output_texts


@torch.no_grad()
def generate2(
        model,
        tokenizer,
        tokens=None,
        prompt=None,
        embed=None,
        entry_count=1,
        entry_length=67,  # maximum number of words
        top_p=0.8,
        temperature=1.,
        stop_token: str = '.',
):
    model.eval()
    generated_num = 0
    generated_list = []
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")
    device = next(model.parameters()).device

    for entry_idx in range(entry_count):

        # previous tokens / prompts
        if embed is not None:
            generated = embed
        else:
            if tokens is None:
                tokens = torch.tensor(tokenizer.encode(prompt))
                tokens = tokens.unsqueeze(0).to(device)
            generated = model.llm.input_token_embedding()(tokens)

        for i in range(entry_length):
            outputs = model.llm(inputs_embeds=generated)
            logits = outputs.logits

            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[:, indices_to_remove] = filter_value
            next_token = torch.argmax(logits, -1).unsqueeze(0)
            next_token_embed = model.llm.input_token_embedding()(next_token)

            if tokens is None:
                tokens = next_token
            else:
                tokens = torch.cat((tokens, next_token), dim=1)
            generated = torch.cat((generated, next_token_embed), dim=1)

            if stop_token_index == next_token.item() or next_token.item() == 764:
                break

        output_list = list(tokens.squeeze().cpu().numpy())
        output_text = tokenizer.decode(output_list, skip_special_tokens=True)
        generated_list.append(output_text)

    return generated_list[0]
