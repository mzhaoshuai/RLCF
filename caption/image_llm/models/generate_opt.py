# coding=utf-8
# https://github.com/salesforce/LAVIS/blob/main/lavis/models/blip2_models/blip2_opt.py
import torch


@torch.no_grad()
def generate(
    opt_model,
    opt_tokenizer,
    prompt="",
    query_embeds=None,
    use_nucleus_sampling=False,
    num_beams=5,
    max_length=50,
    min_length=1,
    top_p=0.92,
    repetition_penalty=1.0,
    length_penalty=1.0,
    num_captions=1,
    temperature=1,
    device='cpu'
):
    """
    Args:
        samples (dict): A dictionary containing the following keys:
            - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
        use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
        num_beams (int): Number of beams for beam search. 1 means no beam search.
        max_length (int): The maximum length of the sequence to be generated.
        min_length (int): The minimum length of the sequence to be generated.
        top_p (float): The cumulative probability for nucleus sampling.
        repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
        num_captions (int): Number of captions to be generated for each image.
    Returns:
        captions (list): A list of strings of length batch_size * num_captions.
    """
    if prompt is not None:
        prompt = [prompt] * query_embeds.size(0)
        opt_tokens = opt_tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = opt_tokens.input_ids
    else:
        input_ids = None
    
    if query_embeds is not None:
        atts_opt = torch.ones(query_embeds.size()[:-1], dtype=torch.long).to(device)
        if prompt is not None:
            attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)
        else:
            attention_mask = atts_opt
    else:
        attention_mask = opt_tokens.attention_mask

    eos_token_id = opt_tokenizer("\n", add_special_tokens=False).input_ids[0]

    if use_nucleus_sampling:
        if query_embeds is not None:
            query_embeds = query_embeds.repeat_interleave(num_captions, dim=0)
        num_beams = 1
    else:
        if query_embeds is not None:
            query_embeds = query_embeds.repeat_interleave(num_beams, dim=0)

    outputs = opt_model.generate(
        input_ids=input_ids,
        query_embeds=query_embeds,
        attention_mask=attention_mask,
        do_sample=use_nucleus_sampling,
        top_p=top_p,
        top_k=0 if use_nucleus_sampling else 50,
        temperature=temperature,
        num_beams=num_beams,
        max_new_tokens=max_length,
        min_length=min_length,
        eos_token_id=eos_token_id,
        repetition_penalty=repetition_penalty,
        length_penalty=length_penalty,
        num_return_sequences=num_captions,
    )

    # import pdb; pdb.set_trace()
    prompt_length = opt_tokens.input_ids.shape[1]
    output_text = opt_tokenizer.batch_decode(outputs[:, prompt_length:], skip_special_tokens=True)
    output_text = [text.strip() for text in output_text]

    return output_text


if __name__ == "__main__":
    from transformers import AutoTokenizer
    from modeling_opt import OPTForCausalLM, OPTConfig

    config_dir = "opt-350m"
    model = OPTForCausalLM.from_pretrained(config_dir)
    tokenizer = AutoTokenizer.from_pretrained(config_dir, use_fast=False)
    prompt = "Hello world! I am"
    query_embeds = None

    results = generate(model, tokenizer, prompt, query_embeds)
    print(results)

    # Step 2: prepare the inputs for the generation method manually and call it
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    query_embeds = model.get_input_embeddings()(input_ids)
    print(query_embeds.shape)

    prompt = "Hello "
    results = generate(model, tokenizer, prompt, query_embeds)
    print(results)
