import torch

def find_target_after_phrase(token_ids, tokenizer, phrase = "The final answer", target_word = '.'):
    """
    Find the token index corresponding to the first period after the phrase "The final answer".
    
    Args:
        token_ids: A 1D sequence of token IDs (list or tensor)
        tokenizer: The tokenizer used to decode the tokens
        
    Returns:
        The index of the first period token after "The final answer" phrase,
        or -1 if no such token is found.
    """
    if phrase not in tokenizer.decode(token_ids, skip_special_tokens=True):
        return -1
    
    # Convert to list if it's a tensor
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()

    found_target = False
    
    # Iterate through the token sequence
    for i in range(len(token_ids)):
        # Decode all tokens up to current position to get the current text
        current_text = tokenizer.decode(token_ids[:i+1], skip_special_tokens=True)
        
        # Check if we've found the target phrase
        if not found_target and phrase.lower() in current_text.lower():
            found_target = True
            continue
        
        # If we've found the target phrase, look for the period
        if found_target:
            # Get the token text at current position
            token_text = tokenizer.decode([token_ids[i]], skip_special_tokens=True)
            
            # Check if this token contains a period
            if target_word in token_text:
                return i
    
    # If no period found after "The final answer"
    return -1



def find_target_after_phrase_str(strng, phrase = "The final answer", target_word = '>>'):
    try:
        start_idx = strng.find(phrase)
        if start_idx == -1:
            return -1
        end_idx = strng.find(target_word, start_idx)
        if end_idx == -1:
            return -1
        return strng[:end_idx + len(target_word)]
    except:
        return -1