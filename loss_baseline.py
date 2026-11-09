"""Piece 1: Zero-shot baseline and answer-only loss mask."""
import torch
import lab06_starter as S


def answer_only_labels(input_ids: torch.Tensor, attention_mask: torch.Tensor, prompt_lens: torch.Tensor) -> torch.Tensor:
    """Construct a label tensor that scores only the single answer token.
    
    In causal language modeling, Hugging Face aligns logits[:, :-1] with labels[:, 1:].
    The answer token sits exactly at index plen in input_ids (where plen is prompt length).
    Logits emitted at index (plen - 1) will predict labels[:, plen].
    All other positions are masked with -100 so they contribute zero loss.
    """
    labels = torch.full_like(input_ids, -100)
    for i, plen in enumerate(prompt_lens):
        # Index plen is the single answer token position in input_ids
        labels[i, plen] = input_ids[i, plen]
    return labels


def verify_answer_only_mask():
    """Verify that answer_only_labels satisfies the assignment requirements."""
    print("--- Verifying answer_only_labels Mask Integrity ---")
    # Synthetic batch of 3 examples
    dummy_input_ids = torch.tensor([
        [101, 2054, 2003, 1037, 3899, 9999, 0, 0],   # prompt len 5, answer 9999 at idx 5, 2 pads
        [101, 1045, 2514, 8888, 0, 0, 0, 0],         # prompt len 3, answer 8888 at idx 3, 4 pads
        [101, 7777, 0, 0, 0, 0, 0, 0]                # prompt len 1, answer 7777 at idx 1, 6 pads
    ], dtype=torch.long)
    dummy_mask = torch.tensor([
        [1, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0]
    ], dtype=torch.long)
    dummy_plens = torch.tensor([5, 3, 1], dtype=torch.long)

    labels = answer_only_labels(dummy_input_ids, dummy_mask, dummy_plens)

    # 1. Exactly one token per example is scored (not equal to -100)
    scored_per_row = (labels != -100).sum(dim=-1)
    assert (scored_per_row == 1).all(), f"Failed: expected exactly 1 scored token per row, got {scored_per_row.tolist()}"

    # 2. No prompt token is scored (all positions 0 until plen - 1 must be -100)
    for i, plen in enumerate(dummy_plens):
        assert (labels[i, :plen] == -100).all(), f"Failed: prompt token was scored in row {i}!"
        # 3. The scored token is exactly the answer token
        assert labels[i, plen] == dummy_input_ids[i, plen], f"Failed: target mismatch at index {plen} in row {i}!"
        # 4. Padded tokens are also -100
        assert (labels[i, plen + 1:] == -100).all(), f"Failed: padding token was scored in row {i}!"

    print("Assertion passed: Exactly 1 token per example is scored.")
    print("Assertion passed: Zero prompt tokens are scored.")
    print("Assertion passed: Zero padding tokens are scored.\n")


def run_piece1():
    print("=================================================================")
    print("PIECE 1: Zero-Shot Baseline and Answer-Only Loss Mask")
    print("=================================================================")
    
    # 1. Run unit verification
    verify_answer_only_mask()

    # 2. Load model, tokenizer, and emotion task
    print(f"Loading {S.MODEL_NAME} on {S.DEVICE}...")
    model, tok = S.load_model()
    train_ex, test_ex, label_ids = S.load_task("emotion", n_train=600, n_test=500, tokenizer=tok, data_seed=0)

    # 3. Print sample prompt and answer
    sample = train_ex[0]
    print("Dataset Sample Inspection:")
    print(f"  Prompt : {repr(tok.decode(sample['prompt_ids']))}")
    print(f"  Answer : {repr(tok.decode([sample['answer_id']]))} (Label ID: {sample['label']})")

    # 4. Evaluate Zero-Shot Accuracy
    print("\nEvaluating Zero-Shot Baseline on 500 test examples...")
    zero_shot_acc = S.evaluate(model, tok, test_ex, label_ids, batch_size=32)
    chance_acc = 1.0 / len(label_ids)

    print(f"\nZero-Shot Accuracy : {zero_shot_acc:.4f} ({zero_shot_acc * 100:.2f}%)")
    print(f"Random Chance      : {chance_acc:.4f} ({chance_acc * 100:.2f}%)")
    diff = zero_shot_acc - chance_acc
    print(f"Delta over Chance  : {diff:+.4f} ({diff * 100:+.2f}%)\n")

    # 5. Explanatory Statement Required by Lab 06
    print("--- Conceptual Analysis of Loss Formulation ---")
    print("Q: What does the starter's all_tokens_labels train the model to do with the prompt,")
    print("   and why is that not the skill being measured?")
    print("A: The starter's version trains the model to reproduce the conversational grammar and syntax")
    print("   of the prompt text itself, allocating the vast majority of gradient updates to general")
    print("   language modeling rather than task-specific label prediction.")
    print("   This is not the skill being measured because classification requires the model to condition")
    print("   on the completed prompt and allocate highest logit probability specifically to the correct")
    print("   answer token; scoring prompt tokens dilutes the classification signal with unrelated loss.")


if __name__ == "__main__":
    run_piece1()