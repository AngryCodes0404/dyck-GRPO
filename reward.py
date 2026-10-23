# dyck_reward.py

BRACKET_PAIRS = {
    "(": ")",
    "[": "]",
    "{": "}",
}
OPEN_TO_CLOSE = BRACKET_PAIRS
CLOSE_TO_OPEN = {v: k for k, v in BRACKET_PAIRS.items()}
OPEN_SET = set(OPEN_TO_CLOSE.keys())
CLOSE_SET = set(CLOSE_TO_OPEN.keys())


def is_valid_dyck(seq: str) -> bool:

    stack = []
    for ch in seq:
        if ch in OPEN_SET:
            stack.append(ch)
        elif ch in CLOSE_SET:
            if not stack:
                return False
            top = stack.pop()
            if OPEN_TO_CLOSE[top] != ch:
                return False
    return len(stack) == 0


def extract_brackets(seq: str) -> str:
    return "".join(ch for ch in seq if ch in OPEN_SET or ch in CLOSE_SET)


def sequence_similarity(a: str, b: str) -> float:

    a_b = extract_brackets(a)
    b_b = extract_brackets(b)
    # simple dynamic programming edit distance
    na, nb = len(a_b), len(b_b)
    if na == 0 and nb == 0:
        return 1.0
    dp = [[0] * (nb + 1) for _ in range(na + 1)]
    for i in range(na + 1):
        dp[i][0] = i
    for j in range(nb + 1):
        dp[0][j] = j
    for i in range(1, na + 1):
        for j in range(1, nb + 1):
            cost = 0 if a_b[i - 1] == b_b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # deletion
                dp[i][j - 1] + 1,  # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )
    dist = dp[na][nb]
    max_len = max(1, max(na, nb))
    return 1.0 - (dist / max_len)


def dyck_reward(
    prompt: str,
    generated: str,
    target_answer: str,
    minimal_mode: bool = True,
) -> float:

    lines = [ln.strip() for ln in generated.splitlines() if ln.strip()]
    if not lines:
        return -0.5  # empty output
    seq = lines[0]

    reward = 0.0

    if len(seq) > 0 and seq in prompt:
        reward -= 0.5

    if seq == target_answer:
        reward += 3.0

    if is_valid_dyck(seq):
        reward += 1.0
    else:
        reward -= 0.5

    sim = sequence_similarity(seq, target_answer)
    reward += sim  # between 0 and 1

    if minimal_mode:
        len_diff = len(seq) - len(target_answer)
        if len_diff > 5:
            reward -= min(1.0, len_diff / 20.0)

    if reward > 5.0:
        reward = 5.0
    if reward < -1.0:
        reward = -1.0

    return float(reward)
