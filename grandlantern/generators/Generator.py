from grandlantern.backend import np
from grandlantern.matrix import Matrix
from grandlantern.dataiterators import TOKEN_BOS, TOKEN_EOS

class BaseGenerator():
    def generate(self, model, X, mask=None):
        pass

    def __str__(self):
        return "BaseGenerator"


class AutoRegressiveEncoderGenerator(BaseGenerator):
    token_bos: int
    token_eos: int
    max_len: int
    strategy: str
    temperature: float

    def __init__(self, token_bos = TOKEN_BOS, token_eos = TOKEN_EOS, max_len = 1000, strategy = 'greedy', temperature = 1.0):
        self.token_bos = token_bos
        self.token_eos = token_eos
        self.max_len = max_len
        self.strategy = strategy
        self.temperature = temperature

    def generate(self, model, X, src_mask=None):
        # Encode
        context = model.encode(X, src_mask, train_mode=False)

        # BOS sequence
        n_batches = X.shape[0]
        seq = Matrix(np.full(shape=(X.shape[0], 1), fill_value=self.token_bos), require_grad=False)

        # Autogenerator
        for i in range(self.max_len):
            current_logits = model.decode_step(seq, context, src_mask=src_mask, train_mode=False)
            current_token = self._pick_(current_logits).reshape((n_batches, 1))
            seq = Matrix.concat([seq, current_token], axis=1)
            if self.token_eos is not None and np.all(current_token.value == self.token_eos):
                break
        return seq

    def _pick_(self, logits):
        if self.strategy == "greedy":
            return Matrix.argmax(logits, axis=-1)
        elif self.strategy == "sample":
            probs = Matrix.exp(logits / self.temperature)
            denom = Matrix.sum(probs, axis=-1, keepdims=True)
            probs = probs / denom
            idx = Matrix(np.array([np.random.choice(probs.shape[-1], p=p) for p in probs.value]))
            return idx
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

