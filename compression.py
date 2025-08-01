import math
import torch
from torch_dct import dct, idct

class Baseline:
    def __init__(self, **kwargs):
        pass

    def encode(self, x, **kwargs):
        return x, 1
    
    def decode(self, x, y, **kwargs):
        return x.clone().detach()

class VanillaSparsification:
    def __init__(self, ratio, **kwargs) -> None:
        self.ratio = ratio

    def encode(self, x, **kwargs):
        with torch.no_grad():
            k = int(self.ratio * x.size(0))
            _, indices = torch.topk(torch.abs(x), k=k)
            values = x[indices]
        return len(x), values, indices
    
    def decode(self, sz, values, indices, **kwargs):
        with torch.no_grad():
            x = torch.zeros(sz).to(values)
            x[indices] = values
        return x
    
class ConstraintSparsification:
    def __init__(self, ratio, positive, **kwargs) -> None:
        self.ratio = ratio
        self.positive = positive

    def encode(self, x, tilde_gradients, **kwargs):
        with torch.no_grad():
            product = x * tilde_gradients
            if self.positive:
                positive_mask = product >= 0
            else:
                positive_mask = product <= 0
            positive_indices = torch.where(positive_mask)[0]

            k = int(self.ratio * x.size(0))
            x_filtered = x[positive_indices]
            assert len(x_filtered) >= k
            _, topk_indices = torch.topk(torch.abs(x_filtered), k)
            indices = positive_indices[topk_indices]
            values = x[indices]
        return len(x), values, indices
    
    def decode(self, sz, values, indices, **kwargs):
        with torch.no_grad():
            x = torch.zeros(sz).to(values)
            x[indices] = values
        return x


class VanillaQuantization:
    def __init__(self, bit, lower_percentile=1, upper_percentile=99, sample=True, sample_size=10000, **kwargs) -> None:
        self.bit = bit
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.sample = sample
        self.sample_size = sample_size

    def encode(self, x, **kwargs):
        with torch.no_grad():
            if self.sample:
                rand_idx = torch.randint(0, x.numel(), (self.sample_size,), device=x.device)
                x_sample = x.view(-1)[rand_idx]
                min_value = torch.quantile(x_sample, self.lower_percentile/100).item()
                max_value = torch.quantile(x_sample, self.upper_percentile/100).item()
            else:
                min_value = torch.quantile(x, self.lower_percentile/100).item()
                max_value = torch.quantile(x, self.upper_percentile/100).item()
            k = ((2 ** self.bit) - 1) / (max_value - min_value)
            b = -min_value * k
            y = torch.round(k * x + b)
        return y, k, b

    def decode(self, y, k, b):
        with torch.no_grad():
            return (y - b) / k

class ConstraintQuantization:
    def __init__(self, bit, positive, tol=0.45, lower_percentile=1, upper_percentile=99, sample=True, sample_size=10000, **kwargs) -> None:
        self.bit = bit
        self.positive = positive
        self.tol = tol
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.sample = sample
        self.sample_size = sample_size

    def encode(self, x, tilde_gradients, min_value=None, max_value=None, **kwargs):
        with torch.no_grad():
            if self.positive:
                mask = tilde_gradients > 0
            else:
                mask = tilde_gradients < 0

            if self.sample:
                rand_idx = torch.randint(0, x.numel(), (self.sample_size,), device=x.device)
                x_sample = x.view(-1)[rand_idx]
                min_value = torch.quantile(x_sample, self.lower_percentile/100).item()
                max_value = torch.quantile(x_sample, self.upper_percentile/100).item()
            else:
                min_value = torch.quantile(x, self.lower_percentile/100).item()
                max_value = torch.quantile(x, self.upper_percentile/100).item()

            k = ((2 ** self.bit) - 1) / (max_value - min_value)
            b = -min_value * k
            y = k * x + b
            y_rounded = torch.round(y)

            frac = torch.abs(y - y_rounded)
            need_adjust = (
                (mask & (y_rounded < y) | ~mask & (y_rounded > y))
                & (frac >= self.tol)
            )
            
            y = torch.where(
                need_adjust,
                torch.where(mask, torch.ceil(y), torch.floor(y)),
                y_rounded
            )
        return y, k, b

    def decode(self, y, k, b):
        with torch.no_grad():
            return (y - b) / k

class VanillaLowRank:
    def __init__(self, ratio, **kwargs):
        self.ratio = ratio

    def encode(self, x, **kwargs):
        with torch.no_grad():
            n = x.numel()
            side_len = math.ceil(math.sqrt(n))
            num_padding = side_len ** 2 - n
            padded = torch.zeros(side_len ** 2, device=x.device, dtype=x.dtype)
            padded[:n] = x
            matrix = padded.reshape(side_len, side_len)

            rank = int(self.ratio * n / (2 * side_len + 1))
            assert rank > 0

            U, S, V = torch.svd_lowrank(matrix, q=rank)
            Vt = V.T

            # U, S, Vt = torch.linalg.svd(matrix, full_matrices=False)
            # U = U[:, :rank]
            # S = S[:rank]
            # Vt = Vt[:rank, :]
            return U, S, Vt, num_padding
        
    def decode(self, U, S, Vt, num_padding, **kwargs):
        with torch.no_grad():
            reconstructed = U @ torch.diag(S) @ Vt
            flattened = reconstructed.flatten()
            if num_padding > 0:
                flattened = flattened[:-num_padding]
        return flattened
    
class ConstraintLowRank:
    def __init__(self, ratio, positive, factor=2, max_attempt=5, **kwargs):
        self.ratio = ratio
        self.positive = positive
        self.factor = factor
        self.max_attemp = max_attempt

    def encode(self, x, tilde_gradients, **kwargs):
        with torch.no_grad():
            n = x.numel()
            side_len = math.ceil(math.sqrt(n))
            num_padding = side_len ** 2 - n
            padded_x = torch.zeros(side_len ** 2, device=x.device, dtype=x.dtype)
            padded_g = torch.zeros(side_len ** 2, device=x.device, dtype=x.dtype)

            padded_x[:n] = x
            padded_g[:n] = tilde_gradients

            matrix = padded_x.reshape(side_len, side_len)

            rank = int(self.ratio * n / (2 * side_len + 1))
            assert rank > 0
            
            init_factor = self.factor
            for _ in range(self.max_attemp):
                rank_predict = int(rank * init_factor)
                U, S, V = torch.svd_lowrank(matrix, q=rank_predict)
                Vt = V.T

                mask = torch.zeros(rank_predict, dtype=torch.bool)
                for i in range(rank_predict):
                    tmp = torch.dot((S[i] * torch.outer(U[:, i], Vt[i, :])).view(-1), padded_g)
                    if self.positive and tmp >= 0:
                        mask[i] = True
                    elif (not self.positive) and tmp < 0:
                        mask[i] = True
                    if mask.sum() == rank:
                        break

                if mask.sum() == rank:
                    break
                else:
                    init_factor += 1
            
            if mask.sum() < rank:
                remaining_indices = (~mask).nonzero().squeeze(-1)
                needed = rank - mask.sum()
                mask[remaining_indices[:needed]] = True
            
            assert mask.sum() == rank, f"{mask.sum()=}, {rank=}"

            U = U[:, mask]
            S = S[mask]
            Vt = Vt[mask, :]
        return U, S, Vt, num_padding
    
    def decode(self, U, S, Vt, num_padding, **kwargs):
        with torch.no_grad():
            reconstructed = U @ torch.diag(S) @ Vt
            flattened = reconstructed.flatten()
            if num_padding > 0:
                flattened = flattened[:-num_padding]
        return flattened


class QuantileQuantization:
    def __init__(self, bit, offset, asymmetric=False, stats='norm', lower_percentile=1, upper_percentile=99, sample=True, sample_size=10000, **kwargs) -> None:
        self.bit = bit
        self.offset = offset
        if stats == 'norm':
            from scipy.stats import norm
            self.ppf = norm.ppf
        else:
            raise NotImplementedError
        self.quantiles = self.create_normal_map(bit, offset, asymmetric)
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.sample = sample
        self.sample_size = sample_size

    def encode(self, x, **kwargs):
        with torch.no_grad():
            self.quantiles = self.quantiles.to(x.device)
            if self.sample:
                rand_idx = torch.randint(0, x.numel(), (self.sample_size,), device=x.device)
                x_sample = x.view(-1)[rand_idx]
                min_value = torch.quantile(x_sample, self.lower_percentile/100).item()
                max_value = torch.quantile(x_sample, self.upper_percentile/100).item()
            else:
                min_value = torch.quantile(x, self.lower_percentile/100).item()
                max_value = torch.quantile(x, self.upper_percentile/100).item()
            
            max_value = max(abs(max_value), abs(min_value))
            x = x.detach().clone()
            # max_value = x.abs().max().item()
            sz = x.shape
            x_scale = (x / max_value).view(-1)

            positions = torch.searchsorted(self.quantiles, x_scale)
            positions = torch.clamp(positions, 0, len(self.quantiles) - 1)

            left = self.quantiles[torch.clamp(positions - 1, 0)]
            right = self.quantiles[positions]

            closest_index = torch.where(
                (x_scale - left).abs() < (x_scale - right).abs(),
                positions - 1,
                positions
            )

            res = closest_index.view(sz)
        return res, max_value

    def decode(self, x, max_value, **kwargs):
        with torch.no_grad():
            return self.quantiles[x.long()] * max_value

    def create_normal_map(self, bit=4, offset=0.8, asymmetric=False):
        bins = 2 ** bit
        if asymmetric:
            v1 = self.ppf(torch.linspace(offset, 0.5, bins // 2)[:-1]).tolist()[::-1]
            v2 = [0, 0]
            v3 = (-self.ppf(torch.linspace(offset, 0.5, bins // 2)[:-1])).tolist()
            v = v3 + v2 + v1
        else:
            v1 = self.ppf(torch.linspace(offset, 0.5, bins // 2 + 1)[:-1]).tolist()[::-1]
            v2 = [0]
            v3 = (-self.ppf(torch.linspace(offset, 0.5, bins // 2)[:-1])).tolist()
            v = v3 + v2 + v1

        values = torch.Tensor(v)
        values /= values.max()
        return values

class EnhancedQuantileQuantization(QuantileQuantization):
    def __init__(self, bit, offset, tol=0.45, asymmetric=False, stats='norm', lower_percentile=1, upper_percentile=99, sample=True, sample_size=10000, **kwargs) -> None:
        super().__init__(bit, offset, asymmetric, stats, lower_percentile, upper_percentile, sample, sample_size)
        self.tol = tol
   
        
    def encode(self, x, tilde_gradients, **kwargs):
        with torch.no_grad():
            self.quantiles = self.quantiles.to(x.device)
            mask = tilde_gradients > 0

            if self.sample:
                rand_idx = torch.randint(0, x.numel(), (self.sample_size,), device=x.device)
                x_sample = x.view(-1)[rand_idx]
                min_value = torch.quantile(x_sample, self.lower_percentile/100).item()
                max_value = torch.quantile(x_sample, self.upper_percentile/100).item()
            else:
                min_value = torch.quantile(x, self.lower_percentile/100).item()
                max_value = torch.quantile(x, self.upper_percentile/100).item()
            
            max_value = max(abs(max_value), abs(min_value))


            x = x.detach().clone()
            # max_value = x.abs().max().item()
            sz = x.shape
            x_scale = (x / max_value).view(-1)

            positions = torch.searchsorted(self.quantiles, x_scale)
            positions = torch.clamp(positions, 0, len(self.quantiles) - 1)

            left = self.quantiles[torch.clamp(positions - 1, 0)]
            right = self.quantiles[positions]

            closest_index = torch.where(
                (x_scale - left).abs() < (x_scale - right).abs(),
                positions - 1,
                positions
            )
            ratio = (x_scale - left) / (right - left)
            tol_mask = (self.tol < ratio) & (ratio < (1 - self.tol))
            x_recover = self.quantiles[closest_index.long()]
            
            need_adjust = (
                (mask & (x_recover < x_scale) | ~mask & (x_recover > x_scale))
                &
                tol_mask
            )
            closest_index_adjust = torch.where(
                need_adjust,
                torch.where(mask, closest_index + 1, closest_index - 1),
                closest_index
            )
            res = closest_index_adjust.view(sz)
        return res, max_value

class RandTopkSparsification:
    def __init__(self, ratio, k=None, alpha=0.1, **kwargs) -> None:
        self.ratio = ratio
        self.alpha = alpha
        self.k = k

    def encode(self, x, **kwargs):
        with torch.no_grad():
            x = x.detach().clone()
            sz = x.shape
            x = x.view(-1)
            if self.k is None:
                k = int(self.ratio * x.size(0))
            else:
                k = self.k
            _, top_k_indices = torch.topk(x, k)
            N1, N2 = k, x.size(0) - k
            probabilities = torch.ones_like(x)
            probabilities[top_k_indices] = (1 - self.alpha) / N1
            probabilities[torch.where(probabilities == 1)] = self.alpha / N2
            if probabilities.shape[0] > 2^24:
                chunk_size = 1_000_000
                samples = []
                for i in range(0, probabilities.shape[0], chunk_size):
                    chunk = probabilities[i:i+chunk_size]
                    chunk_probs = chunk / chunk.sum()
                    samples.append(torch.multinomial(chunk_probs, k, replacement=True))

                selected_neurons = torch.cat(samples)
            else:
                selected_neurons = torch.multinomial(probabilities, k, replacement=False)
            return sz, x[selected_neurons], selected_neurons
        
    def decode(self, sz, values, indices, **kwargs):
        with torch.no_grad():
            x = torch.zeros(sz).view(-1).to(values.device)
            x[indices] = values
            x = x.view(sz)
        return x

class THCQuantization:
    def __init__(self, bit, lower_percentile=1, upper_percentile=99, sample=True, sample_size=10000, seed=42, **kwargs):
        self.bit = bit
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.sample = sample
        self.sample_size = sample_size
        self.seed = seed

    def encode(self, x, **kwargs):
        with torch.no_grad():
            torch.manual_seed(self.seed)
            rand_sign = torch.randint(0, 2, x.shape, device=x.device) * 2 - 1
            x_trans = x * rand_sign
            if self.sample:
                rand_idx = torch.randint(0, x_trans.numel(), (self.sample_size,), device=x.device)
                x_sample = x_trans.view(-1)[rand_idx]
                min_value = torch.quantile(x_sample, self.lower_percentile/100).item()
                max_value = torch.quantile(x_sample, self.upper_percentile/100).item()
            else:
                min_value = torch.quantile(x_trans, self.lower_percentile/100).item()
                max_value = torch.quantile(x_trans, self.upper_percentile/100).item()
            k = ((2 ** self.bit) - 1) / (max_value - min_value)
            b = -min_value * k
            y = torch.round(k * x_trans + b)
        return y, k, b

    def decode(self, y, k, b, **kwargs):
        with torch.no_grad():
            torch.manual_seed(self.seed)
            x_trans = (y - b) / k
            rand_sign = torch.randint(0, 2, x_trans.shape, device=x_trans.device) * 2 - 1
            x = x_trans * rand_sign
        return x

class EnhancedTHCQuantization:
    def __init__(self, bit, lower_percentile=1, tol=0.45, upper_percentile=99, sample=True, sample_size=10000, seed=42, **kwargs):
        self.bit = bit
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.sample = sample
        self.sample_size = sample_size
        self.tol = tol
        self.seed = seed

    def encode(self, x, tilde_gradients, **kwargs):
        with torch.no_grad():
            mask = tilde_gradients > 0

            torch.manual_seed(self.seed)
            rand_sign = torch.randint(0, 2, x.shape, device=x.device) * 2 - 1
            x_trans = x * rand_sign
            if self.sample:
                rand_idx = torch.randint(0, x_trans.numel(), (self.sample_size,), device=x.device)
                x_sample = x_trans.view(-1)[rand_idx]
                min_value = torch.quantile(x_sample, self.lower_percentile/100).item()
                max_value = torch.quantile(x_sample, self.upper_percentile/100).item()
            else:
                min_value = torch.quantile(x_trans, self.lower_percentile/100).item()
                max_value = torch.quantile(x_trans, self.upper_percentile/100).item()
            k = ((2 ** self.bit) - 1) / (max_value - min_value)
            b = -min_value * k
            y = k * x_trans + b
            y_rounded = torch.round(y)
            frac = torch.abs(y - y_rounded)
            need_adjust = (
                (mask & (y_rounded < y) | ~mask & (y_rounded > y))
                & (frac >= self.tol)
            )
            y = torch.where(
                need_adjust,
                torch.where(mask, torch.ceil(y), torch.floor(y)),
                y_rounded
            )
        return y, k, b

    def decode(self, y, k, b, **kwargs):
        with torch.no_grad():
            torch.manual_seed(self.seed)
            x_trans = (y - b) / k
            rand_sign = torch.randint(0, 2, x_trans.shape, device=x_trans.device) * 2 - 1
            x = x_trans * rand_sign
        return x

class FedTC:
    def __init__(self, ratio, bit, lower_percentile=1, upper_percentile=99, sample=True, sample_size=10000, **kwargs):
        self.ratio = ratio
        self.bit = bit
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.sample = sample
        self.sample_size = sample_size

    def _dct(self, x: torch.Tensor) -> torch.Tensor:
        return dct(x.cpu())

    def _idct(self, x: torch.Tensor) -> torch.Tensor:
        return idct(x.cpu())

    def encode(self, x, **kwargs):
        with torch.no_grad():
            x_t = self._dct(x).to(x.device)
            k = int(self.ratio * x_t.numel())
            _, indices = torch.topk(torch.abs(x_t), k=k)
            values = x_t[indices]
            if self.sample:
                rand_idx = torch.randint(0, values.numel(), (self.sample_size,), device=x.device)
                x_sample = values.view(-1)[rand_idx]
                min_value = torch.quantile(x_sample, self.lower_percentile/100).item()
                max_value = torch.quantile(x_sample, self.upper_percentile/100).item()
            else:
                min_value = torch.quantile(values, self.lower_percentile/100).item()
                max_value = torch.quantile(values, self.upper_percentile/100).item()
            k = ((2 ** self.bit) - 1) / (max_value - min_value)
            b = -min_value * k
            y = torch.round(k * values + b)
        return y, k, b, x.numel(), indices, values

    def decode(self, y, k, b, sz, indices, values, **kwargs):
        with torch.no_grad():
            values = (y - b) / k
            x_t = torch.zeros(sz, device=values.device)
            x_t[indices] = values
            x_recovered = self._idct(x_t).to(x_t.device)
        return x_recovered

class EnhancedFedTC(FedTC):
    def __init__(self, ratio, bit, tol=0.45, lower_percentile=1, upper_percentile=99, sample=True, sample_size=10000, **kwargs):
        self.tol = tol
        super().__init__(ratio, bit, lower_percentile, upper_percentile, sample, sample_size)
    
    def encode(self, x, tilde_gradients, **kwargs):
        with torch.no_grad():
            x_t = self._dct(x).to(x.device)
            tilde_gradients_t = self._dct(tilde_gradients).to(tilde_gradients.device)
            mask_quan = tilde_gradients_t > 0

            k = int(self.ratio * x_t.numel())
            _, indices = torch.topk(torch.abs(x_t), k=k)
            values = x_t[indices]

            mask_quan = mask_quan[indices]
            if self.sample:
                rand_idx = torch.randint(0, values.numel(), (self.sample_size,), device=x.device)
                x_sample = values.view(-1)[rand_idx]
                min_value = torch.quantile(x_sample, self.lower_percentile/100).item()
                max_value = torch.quantile(x_sample, self.upper_percentile/100).item()
            else:
                min_value = torch.quantile(values, self.lower_percentile/100).item()
                max_value = torch.quantile(values, self.upper_percentile/100).item()
            k = ((2 ** self.bit) - 1) / (max_value - min_value)
            b = -min_value * k
            y = k * values + b
            y_rounded = torch.round(y)
            frac = torch.abs(y - y_rounded)
            need_adjust = (
                (mask_quan & (y_rounded < y) | ~mask_quan & (y_rounded > y))
                & (frac >= self.tol)
            )
            y = torch.where(
                need_adjust,
                torch.where(mask_quan, torch.ceil(y), torch.floor(y)),
                y_rounded
            )
        return y, k, b, x.numel(), indices, values