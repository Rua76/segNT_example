import haiku as hk
import jax
import jax.numpy as jnp
from nucleotide_transformer.pretrained import get_pretrained_segment_nt_model

# Initialize CPU as default JAX device. This makes the code robust to memory leakage on
# the devices.
jax.config.update("jax_platform_name", "cpu")

backend = "cpu"
devices = jax.devices(backend)
num_devices = len(devices)
print(f"Devices found: {devices}")

# The number of DNA tokens (excluding the CLS token prepended) needs to be dividible by
# 2 to the power of the number of downsampling block, i.e 4.
max_num_nucleotides = 8

assert max_num_nucleotides % 4 == 0, (
    "The number of DNA tokens (excluding the CLS token prepended) needs to be dividible by"
     "2 to the power of the number of downsampling block, i.e 4.")

parameters, forward_fn, tokenizer, config = get_pretrained_segment_nt_model(
    model_name="segment_nt",
    embeddings_layers_to_save=(29,),
    attention_maps_to_save=((1, 4), (7, 10)),
    max_positions=max_num_nucleotides + 1,
)
forward_fn = hk.transform(forward_fn)
apply_fn = jax.pmap(forward_fn.apply, devices=devices, donate_argnums=(0,))


# Get data and tokenize it
sequences = ["ATTCCGATTCCGATTCCAACGGATTATTCCGATTAACCGATTCCAATT", "ATTTCTCTCTCTCTCTGAGATCGATGATTTCTCTCTCATCGAACTATG"]
tokens_ids = [b[1] for b in tokenizer.batch_tokenize(sequences)]
tokens_str = [b[0] for b in tokenizer.batch_tokenize(sequences)]
tokens = jnp.asarray(tokens_ids, dtype=jnp.int32)

random_key = jax.random.PRNGKey(seed=0)
keys = jax.device_put_replicated(random_key, devices=devices)
parameters = jax.device_put_replicated(parameters, devices=devices)
tokens = jax.device_put_replicated(tokens, devices=devices)

# Infer on the sequence
outs = apply_fn(parameters, keys, tokens)
# Obtain the logits over the genomic features
logits = outs["logits"]
# Transform them in probabilities
probabilities = jnp.asarray(jax.nn.softmax(logits, axis=-1))[...,-1]
print(f"Probabilities shape: {probabilities.shape}")

print(f"Features inferred: {config.features}")

# Get probabilities associated with intron
idx_intron = config.features.index("intron")
probabilities_intron = probabilities[..., idx_intron]
print(f"Intron probabilities shape: {probabilities_intron.shape}")