import haiku as hk
import jax
import jax.numpy as jnp
from nucleotide_transformer.pretrained import get_pretrained_model
from nucleotide_transformer.pretrained import get_pretrained_segment_nt_model

# Get pretrained NT Transformer model
supported_model=  [
        "500M_human_ref",
        "500M_1000G",
        "2B5_1000G",
        "2B5_multi_species",
        "50M_multi_species_v2",
        "100M_multi_species_v2",
        "250M_multi_species_v2",
        "500M_multi_species_v2",
        "1B_agro_nt",
    ]

model_to_load = "500M_human_ref" # change model name here to load different model
parameters, forward_fn, tokenizer, config = get_pretrained_model(
        model_name=model_to_load,
        embeddings_layers_to_save=(20,),
        max_positions=32,
    )

# The number of DNA tokens (excluding the CLS token prepended) needs to be dividible by
# 2 to the power of the number of downsampling block, i.e 4.
max_num_nucleotides = 8
supported_seg_model=  [
        "segment_nt",
        "segment_nt_multi_species",
    ]

model_to_load_seg = "segment_nt" # change model name here to load different model
parameters, forward_fn, tokenizer, config = get_pretrained_segment_nt_model(
    model_name=model_to_load_seg,
    embeddings_layers_to_save=(29,),
    attention_maps_to_save=((1, 4), (7, 10)),
    max_positions=max_num_nucleotides + 1,
)