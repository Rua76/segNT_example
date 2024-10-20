import haiku as hk
import jax
import jax.numpy as jnp
from nucleotide_transformer.pretrained import get_pretrained_model

# Get pretrained model
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
parameters, forward_fn, tokenizer, config = get_pretrained_model(
        model_name="500M_human_ref",
        embeddings_layers_to_save=(20,),
        max_positions=32,
    )
