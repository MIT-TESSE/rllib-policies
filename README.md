# rllib-policies

Repo to build RLlib policies.


# Implementing new policies 

An policy contains one or more networks. Each network processes a set of specified observations (e.g., images, graphs, poses). 

To build a new policy: 

1. Inheret from [NetworkBase](https://github.mit.edu/aiia-suas-disaster-response/rllib-policies/blob/develop/src/rllib_policies/base.py#L10) to define a network in PyTorch. 

2. Inheret from [RllibPolicy](https://github.mit.edu/aiia-suas-disaster-response/rllib-policies/blob/develop/src/rllib_policies/base.py#L96) to define a polciy. 
   - Any custom networks must be initialized in [`init_nets`](https://github.mit.edu/aiia-suas-disaster-response/rllib-policies/blob/develop/src/rllib_policies/base.py#L142).
   - There are predefined actor-critic policies [with](https://github.mit.edu/aiia-suas-disaster-response/rllib-policies/blob/develop/src/rllib_policies/actor_critic.py#L61) and [without](https://github.mit.edu/aiia-suas-disaster-response/rllib-policies/blob/develop/src/rllib_policies/actor_critic.py#L15) a recurrent network
 
See [NatureCNNRNNActorCritic](https://github.mit.edu/aiia-suas-disaster-response/rllib-policies/blob/develop/src/rllib_policies/vision.py#L92) for an example. 

# Examples

To use the CNN-LSTM Actor-Critic policy defined [here](https://github.mit.edu/aiia-suas-disaster-response/rllib_policies/blob/develop/src/rllib_policies/vision.py#L92)

```python

from rllib_policies.vision import NatureCNNRNNActorCritic
ModelCatalog.register_custom_model("nature_cnn_rnn", NatureCNNRNNActorCritic)

model = {
    "custom_model": "nature_cnn_rnn",
    "max_seq_len": 200,
    # keywords to custom model
    "custom_model_config": {
        "rnn_type": "LSTM",
        "hidden_size": 512,
        # cnn specific args
        "fields": ["RGB_LEFT", "DEPTH"],  # keys in observation dictionary 
        "cnn_shape_chw": [4, 192, 256],
    },
}

```

# Disclaimer


DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Under Secretary of Defense for Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Under Secretary of Defense for Research and Engineering.

(c) 2022 Massachusetts Institute of Technology.

MIT Proprietary, Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)

The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.