# Training code in the manuscript

Use availability statements after the training source is published with the
manuscript release. This local addition does not publish the code or create a
release, DOI, or training-data deposit.

## Code availability

The cryoID2 inference pipeline and recovered U-Net training implementation are
available at https://github.com/t00shadow/cryoID_v2. Training and historical
dataset-preparation scripts are provided in the `training/` directory with
separate environment instructions and source provenance. Routine use of cryoID2
relies on pretrained model weights and does not require installation or execution
of the training workflow.

Add the exact release tag, commit, and archived DOI when available. Do not present
the recovered implementation as complete reproduction of all training runs:
the exact historical data splits and checkpoint-to-run mapping still require
confirmation, and the recovered preparation helper does not provide the later
six-letter label conversion.

## Methods

The recovered U-Net training implementation, including the six-letter
residue-model variant, is provided separately from the inference pipeline in the
repository's `training/` directory. The accompanying documentation records its
source revision, configuration, data requirements, and limitations.

The six-letter training script defines class-weighted cross-entropy loss, Adam
with an initial learning rate of 0.0001, and a ReduceLROnPlateau scheduler that
monitors average validation loss (reduction factor 0.5; patience 5 epochs), with
separate early-stopping patience of 15 epochs. These are verified source-code
settings. Describe them as the settings used for the manuscript's released
models only after confirming the corresponding training records.

## Existing revision notes

The October 28 revision notes describe the files available during that review.
The training implementation was subsequently recovered from another branch;
the remaining missing records should now be distinguished from missing code.
The original environment, exact split assignments, six-letter label-preparation
implementation, and mapping of checkpoints to runs remain unresolved here.
