# Collection for network inputs. Used by `Trainer` class for retrieving all
# data input placeholders.
INPUTS = 'inputs'

# Collection for network targets. Used by `Trainer` class for retrieving all
# targets (labels) placeholders.
TARGETS = 'targets'

# Collection for network train ops. Used by `Trainer` class for retrieving all
# optimization processes.
TRAIN_OPS = 'trainops'

# Collection to retrieve layers variables. Variables are stored according to
# the following pattern: /tf.GraphKeys.LAYER_VARIABLES/layer_name (so there
# will have as many collections as layers with variables).
LAYER_VARIABLES = 'layer_variables'

# Collection to store all returned tensors for every layer
LAYER_TENSOR = 'layer_tensor'

# Collection to store all variables that will be restored
EXCL_RESTORE_VARS = 'restore_variables'

# Collection to store the default graph configuration
GRAPH_CONFIG = 'graph_config'

# Collection to store all input variable data preprocessing
DATA_PREP = 'data_preprocessing'

# Collection to store all input variable data preprocessing
DATA_AUG = 'data_augmentation'

# Collection to store all custom learning rate variable
LR_VARIABLES = 'lr_variables'