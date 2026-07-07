# NEAT (Neural Attention Fields) vendored agent package.
#
# Contents:
#   config.py                 - GlobalConfig (vendored verbatim from autonomousvision/neat)
#   architectures/            - AttentionField encoder + attention-field decoder + PID (verbatim)
#   modules.py                - pipeline glue: preprocessing, model runner, controller
#   fetch_weights.sh          - downloads best_encoder.pth / best_decoder.pth / args.txt
#
# Nothing heavy is imported at package import time; torch / torchvision / the
# architecture modules are imported lazily inside the pipeline modules' run().
