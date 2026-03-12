Base model (n=12):
  total=0.202  type=0.167  comp_f1=0.083  layer_iou=0.083  facts=0.153  eff=0.925
  d=1: total=0.088  type=0.000  comp_f1=0.000 (n=3)
  d=2: total=0.114  type=0.000  comp_f1=0.000 (n=3)
  d=3: total=0.091  type=0.000  comp_f1=0.000 (n=3)
  d=4: total=0.516  type=0.667  comp_f1=0.333 (n=3)

Trained LoRA (n=12):
  total=0.800  type=0.583  comp_f1=0.750  layer_iou=1.000  facts=0.889  eff=1.019
  d=1: total=0.794  type=0.667  comp_f1=0.667 (n=3)
  d=2: total=0.794  type=0.667  comp_f1=0.667 (n=3)
  d=3: total=0.794  type=0.667  comp_f1=0.667 (n=3)
  d=4: total=0.819  type=0.333  comp_f1=1.000 (n=3)

Delta (trained - base):
  total: +0.598
  type_accuracy: +0.417
  component_f1: +0.667
  layer_iou: +0.917
  key_fact_score: +0.736